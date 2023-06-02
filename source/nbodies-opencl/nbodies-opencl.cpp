//
// Created by Cubolink on 31-05-2023.
//

#include <iostream>
#include <fstream>
#include <vector>

#include <GL/glew.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#define __CL_ENABLE_EXCEPTIONS

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "controller.h"
#include "data-loader.h"
#include "framerate.h"
#include "particle-renderer.h"

struct float3
{
    float x;
    float y;
    float z;
};

// Number of particles to be rendered
#define NUM_BODIES 16384

// Simulation parameters
float scaleFactor = 1.5f;

// Simulation data storage
float3* dataPositions = nullptr;
float3* dataVelocities = nullptr;
float* dataMasses = nullptr;

cl::Buffer dPositions; // Device side particles positions
cl::Buffer dVelocities; // Device side particles velocities
cl::Buffer dMasses; // Device side particles masses

cl::CommandQueue queue;
cl::Program program;
cl::Kernel nBodiesKernel;

// GL drawing attributes
GLuint VBO = 0;
ParticleRenderer* renderer = nullptr;
float	spriteSize = scaleFactor * 0.25f;

// Controller
Controller* controller = new Controller(scaleFactor, 720.0f, 480.0f);

// OpenCL work-group size
#define GROUP_SIZE 256

// Clamp macro
#define LIMIT(x,min,max) { if ((x)>(max)) (x)=(max); if ((x)<(min)) (x)=(min); }

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_GPU
#endif

std::string load_program(const std::string& input) {
    std::ifstream stream(input.c_str());
    if(!stream.is_open()) {
        std::cout << "Cannot open the file: " << input << std::endl;
        exit(1);
    }
    return {std::istreambuf_iterator<char>(stream),
            (std::istreambuf_iterator<char>())};

}

// Creates the VBO and binds it to a CUDA resource
void createVBO(GLuint* vbo)
{
    // Create vertex buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // Initialize vertex buffer object
    glBufferData(GL_ARRAY_BUFFER, NUM_BODIES * 8 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Deletes the VBO and unbinds it from the CUDA resource
void deleteVBO(GLuint* vbo)
{
    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

// Initializes OpenGL
void initGL()
{
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 "
                         "GL_VERSION_1_5 "
                         "GL_ARB_multitexture "
                         "GL_ARB_vertex_buffer_object"))
    {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(-1);
    }

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0, 0.0, 0.0, 1.0);

    // Particle renderer initialization
    createVBO((GLuint*) &VBO);
    renderer = new ParticleRenderer(NUM_BODIES);
    renderer->setVBO(VBO);
    renderer->setSpriteSize(0.4f);
    renderer->setShaders("../../../data/sprite.vert", "../../../data/sprite.frag");
}

void runSimulation() {  // runOpenCl
    int numGroups = (NUM_BODIES + GROUP_SIZE - 1) / GROUP_SIZE;

    cl::NDRange global(GROUP_SIZE * numGroups);  // Total number of work items
    cl::NDRange local(GROUP_SIZE);  // Work items in each work-group

    nBodiesKernel.setArg(0, dPositions);
    nBodiesKernel.setArg(1, dVelocities);
    nBodiesKernel.setArg(2, dMasses);
    nBodiesKernel.setArg(3, NUM_BODIES);
    queue.enqueueNDRangeKernel(nBodiesKernel, cl::NullRange, global, local);
    nBodiesKernel();
    queue.finish();

    int nBodies = NUM_BODIES;
    // For each body, updates its position and velocity
    // ...

    // New VBO data
    auto vboData = new float[nBodies * 8];

    for (int i = 0; i < nBodies; i++)
    {
        int pIdx = i * 4;
        int vIdx = pIdx + nBodies * 4;

        vboData[pIdx] = dataPositions[i].x;
        vboData[pIdx + 1] = dataPositions[i].y;
        vboData[pIdx + 2] = dataPositions[i].z;
        vboData[pIdx + 3] = 1.f;

        vboData[vIdx] = dataVelocities[i].x;
        vboData[vIdx + 1] = dataVelocities[i].y;
        vboData[vIdx + 2] = dataVelocities[i].z;
        vboData[vIdx + 3] = 1.f;
    }

    // Update the VBO data
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, nBodies * 8 * sizeof(float), vboData);
}

void display()
{
    runSimulation();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // View transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    controller->updateCameraProperties();
    float* camera_trans_lag = controller->getCameraTransLag();
    float* camera_rot_lag = controller->getCameraRotLag();

    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

    renderer->display();

    framerateUpdate();

    glutSwapBuffers();

    glutReportErrors();
}


// Reshape callback
void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 100000.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    spriteSize *= (float) w / controller->getScreenWidth();

    controller->setScreenSize(w, h);
}

// Mouse pressed button callback
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        controller->setButtonState(button + 1);
    else if (state == GLUT_UP)
        controller->setButtonState(0);

    controller->setCameraOxOy(x, y);

    glutPostRedisplay();
}

// Mouse motion when pressed button callback
void motion(int x, int y)
{
    controller->cameraMotion(x, y);
    glutPostRedisplay();
}

// Keyboard pressed key callback
void key(unsigned char key, int x, int y)
{
    switch (key)
    {
        case '\033':
        case 'q':
            exit(0);
            break;
        default:
            break;
    }
    glutPostRedisplay();
}

// Idle callback, mandatory to update rendering
void idle()
{
    glutPostRedisplay();
}


int main(int argc, char** argv)
{
    // Declare host containers
    std::cout << "Hello OpenCL project!" << std::endl;

    // Fill host containers
    dataPositions = new float3[NUM_BODIES];
    dataVelocities = new float3[NUM_BODIES];
    dataMasses = new float[NUM_BODIES];
    loadData("../../../data/dubinski.tab", NUM_BODIES, (float*) dataPositions, (float*) dataVelocities, dataMasses, scaleFactor);

    /// Init OpenCL

    // create a context
    cl::Context context(DEVICE);

    // load in kernel source, creating a program object for the context
    program = cl::Program(context, load_program("../../../source/nbodies-opencl/kernel.cl"), true);
    // get the command queue
    queue = cl::CommandQueue(context);

    // create the kernel functor
    nBodiesKernel = cl::Kernel(program, "nBodiesKernel");

    // copy data to device
    dPositions = cl::Buffer(context, dataPositions, dataPositions+3*NUM_BODIES, true);
    dVelocities = cl::Buffer(context, dataVelocities, dataVelocities+3*NUM_BODIES, true);
    dMasses = cl::Buffer(context, dataMasses, dataMasses+NUM_BODIES, true);

    // allocate results container
    // skipped due writing into the same array

    // Crete app window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(720, 480);
    char windowTitle[256];
    sprintf(windowTitle, "OpenCL Galaxy Simulation (%d bodies)", NUM_BODIES);
    glutCreateWindow(windowTitle);

    // OpenGL setup
    initGL();

    // GL callback functions
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(key);
    glutIdleFunc(idle);

    framerateTitle(windowTitle);

    // Start main loop
    glutMainLoop();

    if (renderer)
        delete renderer;

    return 0;
}