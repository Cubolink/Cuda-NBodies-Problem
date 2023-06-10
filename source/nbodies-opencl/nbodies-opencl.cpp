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
#include "particle-timer.h"

struct float3
{
    float x;
    float y;
    float z;
};

// Number of particles to be loaded and rendered from file
#define NUM_BODIES 16384

// Simulation parameters
float scaleFactor = 1.5f;

// Simulation data storage loaded from file
float3* dataPositions = nullptr;
float3* dataVelocities = nullptr;
float* dataMasses = nullptr;

// Device side data buffers
cl::Buffer dPositions;
cl::Buffer dVelocities;
cl::Buffer dFuturePositions;
cl::Buffer dFutureVelocities;
cl::Buffer dMasses;
cl::Buffer dVBO;
cl::BufferGL dGLVBO;

// OpenCL stuff
cl::CommandQueue queue;
cl::Program program;
cl::Kernel nBodiesKernel;
cl::Context context;

// GL drawing attributes
GLuint VBO = 0;
ParticleRenderer* renderer = nullptr;
float	spriteSize = scaleFactor * 0.25f;

ParticleTimer* particleTimer;

// Controller
Controller* controller = new Controller(scaleFactor, 720.0f, 480.0f);

// OpenCL work-group size
#define GROUP_SIZE 256
// OpenCL number of groups
int clNumGroups;
// Number of particles our kernel will work after padding, if any
int clNumBodies;


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

void initOpenCL() {
    // Round up in case NUM_BODIES is not a multiple of GROUP_SIZE
    clNumGroups = (NUM_BODIES + GROUP_SIZE - 1) / GROUP_SIZE;

    // Number of particles. If clNumGroups was rounded, then there's padding and clNumBodies > NUM_BODIES
    clNumBodies = clNumGroups * GROUP_SIZE;
    int paddedBodies = (clNumBodies - NUM_BODIES);

    particleTimer = new ParticleTimer(clNumBodies);

    // create a context
    cl::Platform clPlatform = cl::Platform::getDefault();
    cl_context_properties properties[] = {
            CL_GL_CONTEXT_KHR, (cl_context_properties) wglGetCurrentContext(),
            CL_WGL_HDC_KHR, (cl_context_properties) wglGetCurrentDC(),
            CL_CONTEXT_PLATFORM, (cl_context_properties) clPlatform(),
            0
    };
    context = cl::Context(DEVICE, properties);

    // get the command queue
    queue = cl::CommandQueue(context);

    // load in kernel source, creating a program object for the context
    program = cl::Program(context, load_program("../../../source/nbodies-opencl/kernel.cl"), true);

    // create the kernel functor
    nBodiesKernel = cl::Kernel(program, "nBodiesKernel");

    // Init device data, copying data from host for positions, velocities and masses
    dPositions = cl::Buffer(context, CL_MEM_READ_WRITE, clNumBodies*sizeof(float3));
    dVelocities = cl::Buffer(context, CL_MEM_READ_WRITE, clNumBodies*sizeof(float3));
    dFuturePositions = cl::Buffer(context, CL_MEM_READ_WRITE, clNumBodies*sizeof(float3));
    dFutureVelocities = cl::Buffer(context, CL_MEM_READ_WRITE, clNumBodies*sizeof(float3));
    dMasses = cl::Buffer(context, CL_MEM_READ_WRITE, clNumBodies*sizeof(float));
    dVBO = cl::Buffer(context, CL_MEM_WRITE_ONLY, 8*sizeof(float) * clNumBodies);
    dGLVBO = cl::BufferGL(context, CL_MEM_READ_WRITE, VBO);

    queue.enqueueWriteBuffer(dPositions, CL_TRUE, 0, NUM_BODIES*sizeof(float3), dataPositions);
    queue.enqueueWriteBuffer(dVelocities, CL_TRUE, 0, NUM_BODIES*sizeof(float3), dataVelocities);
    queue.enqueueWriteBuffer(dMasses, CL_TRUE, 0, NUM_BODIES*sizeof(float), dataMasses);
    queue.enqueueFillBuffer(dPositions, 0, NUM_BODIES*sizeof(float3), paddedBodies * sizeof(float3));
    queue.enqueueFillBuffer(dVelocities, 0, NUM_BODIES*sizeof(float3), paddedBodies * sizeof(float3));
    queue.enqueueFillBuffer(dMasses, 0, NUM_BODIES*sizeof(float), paddedBodies * sizeof(float));
    queue.finish();
    //dPositions = cl::Buffer(context, dataPositions, dataPositions+3*NUM_BODIES, false);
    //dVelocities = cl::Buffer(context, dataVelocities, dataVelocities+3*NUM_BODIES, false);
    //dMasses = cl::Buffer(context, dataMasses, dataMasses+NUM_BODIES, true);

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
    // Prepare the kernel
    cl::NDRange global(clNumBodies);  // Total number of work items
    cl::NDRange local(GROUP_SIZE);  // Work items in each work-group

    nBodiesKernel.setArg(0, dVBO);
    nBodiesKernel.setArg(1, dPositions);
    nBodiesKernel.setArg(2, dVelocities);
    nBodiesKernel.setArg(3, dFuturePositions);
    nBodiesKernel.setArg(4, dFutureVelocities);
    nBodiesKernel.setArg(5, dMasses);
    nBodiesKernel.setArg(6, GROUP_SIZE*sizeof(cl_float4), nullptr);  // tileData
    nBodiesKernel.setArg(7, clNumBodies);
    queue.enqueueNDRangeKernel(nBodiesKernel, cl::NullRange, global, local);
    // Start timer iteration and run the kernel
    particleTimer->startIteration();
    nBodiesKernel();
    queue.finish();
    particleTimer->endIteration();

    // Update positions and velocities for next iteration
    queue.enqueueCopyBuffer(dFuturePositions, dPositions, 0, 0, clNumBodies * 3 * sizeof(float));
    queue.enqueueCopyBuffer(dFutureVelocities, dVelocities, 0, 0, clNumBodies * 3 * sizeof(float));

    /// Update the VBO data
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glFinish();
    std::vector<cl::Memory> objs;
    objs.clear();
    objs.push_back(dGLVBO);
    // Flush opengl commands and wait for object acquisition
    if (queue.enqueueAcquireGLObjects(&objs, nullptr, nullptr) != CL_SUCCESS) {
        std::cout << "Failed acquiring GL object." << std::endl;
        exit(248);
    }
    // Only take the NUM_BODIES <= clNumBodies, ignoring the padded data, so this takes two copies instead of one
    queue.enqueueCopyBuffer(dVBO, dGLVBO, 0, 0, 4 * sizeof(float) * NUM_BODIES);
    queue.enqueueCopyBuffer(dVBO, dGLVBO, clNumBodies * 4 * sizeof(float), NUM_BODIES * 4 * sizeof(float), NUM_BODIES * 4 * sizeof(float));
    if (queue.enqueueReleaseGLObjects(&objs) != CL_SUCCESS) {
        std::cout << "Failed releasing GL object." << std::endl;
        exit(247);
    }
    queue.finish();

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
        case 'c':
            std::cout << "Exporting" << std::endl;
            particleTimer->exportData("data/");
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
    // Fill host containers
    dataPositions = new float3[NUM_BODIES];
    dataVelocities = new float3[NUM_BODIES];
    dataMasses = new float[NUM_BODIES];
    loadData("../../../data/dubinski.tab", NUM_BODIES, (float*) dataPositions, (float*) dataVelocities, dataMasses, scaleFactor);

    // Crete app window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(720, 480);
    char windowTitle[256];
    sprintf(windowTitle, "OpenCL Galaxy Simulation (%d bodies)", NUM_BODIES);
    glutCreateWindow(windowTitle);

    // OpenGL setup
    initGL();
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // OpenCL setup
    initOpenCL();

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
    
    delete renderer;
    delete dataPositions;
    delete dataVelocities;
    delete dataMasses;

    return 0;
}