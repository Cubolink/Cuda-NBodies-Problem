/*
 * This code was modified from the NVIDIA CUDA examples
 * S. James Lee, 2008 Fall
 */

#include <GL/glew.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <param/paramgl.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <assert.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "controller.h"
#include "data-loader.h"
#include "framerate.h"
#include "particle-renderer.h"

// Number of particles to be rendered
#define NUM_BODIES 16384

// Simulation parameters
float scaleFactor = 1.5f;

// Simulation data storage
float* dataPositions = nullptr;
float* dataVelocities = nullptr;
float* dataMasses = nullptr;
GLuint vbo = 0;
float* dPositions = nullptr; // Device side particles positions
float* dVelocities = nullptr; // Device side particles velocities
float* dMasses = nullptr; // Device side particles masses

// GL drawing attributes
ParticleRenderer* renderer = nullptr;
float	spriteSize = scaleFactor * 0.25f;

// Controller
Controller* controller = new Controller(scaleFactor, 720.0f, 480.0f);

// Cuda parameters
int threadsPerBlock = 256;

// Clamp macro
#define LIMIT(x,min,max) { if ((x)>(max)) (x)=(max); if ((x)<(min)) (x)=(min); }

// Forward declarations
void initCUDA(int bodies);
void initGL(void);
void runCuda(void);
void display(void);
void reshape(int w, int h);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void key(unsigned char key, int x, int y);
void idle(void);
void createVBO(GLuint* vbo);
void deleteVBO(GLuint* vbo);

// Initialize CUDA data
void initCUDA(int bodies)
{
	// Device particles data
	cudaMalloc((void**) &dPositions, 3 * bodies * sizeof(float));
	cudaMalloc((void**) &dVelocities, 3 * bodies * sizeof(float));
	cudaMalloc((void**) &dMasses, bodies * sizeof(float));

	// Copy initial values to GPU memory
	cudaMemcpy(dPositions, dataPositions, 3 * bodies * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dVelocities, dataVelocities, 3 * bodies * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dMasses, dataMasses, 3 * bodies * sizeof(float), cudaMemcpyHostToDevice);
}

// Initializes OpenGL
void initGL(void)
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
	createVBO((GLuint*) &vbo);
	renderer = new ParticleRenderer(NUM_BODIES);
	renderer->setVBO(vbo);
	renderer->setSpriteSize(0.4f);
	renderer->setShaders("../../../data/sprite.vert", "../../../data/sprite.frag");
}

// ========================================================================
// Start CUDA loop code
// ========================================================================

__global__ void nBodiesKernel(float4* pvbo, float3* positions, float3* velocities, float* masses)
{	
	// Index of my body	
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	int positionIndex = i;
	int velocityIndex = gridDim.x * blockDim.x + positionIndex;
		
	float3 position = positions[i];
	float3 velocity = velocities[i];

	// Update VBO
	pvbo[positionIndex] = make_float4(position.x, position.y, position.z, 1.0f);
	pvbo[velocityIndex] = make_float4(velocity.x, velocity.y, velocity.z, 1.0f);
}

void runCuda(void)
{
	// Map OpenGL vertex buffer object for writing from CUDA
	float4 *dptr;
	cudaGLMapBufferObject((void**) &dptr, vbo);

	int blockSize = 256; // Blocks of size 16 x 16

  // Round up in case N is not a multiple of blockSize
  int numBlocks = (NUM_BODIES + blockSize - 1) / blockSize;

	// Run the kernel
	nBodiesKernel<<<numBlocks, blockSize>>>(dptr, (float3*) dPositions, (float3*) dVelocities, dMasses);

	// Unmap vertex buffer object
	cudaGLUnmapBufferObject(vbo);
}

// ========================================================================
// End CUDA loop code
// ========================================================================

// Display function called in main loop
void display(void)
{
	// Update simulation
	runCuda();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  

	// View transform
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	controller->updateCameraProperties();

	float* cameraTransLag = controller->getCameraTransLag();
	float* cameraRotLag = controller->getCameraRotLag();

	glTranslatef(cameraTransLag[0], cameraTransLag[1], cameraTransLag[2]);
	glRotatef(cameraRotLag[0], 1.0, 0.0, 0.0);
	glRotatef(cameraRotLag[1], 0.0, 1.0, 0.0);
	
	// Render bodies
	renderer->setSpriteSize(spriteSize);
	renderer->display();
	
	// Update FPS
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
		case '=': // Increase sprite size
			spriteSize += scaleFactor*0.02f;
			LIMIT(spriteSize, 0.1f, scaleFactor*2.0f);
			break;
		case '-': // Decrease sprite size
			spriteSize -= scaleFactor*0.02f;
			LIMIT(spriteSize, 0.1f, scaleFactor*2.0f);
			break;
	}
	glutPostRedisplay();
}

// Idle callback, mandatory to update rendering
void idle(void)
{
	glutPostRedisplay();
}

// Creates the VBO and binds it to a CUDA resource
void createVBO(GLuint* vbo)
{
	// Create vertex buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// Initialize vertex buffer object
	glBufferData(GL_ARRAY_BUFFER, NUM_BODIES * 8 * sizeof(float), 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Register buffer object with CUDA
	cudaGLRegisterBufferObject(*vbo);
}

// Deletes the VBO and unbinds it from the CUDA resource
void deleteVBO(GLuint* vbo)
{
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	cudaGLUnregisterBufferObject(*vbo);

	*vbo = 0;
}

// ======================
//          Main         
// ======================
int main(int argc, char** argv)
{
	dataPositions = new float[NUM_BODIES * 3];
	dataVelocities = new float[NUM_BODIES * 3];
	dataMasses = new float[NUM_BODIES];
	// Data loading
	loadData("../../../data/dubinski.tab", NUM_BODIES, dataPositions, dataVelocities, dataMasses, scaleFactor);
		
	// Create app window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(720, 480);
	char windowTitle[256];
	sprintf(windowTitle, "CUDA Galaxy Simulation (%d bodies)", NUM_BODIES); 
	glutCreateWindow(windowTitle);
    
	// OpenGL setup	
	initGL();
	
	// CUDA setup
  initCUDA(NUM_BODIES);
    
	// GL callback functions
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(key);
	glutIdleFunc(idle);

	// FPS on title
	framerateTitle(windowTitle);
	
	// Start main loop
  glutMainLoop();

	deleteVBO((GLuint*) &vbo);

	if (dataPositions)
			free(dataPositions);
	if (dataVelocities)
			free(dataVelocities);
	
	if (renderer)
		delete renderer;
	
    return 0;

}