// ==================================
// Modified from NVIDIA CUDA examples
// ==================================

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
#include <math.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "controller.h"
#include "data-loader.h"
#include "framerate.h"
#include "particle-renderer.h"
#include "particle-timer.h"

// Number of particles to be loaded from file
#define NUM_BODIES 16384

// Block size
#define BLOCK_SIZE 256

// Block dimension
// #define GLOBAL_MEMORY
#define LOCAL_MEMORY

// Memory configuration
#define ONE_DIM_BLOCK
// #define TWO_DIM_BLOCK

// Number of CUDA blocks after padding
int numBlocks;

// Number of particles after padding, if there exists padding
int numBodies;

// Simulation parameters
float scaleFactor = 1.5f;

// Simulation data storage loaded from file
float* dataPositions = nullptr;
float* dataVelocities = nullptr;
float* dataMasses = nullptr;

GLuint VBO = 0; // OpenGL VBO
struct cudaGraphicsResource *cudaVBOResource; // CUDA Graphics Resource pointer

float* hPositions = nullptr;
float* hVelocities = nullptr;
float* hMasses = nullptr;

float* dPositions = nullptr; // Device side particles positions
float* dVelocities = nullptr; // Device side particles velocities
float* dFuturePositions = nullptr; // Device side particles future positions
float* dFutureVelocities = nullptr; // Device side particles future velocities
float* dMasses = nullptr; // Device side particles masses

// GL drawing attributes
float	spriteSize = scaleFactor * 0.25f;
ParticleRenderer* renderer = nullptr;

// Timer
ParticleTimer* particleTimer;

// Controller
Controller* controller = new Controller(scaleFactor, 720.0f, 480.0f);

// Clamp macro
#define LIMIT(x,min,max) { if ((x)>(max)) (x)=(max); if ((x)<(min)) (x)=(min); }

// Forward declarations
void initCUDA();
void initGL();
void runCuda();
void display();
void reshape(int w, int h);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void key(unsigned char key, int x, int y);
void idle();
void createVBO(GLuint* vbo);
void deleteVBO(GLuint* vbo);

// Initialize CUDA data
void initCUDA()
{	
	// Round up in case NUM_BODIES is not a multiple of BLOCK_SIZE
	numBlocks = (NUM_BODIES + BLOCK_SIZE - 1) / BLOCK_SIZE;

	// Number of particles after padding, if there exists padding
	numBodies = numBlocks * BLOCK_SIZE;

	particleTimer = new ParticleTimer(numBodies);

	hPositions = new float[numBodies * 3];
	hVelocities = new float[numBodies * 3];
	hMasses = new float[numBodies];

	// Apply padding in case of round up
	std::fill_n(hPositions, 3 * numBodies, 0.0f);
	memcpy(hPositions, dataPositions, 3 * NUM_BODIES * sizeof(float));
	std::fill_n(hVelocities, 3 * numBodies, 0.0f);
	memcpy(hVelocities, dataVelocities, 3 * NUM_BODIES * sizeof(float));
	std::fill_n(hMasses, numBodies, 0.0f);
	memcpy(hMasses, dataMasses, NUM_BODIES * sizeof(float));

	// Device particles data
	cudaMalloc((void**) &dPositions, 3 * numBodies * sizeof(float));
	cudaMalloc((void**) &dVelocities, 3 * numBodies * sizeof(float));
	cudaMalloc((void**) &dFuturePositions, 3 * numBodies * sizeof(float));
	cudaMalloc((void**) &dFutureVelocities, 3 * numBodies * sizeof(float));
	cudaMalloc((void**) &dMasses, numBodies * sizeof(float));

	// Copy initial values to GPU memory
	cudaMemcpy(dPositions, hPositions, 3 * numBodies * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dVelocities, hVelocities, 3 * numBodies * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dMasses, hMasses, numBodies * sizeof(float), cudaMemcpyHostToDevice);

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
	createVBO((GLuint*) &VBO);
	renderer = new ParticleRenderer(numBodies);
	renderer->setVBO(VBO);
	renderer->setSpriteSize(0.4f);
	renderer->setShaders("../../../data/sprite.vert", "../../../data/sprite.frag");
}

// ========================================================================
// Start CUDA code
// ========================================================================

__device__
float3 bodyBodyInteraction(float3 iBody, float4 jData, float3 ai)
{
    float3 r{};
    r.x = jData.x - iBody.x;
    r.y = jData.y - iBody.y;
    r.z = jData.z - iBody.z;

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    float dist = sqrt(distSqr);
    float distCube = distSqr * dist;

    if (distCube < 1.f) return ai;

    float s = jData.w / distCube;

    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

__global__
void nBodiesKernelGlobal1D(float4* pvbo, float3* positions, float3* velocities, float3* futurePositions, float3* futureVelocities, float* masses, int nBodies)
{
	float dt = 0.001f;

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	float3 position = positions[x];
	float3 velocity = velocities[x];
	float3 acceleration = {.0f, .0f, .0f};

  int j;
  for (j = 0; j < nBodies; j++) {
    float3 jPosition = positions[j];
    float4 jData = make_float4(jPosition.x, jPosition.y, jPosition.z, masses[j]);
    acceleration = bodyBodyInteraction(position, jData, acceleration);
  }

	velocity.x += acceleration.x * dt;
	velocity.y += acceleration.y * dt;
	velocity.z += acceleration.z * dt;

	position.x += velocity.x * dt;
	position.y += velocity.y * dt;
	position.z += velocity.z * dt;

	futurePositions[x] = position;
	futureVelocities[x] = velocity;

	int positionIndex = x;
	int velocityIndex = gridDim.x * blockDim.x + positionIndex;

	pvbo[positionIndex] = make_float4(position.x, position.y, position.z, 1.0f);
	pvbo[velocityIndex] = make_float4(velocity.x, velocity.y, velocity.z, 1.0f);
}

__global__
void nBodiesKernelLocal1D(float4* pvbo, float3* positions, float3* velocities, float3* futurePositions, float3* futureVelocities, float* masses, int bodies)
{
	extern __shared__ float4 tileData[];

	float dt = 0.001f;

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	float3 position = positions[x];
	float3 velocity = velocities[x];
	float3 acceleration = {.0f, .0f, .0f};

	int k, tile;
	for (tile = 0; tile * blockDim.x < bodies; tile++) {    
		int idx = tile * blockDim.x + threadIdx.x;     

		float3 jPosition = positions[idx];
		tileData[threadIdx.x] = make_float4(jPosition.x, jPosition.y, jPosition.z, masses[idx]);     

		__syncthreads();     
		
		for (k = 0; k < blockDim.x; k++) {     
			acceleration = bodyBodyInteraction(position, tileData[k], acceleration);   
		}

		__syncthreads();   
	}

	velocity.x += acceleration.x * dt;
	velocity.y += acceleration.y * dt;
	velocity.z += acceleration.z * dt;

	position.x += velocity.x * dt;
	position.y += velocity.y * dt;
	position.z += velocity.z * dt;

	futurePositions[x] = position;
	futureVelocities[x] = velocity;

	int positionIndex = x;
	int velocityIndex = gridDim.x * blockDim.x + positionIndex;

	pvbo[positionIndex] = make_float4(position.x, position.y, position.z, 1.0f);
	pvbo[velocityIndex] = make_float4(velocity.x, velocity.y, velocity.z, 1.0f);
}

void runCuda(void)
{
	// Map OpenGL vertex buffer object for writing from CUDA
	float4 *dptr;
	cudaGraphicsMapResources(1, &cudaVBOResource, 0);
	size_t numBytes;
  cudaGraphicsResourceGetMappedPointer((void**) &dptr, &numBytes, cudaVBOResource);

	// Start timer iteration
	particleTimer->startIteration();

	// Run the kernel
	#ifdef GLOBAL_MEMORY
			#ifdef ONE_DIM_BLOCK
					nBodiesKernelGlobal1D<<<numBlocks, BLOCK_SIZE>>>(dptr, (float3*) dPositions, (float3*) dVelocities, (float3*) dFuturePositions, (float3*) dFutureVelocities, dMasses, numBodies);
			#endif
	#elif defined LOCAL_MEMORY
			#ifdef ONE_DIM_BLOCK
					nBodiesKernelLocal1D<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float4)>>>(dptr, (float3*) dPositions, (float3*) dVelocities, (float3*) dFuturePositions, (float3*) dFutureVelocities, dMasses, numBodies);
			#endif
	#endif

	// Synchronize the device with host
	cudaDeviceSynchronize();

	// End timer iteration
	particleTimer->endIteration();

	// Update positions and velocities for next iteration
	cudaMemcpy(dPositions, dFuturePositions, 3 * numBodies * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dVelocities, dFutureVelocities, 3 * numBodies * sizeof(float), cudaMemcpyDeviceToDevice);

	// Unmap vertex buffer object
	cudaGraphicsUnmapResources(1, &cudaVBOResource, 0);
}

// ========================================================================
// End CUDA code
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
		case 'c':
			particleTimer->exportData("data/");
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
	glBufferData(GL_ARRAY_BUFFER, numBodies * 8 * sizeof(float), 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Register this buffer object with CUDA
  cudaGraphicsGLRegisterBuffer(&cudaVBOResource, *vbo, cudaGraphicsMapFlagsWriteDiscard);
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
    
	// CUDA setup
  initCUDA();

	// OpenGL setup	
	initGL();
    
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

	deleteVBO((GLuint*) &VBO);

	// Free heap memory
	free(dataPositions);
	free(dataVelocities);
	free(dataMasses);
	free(hPositions);
	free(hVelocities);
	free(hMasses);
	delete renderer;
	
  return 0;
}