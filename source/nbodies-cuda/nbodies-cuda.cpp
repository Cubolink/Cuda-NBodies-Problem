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
#include "framerate.h"
#include "particle-renderer.h"

// Simulation parameters
float 	scaleFactor = 1.5f;		// 10.0f, 50
float 	velFactor = 8.0f;			// 15.0f, 100
float	massFactor = 120000.0f;	// 50000000.0,
float 	gStep = 0.001f;				// 0.005f
int 	gOffset = 0;
int 	gApprx = 4;

// Simulation data storage
float* 	gPos = 0;
float* 	gVel = 0;
GLuint	vbo = 0;				// 8 float (4 position, 4 color)
float*	dParticleData = 0;		// device side particle data storage
float*	hParticleData = 0;		// host side particle data storage

// GL drawing attributes
ParticleRenderer* renderer = 0;
int 	numBodies = 16384;
float	spriteSize = scaleFactor*0.25f;

// Controller
Controller* controller = new Controller(scaleFactor, 720.0f, 480.0f);

// Cuda parameters
int threadsPerBlock = 256;

// Clamp macro
#define LIMIT(x,min,max) { if ((x)>(max)) (x)=(max); if ((x)<(min)) (x)=(min); }

// Forward declarations
void init(int bodies);
void initGL(void);
void runCuda(void);
void display(void);
void reshape(int w, int h);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void key(unsigned char key, int x, int y);
void idle(void);
void loadData(char* filename, int bodies);
void createVBO(GLuint* vbo);
void deleteVBO(GLuint* vbo);

// Initialize CUDA data
void initCUDA(int bodies)
{
	// Host particle data (position, velocity)
	hParticleData = (float *) malloc (8 * bodies * sizeof(float));
	
	// device particle data
	cudaMalloc( (void**) &dParticleData, 8 * bodies * sizeof(float));
	
	// load inital data set
	int pCounter;
	int idx = 0;
	int vidx = 0;
	int offset = 0;
	for (int i = 0; i < bodies; i++)
	{
		// float array index
		idx = i * 4;
		vidx = bodies*4 + idx;
		
		if ((i % 2) == 0)
			offset = idx;
		else
			offset = (idx + (bodies / 2)) % (bodies * 4);
		// set value from global data storage
		hParticleData[idx + 0]		= gPos[offset + 0];	// x
		hParticleData[idx + 1] 	= gPos[offset + 1];	// y
		hParticleData[idx + 2] 	= gPos[offset + 2];	// z
		hParticleData[idx + 3] 	= gPos[offset + 3];	// mass
		hParticleData[vidx + 0] 	= gVel[offset + 0];	// vx
		hParticleData[vidx + 1]	= gVel[offset + 1];	// vy
		hParticleData[vidx + 2] 	= gVel[offset + 2];	// vz
		hParticleData[vidx + 3] 	= gVel[offset + 3];	// padding
		
	}
	
	// copy initial value to GPU memory
	cudaMemcpy(dParticleData, hParticleData, 8 * bodies * sizeof(float), cudaMemcpyHostToDevice);
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
	renderer = new ParticleRenderer(numBodies);
	renderer->setVBO(vbo);
	renderer->setSpriteSize(0.4f);
	renderer->setShaders("../../../data/sprite.vert", "../../../data/sprite.frag");
}

////////////////////////////////////////////////////////////////////////////////
// CUDA kernel interface
extern "C" 
void cudaComputeGalaxy(float4* pos, float4 * pdata, int width, int height, 
					   float step, int apprx, int offset);

////////////////////////////////////////////////////////////////////////////////
void runCuda(void)
{
	// Map OpenGL vertex buffer object for writing from CUDA
	float4 *dptr;
	cudaGLMapBufferObject((void**) &dptr, vbo);

	// only compute 1/16 at one time
	gOffset = (gOffset+1) % (gApprx);

	// execute the kernel
	// each block has 16x16 threads, grid 16xX: X will be decided by the # of bodies
	cudaComputeGalaxy(dptr, (float4*) dParticleData, 256, numBodies / 256, 
							gStep, gApprx, gOffset);

	// Unmap vertex buffer object
	cudaGLUnmapBufferObject(vbo);
}

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

////////////////////////////////////////////////////////////////////////////////
#define MAXSTR 256
float mx,my,mz,Mx,My,Mz;
void loadData(char* filename, int bodies)
{
    int skip = 49152 / bodies;
    //int skip = 81920 / bodies;
    
    FILE *fin;
    
    if ((fin = fopen(filename, "r")))
    {
    
    	char buf[MAXSTR];
    	float v[7];
    	int idx = 0;
    	
    	// allocate memory
    	gPos	= (float*)malloc(sizeof(float)*bodies*4);
    	gVel	= (float*)malloc(sizeof(float)*bodies*4);
    	 
    	// total 81920 particles
    	// 16384 Gal. Disk
    	// 16384 And. Disk
    	// 8192  Gal. bulge
    	// 8192  And. bulge
    	// 16384 Gal. halo
    	// 16384 And. halo
    	int k=0;
    	for (int i=0; i< bodies; i++,k++)
    	{
    		// depend on input size...
    		for (int j=0; j < skip; j++,k++)
    			fgets (buf, MAXSTR, fin);	// lead line
    		
    		sscanf(buf, "%f %f %f %f %f %f %f", v+0, v+1, v+2, v+3, v+4, v+5, v+6);
    		
    		// update index
    		idx = i * 4;
    		
    		// position
    		gPos[idx+0] = v[1]*scaleFactor;
    		gPos[idx+1] = v[2]*scaleFactor;
    		gPos[idx+2] = v[3]*scaleFactor;
    		
    		// mass
    		gPos[idx+3] = v[0]*massFactor;
    		//gPos[idx+3] = 1.0f;
    		//printf("mass : %f\n", gPos[idx+3]);
    		
    		// velocity
    		gVel[idx+0] = v[4]*velFactor;
    		gVel[idx+1] = v[5]*velFactor;
    		gVel[idx+2] = v[6]*velFactor;
    		gVel[idx+3] = 1.0f;
    		
    	}   
    }
    else
    {
    	printf("cannot find file...: %s\n", filename);
    	exit(0);
    }
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

// ===========================
// ===========================
int main(int argc, char** argv)
{
	// Data loading
	loadData("../../../data/dubinski.tab", numBodies);
		
	// Create app window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(720, 480);
	char windowTitle[256];
	sprintf(windowTitle, "CUDA Galaxy Simulation (%d bodies)", numBodies); 
	glutCreateWindow(windowTitle);
    
	// OpenGL setup	
	initGL();
	
	// CUDA setup
  initCUDA(numBodies);
    
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

	if (gPos)
			free(gPos);
	if (gVel)
			free(gVel);
	
	if (renderer)
		delete renderer;
	
    return 0;

}
