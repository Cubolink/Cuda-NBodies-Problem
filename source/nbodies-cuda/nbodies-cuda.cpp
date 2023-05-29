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
//#include <cutil.h>
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#include <cuda_runtime_api.h>
//#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>

#include "ParticleRenderer.h"
#include "framerate.h"
#include "controller.h"


int cutGetCmdLineArgumenti(int argc, const char** argv, const char* argName, int* value) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], argName) == 0) {
            if (i + 1 < argc) {
                *value = atoi(argv[i + 1]);
                return 1;
            } else {
                // Argument found, missing value
                return 0;
            }
        }
    }
    // Argument is not found
    return 0;
}


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
////////////////////////////////////////////////////////////////////////////////
// simulation parameter
float 	scaleFactor = 1.5f;		// 10.0f, 50
float 	velFactor = 8.0f;			// 15.0f, 100
float	massFactor = 120000.0f;	// 50000000.0,
float 	gStep = 0.001f;				// 0.005f
int 	gOffset = 0;
int 	gApprx = 4;

// GL drawing object
ParticleRenderer* renderer = 0;
int 	numBodies = 16384;
float	gSpriteSize = scaleFactor*0.25f;

// simulation data storage
float* 	gPos = 0;
float* 	gVel = 0;
GLuint	gVBO = 0;				// 8 float (4 position, 4 color)
float*	d_particleData = 0;		// device side particle data storage
float*	h_particleData = 0;		// host side particle data storage

Controller* controller = new Controller(scaleFactor, 720.0f, 480.0f);

// cuda related...
int 	numBlocks = 1;
int 	numThreadsPerBlock = 256;

// useful clamp macro
#define LIMIT(x,min,max) { if ((x)>(max)) (x)=(max); if ((x)<(min)) (x)=(min);}

////////////////////////////////////////////////////////////////////////////////
// forward declaration
void init(int bodies);
void reset(void);
void initGL(void);
void runCuda(void);
void display(void);
void reshape(int w, int h);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void key(unsigned char key, int x, int y);
void special(int key, int x, int y);
void idle(void);
void loadData(char* filename, int bodies);
void createVBO( GLuint* vbo);
void deleteVBO( GLuint* vbo);

////////////////////////////////////////////////////////////////////////////////
void init(int bodies)
{
	
	// blocks per grid
	numBlocks = bodies / numThreadsPerBlock;
	
	// host particle data (position, velocity
	h_particleData = (float *) malloc (8 * bodies * sizeof(float));
	
	// device particle data
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_particleData, 
							   8 * bodies * sizeof(float)));
	
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
		h_particleData[idx + 0]		= gPos[offset + 0];	// x
		h_particleData[idx + 1] 	= gPos[offset + 1];	// y
		h_particleData[idx + 2] 	= gPos[offset + 2];	// z
		h_particleData[idx + 3] 	= gPos[offset + 3];	// mass
		h_particleData[vidx + 0] 	= gVel[offset + 0];	// vx
		h_particleData[vidx + 1]	= gVel[offset + 1];	// vy
		h_particleData[vidx + 2] 	= gVel[offset + 2];	// vz
		h_particleData[vidx + 3] 	= gVel[offset + 3];	// padding
		
	}
	
	// copy initial value to GPU memory
	CUDA_SAFE_CALL( cudaMemcpy(d_particleData, h_particleData,
							   8 * bodies * sizeof(float), 
							   cudaMemcpyHostToDevice) );
}

////////////////////////////////////////////////////////////////////////////////
void reset(void)
{
   	// reset camera
   	controller->cameraReset(scaleFactor);
	
	// reset dataset
	CUDA_SAFE_CALL( cudaMemcpy(d_particleData, h_particleData,
							   8 * numBodies * sizeof(float), 
							   cudaMemcpyHostToDevice) );
}

////////////////////////////////////////////////////////////////////////////////
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
	
	// particle renderer
    renderer = new ParticleRenderer(numBodies);
    createVBO((GLuint*)&gVBO);
    renderer->setVBO(gVBO, numBodies);
    renderer->setSpriteSize(0.4f);
    renderer->setShaders("../../../data/sprite.vert",
    					 "../../../data/sprite.frag");
}

////////////////////////////////////////////////////////////////////////////////
// CUDA kernel interface
extern "C" 
void cudaComputeGalaxy(float4* pos, float4 * pdata, int width, int height, 
					   float step, int apprx, int offset);

////////////////////////////////////////////////////////////////////////////////
void runCuda(void)
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    CUDA_SAFE_CALL( cudaGLMapBufferObject( (void**)&dptr, gVBO) );

	// only compute 1/16 at one time
	gOffset = (gOffset+1) % (gApprx);
	
    // execute the kernel
    // each block has 16x16 threads, grid 16xX: X will be decided by the # of bodies
    cudaComputeGalaxy( dptr, (float4*)d_particleData, 256, numBodies / 256, 
    				   gStep, gApprx, gOffset);

    // unmap buffer object
    CUDA_SAFE_CALL( cudaGLUnmapBufferObject( gVBO) );
    
    
}

////////////////////////////////////////////////////////////////////////////////
void display(void)
{
    // update simulation
    runCuda();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  

    // view transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    controller->updateCameraProperties();
    float* camera_trans_lag = controller->getCameraTransLag();
    float* camera_rot_lag = controller->getCameraRotLag();

    glTranslatef(camera_trans_lag[0], 
		     camera_trans_lag[1], 
		     camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);
	
	// render bodies
    renderer->setSpriteSize(gSpriteSize);
    renderer->display();
	
    // update frame rate
	framerateUpdate();
	
    glutSwapBuffers();

    glutReportErrors();
}

////////////////////////////////////////////////////////////////////////////////
void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 100000.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
    
    //
    gSpriteSize *= (float)w / controller->getScreenWidth();
    
    //
    controller->setScreenSize(w, h);
    
    
}

////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    
    int mods;

    if (state == GLUT_DOWN)
        controller->setButtonState(button + 1);
    else if (state == GLUT_UP)
        controller->setButtonState(0);

    mods = glutGetModifiers();
    if (mods & GLUT_ACTIVE_SHIFT) 
    {
        //buttonState = 2;
    } 
    else if (mods & GLUT_ACTIVE_CTRL) 
    {
        //buttonState = 3;
    }

    controller->setCameraOxOy(x, y);

    glutPostRedisplay();

}

////////////////////////////////////////////////////////////////////////////////
void motion(int x, int y)
{
    controller->cameraMotion(x, y);
	glutPostRedisplay();
	
}

////////////////////////////////////////////////////////////////////////////////
void key(unsigned char key, int x, int y)
{

	switch (key)
	{
	case '\033':
    case 'q':
        exit(0);
        break;
    case 'r':
    	// reset configuration
		reset();
    	break;
    case '=':
    	// increase point size
    	gSpriteSize += scaleFactor*0.02f;
    	LIMIT(gSpriteSize, 0.1f, scaleFactor*2.0f);
    	break;
    case '-':
    	// decrese point size
    	gSpriteSize -= scaleFactor*0.02f;
    	LIMIT(gSpriteSize, 0.1f, scaleFactor*2.0f);
    	break;
    }
    
	glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
void special(int key, int x, int y)
{
	
	glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
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
	
	//printf("bulge min,max: %f %f %f %f %f %f\n", mx, my, mz, Mx, My, Mz);
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint* vbo)
{
    // create buffer object
    glGenBuffers( 1, vbo);
    glBindBuffer( GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = numBodies * 8 * sizeof( float); //4
    glBufferData( GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer( GL_ARRAY_BUFFER, 0);

    // register buffer object with CUDA
    CUDA_SAFE_CALL( cudaGLRegisterBufferObject(*vbo) );

    //CUT_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO( GLuint* vbo)
{
    glBindBuffer( 1, *vbo);
    glDeleteBuffers( 1, vbo);

    CUDA_SAFE_CALL( cudaGLUnregisterBufferObject(*vbo) );

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{

    //CUT_DEVICE_INIT(argc, argv);

    // get number of SMs on this GPU
    int devID;
    cudaDeviceProp props;
    CUDA_SAFE_CALL( cudaGetDevice(&devID) );
    CUDA_SAFE_CALL( cudaGetDeviceProperties(&props, devID) );
    
    // Geforce 8600 (Macbook pro) profile
	//Device 0: "GeForce 8600M GT"
	//Major revision number:                         1
	//Minor revision number:                         1
	//Total amount of global memory:                 512 MB
	//Number of multiprocessors:                     4
	//Number of cores:                               32
	//Total amount of constant memory:               64 KB
	//Total amount of shared memory per block:       16 KB
	//Total number of registers available per block: 8192
	//Warp size:                                     32
	//Maximum number of threads per block:           512
	//Maximum sizes of each dimension of a block:    512 x 512 x 64
	//Maximum sizes of each dimension of a grid:     65535 x 65535 x 1
	//Maximum memory pitch:                          262144 bytes
	//Texture alignment:                             256 bytes
	//Clock rate:                                    0.75 GHz
	//Concurrent copy and execution:                 Yes

	// thread block size
    int p = 256;	// width  (number of threads in col within block)
    int q = 1;		// height (number of threads in row within block)
	
	// get total number of bodies
    if (!cutGetCmdLineArgumenti(argc, (const char**) argv, "n", &numBodies))
    	// default number of bodies is #SMs * 4 * CTA size
    	//numBodies = p * q * 4 * props.multiProcessorCount;
    	numBodies = 8192;

	if (numBodies > 49152)
	{
		numBodies = 49152;
		printf("maximun number of bodies is 49152.\n");
	}
		
	// keep num of threads per block to 256
    if (q * p > 256)
    {
        p = 256 / q;
        printf("Setting p=%d, q=%d to maintain %d threads per block\n", p, q, 256);
    }

    if (q == 1 && numBodies < p)
    {
        p = numBodies;
    }

	// Data loading
	if (numBodies % 4096 != 0)
	{
		printf("number of body must be mulples of 4096\n");
		exit(0);
	}
	loadData("../../../data/dubinski.tab", numBodies);
		
	// OpenGL: create app window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(720, 480);
	char wtitle[256];
	sprintf(wtitle, "CUDA Galaxy Simulation (%d bodies)", numBodies); 
	glutCreateWindow(wtitle);
    
    // GL setup	
	initGL();
	
	// Initialize nbody system...	
    init(numBodies);
    
    // GL callback function
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(key);
    glutSpecialFunc(special);
    glutIdleFunc(idle);

	//
	framerateTitle(wtitle);
	
	// let's start main loop
    glutMainLoop();

	deleteVBO((GLuint*)&gVBO);

	// clean up memory stuff
    if (gPos)
        free(gPos);
    if (gVel)
        free(gVel);
	
	if (renderer)
		delete renderer;
	
    return 0;

}




