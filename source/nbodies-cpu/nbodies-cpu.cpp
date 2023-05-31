#include <iostream>
#include <GL/glew.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cstdlib>
#include <cstdio>

#include "controller.h"
#include "data-loader.h"
#include "data-structs.h"
#include "framerate.h"
#include "particle-renderer.h"
#include "simulation.h"

// Number of particles to be rendered
#define NUM_BODIES 16384

// Simulation parameters
float scaleFactor = 1.5f;

// Simulation data
float3 *dataPositions = nullptr;
float3 *dataVelocities = nullptr;
float *dataMasses = nullptr;

ParticleRenderer* renderer = nullptr;
Controller* controller = new Controller(scaleFactor, 720.0f, 480.0f);

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

	// Particle renderer
	renderer = new ParticleRenderer(NUM_BODIES);
	renderer->setPos((float*) dataPositions);
	renderer->setSpriteSize(0.4f);
	renderer->setShaders("../../../data/sprite.vert", "../../../data/sprite.frag");
}

// ====================================
//          Main loop callback         
// ====================================
void display()
{
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

// Callback called when window is resized
void reshape(int w, int h)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (float) w / (float) h, 0.1, 100000.0);

	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, w, h);

	controller->setScreenSize(w, h);
}

// Called when mouse button is pressed
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
		controller->setButtonState(button + 1);
	else if (state == GLUT_UP)
		controller->setButtonState(0);

	controller->setCameraOxOy(x, y);

	glutPostRedisplay();
}

// Called when moving mouse while mouse button is pressed
void motion(int x, int y)
{
	controller->cameraMotion(x, y);
	glutPostRedisplay();
}

// Called when keyboard key is pressed
void key(unsigned char key, int x, int y)
{
	switch (key)
	{
		case '\033':
		case 'q':
			exit(0);
			break;
	}
	glutPostRedisplay();
}

// Called in order for animations to work 
void idle()
{
	glutPostRedisplay();
}

// ======================
//          Main 
// ======================
int main(int argc, char** argv)
{
	// Data loading
	dataPositions = new float3[NUM_BODIES];
	dataVelocities = new float3[NUM_BODIES];
	dataMasses = new float[NUM_BODIES];
	loadData("../../../data/dubinski.tab", NUM_BODIES, (float*) dataPositions, (float*) dataVelocities, dataMasses, scaleFactor);

	// Create OpenGL app window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(720, 480);
	char windowTitle[256];
	sprintf(windowTitle, "CUDA Galaxy Simulation (%d bodies)", NUM_BODIES); 
	glutCreateWindow(windowTitle);

	// GL setup
	initGL();

	// GL callback functions
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(key);
	glutIdleFunc(idle);

	// Start main loop
	glutMainLoop();

	if (renderer)
		delete renderer;

	return 0;
}