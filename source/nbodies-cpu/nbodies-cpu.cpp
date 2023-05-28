#include <iostream>
#include <GL/glew.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cstdlib>
#include <cstdio>

#include "simulation.h"
#include "data-loader.h"

#include "ParticleRenderer.h"
#include "framerate.h"
#include "controller.h"

ParticleRenderer* renderer = nullptr;
int 	numBodies = 16384;

// simulation parameter
float scaleFactor = 1.5f;


// simulation data
float3 *dataPositions;
float3 *dataVelocities;
float *dataMasses;

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

// particle renderer
  renderer = new ParticleRenderer(numBodies);
  renderer->setMPos((float*) dataPositions);
  renderer->setSpriteSize(0.4f);
  renderer->setShaders("../../../data/sprite.vert", "../../../data/sprite.frag");
}


void display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // view transform
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  controller->updateCameraProperties();
  float* camera_trans_lag = controller->getCameraTransLag();
  float* camera_rot_lag = controller->getCameraTransLag();

  glTranslatef(camera_trans_lag[0],
                 camera_trans_lag[1],
                 camera_trans_lag[2]);
  glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
  glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

  renderer->display(1);
  framerateUpdate();

  glutSwapBuffers();

  glutReportErrors();
}

void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 100000.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    controller->setScreenSize(w, h);
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        controller->setButtonState(button + 1);
    else if (state == GLUT_UP)
        controller->setButtonState(0);

    controller->setCameraOxOy(x, y);

    glutPostRedisplay();

}

void motion(int x, int y)
{
    controller->cameraMotion(x, y);
    glutPostRedisplay();

}

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

void special(int key, int x, int y)
{

    glutPostRedisplay();
}

void idle()
{
    glutPostRedisplay();
}

int main(int argc, char** argv)
{
    // Data loading
    dataPositions = new float3[numBodies];
    dataVelocities = new float3[numBodies];
    dataMasses = new float[numBodies];
    loadData("../../../data/dubinski.tab", numBodies,
             dataPositions, dataVelocities, dataMasses);

    // OpenGL: create app window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(720, 480);
	char wtitle[256];
	sprintf(wtitle, "CUDA Galaxy Simulation (%d bodies)", numBodies); 
	glutCreateWindow(wtitle);

    // GL setup
	initGL();

    // GL callback function
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(key);
    glutSpecialFunc(special);
    glutIdleFunc(idle);

    // let's start main loop
    glutMainLoop();

    if (renderer)
        delete renderer;

    return 0;
}