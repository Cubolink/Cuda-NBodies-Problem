#include <iostream>
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
#include "ParticleRenderer.h"

ParticleRenderer* renderer = 0;
int 	numBodies = 16384;

GLuint	gVBO = 0;				// 8 float (4 position, 4 color)

void createVBO(GLuint* vbo)
{
    // create buffer object
    glGenBuffers( 1, vbo);
    glBindBuffer( GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = numBodies * 8 * sizeof( float); //4
    glBufferData( GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer( GL_ARRAY_BUFFER, 0);

    //CUT_CHECK_ERROR_GL();
}

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
  createVBO((GLuint*) &gVBO);
  renderer->setVBO(gVBO, numBodies);
  renderer->setSpriteSize(0.4f);
  renderer->setShaders("../../../data/sprite.vert", "../../../data/sprite.frag");
}


void deleteVBO( GLuint* vbo)
{
    glBindBuffer( 1, *vbo);
    glDeleteBuffers( 1, vbo);

    *vbo = 0;
}


void display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  renderer->display(1);

  glutSwapBuffers();

  glutReportErrors();
}


int main(int argc, char** argv)
{
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
  // glutReshapeFunc(reshape);
  // glutMouseFunc(mouse);
  // glutMotionFunc(motion);
  // glutKeyboardFunc(key);
  // glutSpecialFunc(special);
  // glutIdleFunc(idle);

  // let's start main loop
  glutMainLoop();

	deleteVBO((GLuint*) &gVBO);

	if (renderer)
		delete renderer;

  return 0;
}