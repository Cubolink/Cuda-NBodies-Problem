/*
 * This code was modified from the NVIDIA CUDA examples
 * S. James Lee, 2008 Fall
 */


#include "particle-renderer.h"
#include <GL/glew.h>
#include <stdio.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <math.h>
#include <assert.h>

#define GL_POINT_SPRITE_ARB               0x8861
#define GL_COORD_REPLACE_ARB              0x8862
#define GL_VERTEX_PROGRAM_POINT_SIZE_NV   0x8642

// ======================
// ======== Util ========
// ======================

inline float evalHermite(float pA, float pB, float vA, float vB, float u)
{
	float u2 = (u * u), u3 = u2 * u;
	float B0 = 2 * u3 - 3 * u2 + 1;
	float B1 = -2 * u3 + 3 * u2;
	float B2 = u3 - 2 * u2 + u;
	float B3 = u3 - u;
	return(B0 * pA + B1 * pB + B2 * vA + B3 * vB);
}

unsigned char* createGaussianMap(int N)
{
	float *M = new float[2 * N * N];
	unsigned char *B = new unsigned char[4 * N * N];
	float X, Y, Y2, Dist;
	float Incr = 2.0f / N;
	int i = 0;  
	int j = 0;
	Y = -1.0f;
	for (int y = 0; y < N; y++, Y += Incr)
	{
		Y2 = Y * Y;
		X = -1.0f;
		for (int x = 0; x < N; x++, X += Incr, i += 2, j += 4)
		{
				Dist = (float) sqrtf(X * X + Y2);
				if (Dist > 1) Dist = 1;
				M[i + 1] = M[i] = evalHermite(1.0f, 0, 0, 0, Dist);
				B[j + 3] = B[j + 2] = B[j + 1] = B[j] = (unsigned char) (M[i] * 255);
		}
	}
	delete [] M;
	return(B);
}

char* textFileRead(const char* fn)
{
	FILE *fp;
	char *content = nullptr;

	int count = 0;

	if (fn != nullptr)
	{
		fp = fopen(fn, "rt");

		if (fp != nullptr)
		{
      fseek(fp, 0, SEEK_END);
      count = ftell(fp);
      rewind(fp);

			if (count > 0)
			{
				content = (char*) malloc(sizeof(char) * (count + 1));
				count = fread(content, sizeof(char), count, fp);
				content[count] = '\0';
			}
			fclose(fp);
		}
	}
	
	if (content == nullptr)
	{
		fprintf(stderr, "Error: could not load in file %s\n", fn);
		exit(1);
	}

	return content;
}           

void printShaderLog(GLuint prog)
{
	GLint infoLogLength = 0;
	GLsizei charsWritten  = 0;
	GLchar *infoLog;

	glGetShaderiv(prog, GL_INFO_LOG_LENGTH, &infoLogLength);

	if (infoLogLength > 0)
	{
		infoLog = (char *) malloc(infoLogLength);
		glGetShaderInfoLog(prog, infoLogLength, &charsWritten, infoLog);
		printf("%s\n",infoLog);
		free(infoLog);
	}
}

void printProgramLog(GLuint shad)
{
	GLint infoLogLength = 0;
	GLsizei charsWritten  = 0;
	GLchar *infoLog;

	glGetProgramiv(shad, GL_INFO_LOG_LENGTH, &infoLogLength);

	if (infoLogLength > 0)
	{
		infoLog = (char *) malloc(infoLogLength);
		glGetProgramInfoLog(shad, infoLogLength, &charsWritten, infoLog);
		printf("%s\n",infoLog);
		free(infoLog);
	}
}

// ==================================
// ======== ParticleRenderer ========
// ==================================


ParticleRenderer::ParticleRenderer(int nParticles)
: m_pos(nullptr),
  m_numParticles(nParticles),
  m_spriteSize(2.0f),
  m_program(0),
  m_texture(0),
  m_vbo(0)
{
	_createTexture(32);
}

ParticleRenderer::~ParticleRenderer()
{
	if (m_pos != nullptr)
	{
		free(m_pos);
		m_pos = nullptr;
	}
}

void ParticleRenderer::setShaders(const char* vert, const char* frag) 
{
  GLuint v,f, pro;
	char *vs, *fs;

	v = glCreateShader(GL_VERTEX_SHADER);
	f = glCreateShader(GL_FRAGMENT_SHADER);

	vs = textFileRead(vert);
	fs = textFileRead(frag);

	const char * vv = vs;
	const char * ff = fs;

	glShaderSource(v, 1, &vv, nullptr);
	glShaderSource(f, 1, &ff, nullptr);

	free(vs); free(fs);

	glCompileShader(v);
	glCompileShader(f);

	printShaderLog(v);
	printShaderLog(f);

	pro = glCreateProgram();
	glAttachShader(pro, v);
	glAttachShader(pro, f);

	glLinkProgram(pro);
	printProgramLog(pro);
	
	m_program = pro;
}

void ParticleRenderer::display()
{
	// Setup point sprites
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glPointSize(m_spriteSize);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	glEnable(GL_BLEND);
	glDepthMask(GL_FALSE);

	glUseProgram(m_program);
	glUniform1i(glGetUniformLocation(m_program, "mode"), 1);
	glUniform1i(glGetUniformLocation(m_program, "splatTexture"), 0);

	glActiveTextureARB(GL_TEXTURE0_ARB);
	glBindTexture(GL_TEXTURE_2D, m_texture);

	glutReportErrors();

	glColor3f(1, 1, 1);
	
	_drawPoints();

	glUseProgram(0);

	glutReportErrors();

	glDisable(GL_POINT_SPRITE_ARB);
	glDisable(GL_BLEND);
	glDepthMask(GL_TRUE);
}

void ParticleRenderer::_createTexture(int resolution)
{
	unsigned char* data = createGaussianMap(resolution);
	glGenTextures(1, (GLuint*) &m_texture);
	glBindTexture(GL_TEXTURE_2D, m_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, resolution, resolution, 0, 
								GL_RGBA, GL_UNSIGNED_BYTE, data);
	
	delete [] data;
}

void ParticleRenderer::_drawPoints(bool color)
{
	if (!m_vbo)
	{
		glBegin(GL_POINTS);
		{
				int k = 0;
				for (int i = 0; i < m_numParticles; ++i)
				{
						glVertex3fv(&m_pos[k]);
						k += 3;
				}
		}
		glEnd();
	}
	else
	{  
		// Render from the VBO
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		glVertexPointer(4, GL_FLOAT, 0, 0);
		glColorPointer(4, GL_FLOAT, 0, (GLvoid *) (m_numParticles * sizeof(float) * 4));

		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);

		glDrawArrays(GL_POINTS, 0, m_numParticles);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY); 
	}
}
