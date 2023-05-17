/*
 * This code was modified from the NVIDIA CUDA examples
 *     specifically the nbody and ...
 *
 * S. James Lee, 2008 Fall
 */
 
#ifndef __PARTICLERENDERER_H__
#define __PARTICLERENDERER_H__

#define PARTICLE_POINTS 0
#define PARTICLE_SPRITES 1
#define PARTICLE_SPRITES_COLOR 2

class ParticleRenderer
{
public:
    ParticleRenderer(int nParticles);
    ~ParticleRenderer();
    
    void setVBO(unsigned int vbo, int numParticles);
	void setShaders(char * vert, char * frag);

    void display(int mode = PARTICLE_POINTS);

    void setPointSize(float size)  { m_pointSize = size; }
    void setSpriteSize(float size) { m_spriteSize = size; }

protected: // methods
    void _createTexture(int resolution);
    void _drawPoints(bool color = false);

protected: // data
	int 	m_mode;
    float*	m_pos;
    int		m_numParticles;

    float	m_pointSize;
    float	m_spriteSize;

    unsigned int m_program;
    unsigned int m_texture;
    unsigned int m_vbo;
};



#endif