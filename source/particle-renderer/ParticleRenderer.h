/*
 * This code was modified from the NVIDIA CUDA examples
 * S. James Lee, 2008 Fall
 */
 
#ifndef __PARTICLERENDERER_H__
#define __PARTICLERENDERER_H__

class ParticleRenderer
{
public:
    ParticleRenderer(int nParticles);
    ~ParticleRenderer();
    
    void setVBO(unsigned int vbo, int numParticles);
	void setShaders(const char* vert, const char* frag);

    void display();

    void setPos(float *pos) { m_pos = pos; }
    void setSpriteSize(float size) { m_spriteSize = size; }

protected: // Methods
    void _createTexture(int resolution);
    void _drawPoints(bool color = false);

protected: // Data
    float*	m_pos;
    int		m_numParticles;
    float	m_spriteSize;

    unsigned int m_program;
    unsigned int m_texture;
    unsigned int m_vbo;
};

#endif