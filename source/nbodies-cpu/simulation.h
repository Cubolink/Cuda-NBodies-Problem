// =================================
// Created by Cubelink on 20-05-2023
// =================================

#ifndef NBODIES_PROBLEM_SIMULATION_H
#define NBODIES_PROBLEM_SIMULATION_H

#include <iostream>
#include <cmath>
#include "data-structs.h"
#include <GL/glew.h>

float3 bodyBodyInteraction(float3 iBody, float3 jBody, float3 ai)
{
    float3 r{};
    r.x = jBody.x - iBody.x;
    r.y = jBody.y - iBody.y;
    r.z = jBody.z - iBody.z;

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    float dist = sqrt(distSqr);
    float distCube = distSqr * dist;

    if (distCube < 1.f) return ai;

    ai.x += r.x;
    ai.y += r.y;
    ai.z += r.z;

    return ai;
}

void nBodiesKernel(int i, float3 *particlesData, int nBodies)
{
    float dt = 0.001;

    unsigned int pIdx = i;
    unsigned int vIdx = nBodies + pIdx;  // After all positions, it stores velocities

    float3 position = particlesData[pIdx];
    float3 velocity = particlesData[vIdx];

    float3 acceleration = {.0f, .0f, .0f};

    for (int j = 0; j < nBodies; j++)
    {
        acceleration = bodyBodyInteraction(position, particlesData[j], acceleration);
    }

    // Update velocity
    velocity.x += acceleration.x * dt;
    velocity.y += acceleration.y * dt;
    velocity.z += acceleration.z * dt;

    // Update position
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;
    position.z += velocity.z * dt;

    // Update particles data
    particlesData[pIdx] = position;
    particlesData[vIdx] = velocity;
}

/**
 * Updates positions and velocities of all bodies
 * @param pdata
 * @param nBodies
 */
void cpuComputeNBodies(float3 *particlesData, GLuint vbo, int nBodies)
{
    /*
    // For each body, updates its position and velocity
    for (int i = 0; i < nBodies; i++) {
        std::cout << i << "/" << nBodies << std::endl;
        nBodiesKernel(i, pdata, nBodies);
    }
     */

    // New VBO data
    auto vboData = new float[nBodies * 8];

    for (int i = 0; i < nBodies; i++)
    {
        int pIdx = i * 4;
        int vIdx = pIdx + nBodies * 4;

        vboData[pIdx] = particlesData[i].x;
        vboData[pIdx + 1] = particlesData[i].y;
        vboData[pIdx + 2] = particlesData[i].z;
        vboData[pIdx + 3] = 1.f;

        vboData[vIdx] = particlesData[nBodies + i].x;
        vboData[vIdx + 1] = particlesData[nBodies + i].y;
        vboData[vIdx + 2] = particlesData[nBodies + i].z;
        vboData[vIdx + 3] = 1.f;
    }

    // Update the VBO data
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, nBodies * 8 * sizeof(float), vboData);
}


#endif //NBODIES_PROBLEM_SIMULATION_H
