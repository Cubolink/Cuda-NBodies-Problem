// =================================
// Created by Cubelink on 20-05-2023
// =================================

#ifndef NBODIES_PROBLEM_SIMULATION_H
#define NBODIES_PROBLEM_SIMULATION_H

#include <iostream>
#include <cmath>
#include <GL/glew.h>
#include "data-structs.h"
#include "particle-timer.h"

float3 bodyBodyInteraction(float3 iBody, float3 jBody, float jMass, float3 ai)
{
    float3 r{};
    r.x = jBody.x - iBody.x;
    r.y = jBody.y - iBody.y;
    r.z = jBody.z - iBody.z;

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    float dist = sqrt(distSqr);
    float distCube = distSqr * dist;

    if (distCube < 1.f) return ai;

    float s = jMass / distCube;

    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

void nBodiesKernel(int i, float3 *positions, float3 *velocities, float *masses, float3 *futurePositions, float3 *futureVelocities, int nBodies)
{
    float dt = 0.001;

    float3 position = positions[i];
    float3 velocity = velocities[i];

    float3 acceleration = {.0f, .0f, .0f};

    for (int j = 0; j < nBodies; j++)
    {
        acceleration = bodyBodyInteraction(position, positions[j], masses[j], acceleration);
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
    futurePositions[i] = position;
    futureVelocities[i] = velocity;
}

/**
 * Updates positions and velocities of all bodies
 * @param pdata
 * @param nBodies
 */
void cpuComputeNBodies(float3 *positions, float3 *velocities, float *masses, float3 *futurePositions, float3 *futureVelocities, int nBodies, GLuint vbo, ParticleTimer* timer)
{
    // Start timer iteration
    timer->startIteration(); 

    // For each body, updates its position and velocity
    for (int i = 0; i < nBodies; i++) {
        nBodiesKernel(i, positions, velocities, masses, futurePositions, futureVelocities, nBodies);
    }

    // End timer iteration
    timer->endIteration(); 
    timer->printParticleEvaluatedPerSecond();

    // New VBO data
    auto vboData = new float[nBodies * 8];

    for (int i = 0; i < nBodies; i++)
    {
        int pIdx = i * 4;
        int vIdx = pIdx + nBodies * 4;

        vboData[pIdx] = futurePositions[i].x;
        vboData[pIdx + 1] = futurePositions[i].y;
        vboData[pIdx + 2] = futurePositions[i].z;
        vboData[pIdx + 3] = 1.f;

        vboData[vIdx] = futureVelocities[i].x;
        vboData[vIdx + 1] = futureVelocities[i].y;
        vboData[vIdx + 2] = futureVelocities[i].z;
        vboData[vIdx + 3] = 1.f;
    }

    // Update the VBO data
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, nBodies * 8 * sizeof(float), vboData);

    // Update positions and velocities data for next iteration
    memcpy(positions, futurePositions, nBodies * sizeof(float3));
    memcpy(velocities, futureVelocities, nBodies * sizeof(float3));
}


#endif //NBODIES_PROBLEM_SIMULATION_H
