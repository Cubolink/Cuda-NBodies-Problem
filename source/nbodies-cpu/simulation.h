//
// Created by major on 20-05-2023.
//

#ifndef NBODIES_PROBLEM_SIMULATION_H
#define NBODIES_PROBLEM_SIMULATION_H

#include <iostream>
#include <cmath>
#include "data-structs.h"
#include <GL/glew.h>

float3 bodyBodyInteraction(float3 i_body, float3 j_body, float3 ai)
{
    float3 r{};
    r.x = j_body.x - i_body.x;
    r.y = j_body.y - i_body.y;
    r.z = j_body.z - i_body.z;

    float distSqr = r.x*r.x + r.y*r.y + r.z*r.z;
    // soft the distSqr here?

    float dist = sqrt(distSqr);
    float distCube = distSqr * dist;

    if (distCube < 1.f) return ai;

    ai.x += r.x;
    ai.y += r.y;
    ai.z += r.z;

    return ai;
}

void galaxyKernel(int i, float3 *pdata, int nBodies)
{
    float dt = 0.001;
    // index of my body
    unsigned int pLoc = i;
    unsigned int vLoc = nBodies + pLoc;  // after all positions, it stores velocities

    float3 myPosition = pdata[pLoc];
    float3 myVelocity = pdata[vLoc];

    float3 acc = {.0f, .0f, .0f};

    unsigned int idx = 0;
    for (int j = 0; j < nBodies; j++)
    {
        acc = bodyBodyInteraction(myPosition, pdata[j], acc);
    }

    // update velocity with above acc
    myVelocity.x += acc.x * dt;
    myVelocity.y += acc.y * dt;
    myVelocity.z += acc.z * dt;

    // update position
    myPosition.x += myVelocity.x * dt;
    myPosition.y += myVelocity.y * dt;
    myPosition.z += myVelocity.z * dt;

    // update pdata
    pdata[pLoc] = myPosition;
    pdata[vLoc] = myVelocity;
}

/**
 * Updates positions and velocities of all bodies
 * @param pdata
 * @param nBodies
 */
void cpuComputeGalaxy(float3 *pdata, int nBodies, GLuint vbo)
{
    /*
    // For each body, updates its position and velocity
    for (int i = 0; i < nBodies; i++) {
        std::cout << i << "/" << nBodies << std::endl;
        galaxyKernel(i, pdata, nBodies);
    }
     */

    auto aux = new float[2 * nBodies * 4];
    for (int i = 0; i < nBodies; i++) {
        int pIdx = 4 * i;
        int vIdx = pIdx + nBodies;
        int offset;

        if (i % 2)
            offset = (pIdx + (nBodies / 2)) % (nBodies * 4);
        else
            offset = pIdx;
        offset /= 4;

        aux[pIdx] = pdata[offset].x;
        aux[pIdx + 1] = pdata[offset].y;
        aux[pIdx + 2] = pdata[offset].z;
        aux[pIdx + 3] = 1.f;

        aux[vIdx] = pdata[nBodies + offset].x;
        aux[vIdx + 1] = pdata[nBodies + offset].y;
        aux[vIdx + 2] = pdata[nBodies + offset].z;
        aux[vIdx + 3] = 1.f;
    }
    std::cout << "updated aux" << std::endl;

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, (long long) (nBodies * 8 * sizeof(float)), aux);
}


#endif //NBODIES_PROBLEM_SIMULATION_H
