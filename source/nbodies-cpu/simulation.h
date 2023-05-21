//
// Created by major on 20-05-2023.
//

#ifndef NBODIES_PROBLEM_SIMULATION_H
#define NBODIES_PROBLEM_SIMULATION_H

#include <cmath>
#include "data-structs.h"

float3 bodyBodyInteraction(float3 i_body, float3 j_body, float3 ai)
{
    float3 r;
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

float3 tile_calculation(float3 myPosition, float3 *pdata, float3 acc)
{
    for (unsigned int i = 0; i < 1; i++)
    {
        acc = bodyBodyInteraction(myPosition, pdata[i], acc);
    }

    return acc;
}

void galaxyKernel(int x, int y, float3 *pos, float3 *pdata, unsigned int width,
                  unsigned int height, float step, int apprx, int offset)
{
    // index of my body
    unsigned int pLoc = y * width + x;
    unsigned int vLoc = width * height + pLoc;  // until width * height stores position, then stores velocities

    // starting index of the position array
    unsigned  int start = ((width * height) / apprx) * offset;

    float3 myPosition = pdata[pLoc];
    float3 myVelocity = pdata[vLoc];

    float3 acc = {.0f, .0f, .0f};

    unsigned int idx = 0;
    unsigned int loop = ((width * height) / apprx);
    for (int i = 0; i < loop; i++)
    {
        acc = tile_calculation(myPosition, pdata, acc);
    }

    // update velocity with above acc
    myVelocity.x += acc.x * step;
    myVelocity.y += acc.y * step;
    myVelocity.z += acc.z * step;

    // update position
    myPosition.x += myVelocity.x * step;
    myPosition.y += myVelocity.y * step;
    myPosition.z += myVelocity.z * step;

    // update pdata
    pdata[pLoc] = myPosition;
    pdata[vLoc] = myVelocity;
}

void cpuComputeGalaxy(float3 *pos, float3 *pdata, int width, int height,
                      float step, int apprx, int offset)
{
    // Here we could split in threads, each one running galaxyKernel
    // But we want CPU to use one thread, so:
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            // loc = i * width + j;
            // (i x j) matrix means i are rows, j are columns => i is vertical , j is horizontal
            // therefore y = i, x = j
            galaxyKernel(j, i, pos, pdata, width, height, step, apprx, offset);
}


#endif //NBODIES_PROBLEM_SIMULATION_H
