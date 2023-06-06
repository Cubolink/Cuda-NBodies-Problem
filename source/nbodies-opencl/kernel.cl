// =================================
// Created by Cubolink on 20-05-2023
// =================================

#pragma OPENCL EXTENSION cl_khr_fp64: enable
float3 bodyBodyInteraction(float3 iBody, float3 jBody, float jMass, float3 ai)
{
    float3 r = {};
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

__kernel void nBodiesKernel(
    __global float4 *pvbo,
    __global float *positions,
    __global float *velocities,
    __global float *masses,
    __local float4 *tileData,
    int nBodies) {
    // Warning: float3 are 16-bit in openCL, ie: they are as float4.
    // This may complicate things
    int i = get_global_id(0);
    int local_i = get_local_id(0);

    float dt = 0.001;
    float3 position = {positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]};
    float3 velocity = {velocities[3 * i], velocities[3 * i + 1], velocities[3 * i + 2]};
    float3 acceleration = {.0f, .0f, .0f};

    for (int tile = 0; tile * get_local_size(0) < nBodies; tile++) {
        // Copy global to local memory, only one tile in an iteration
        // Each thread copy one part of the tile
        int j = tile * get_local_size(0) + get_local_id(0);
        tileData[local_i].x = positions[3*j];
        tileData[local_i].y = positions[3*j + 1];
        tileData[local_i].z = positions[3*j + 2];
        tileData[local_i].w = masses[j];

        barrier(CLK_LOCAL_MEM_FENCE);

        // Each thread computes the acceleration of its body inteacting with all bodies in the tile
        for (int k = 0; k < get_local_size(0); k++)
        {
            float3 kPosition = {tileData[k].x, tileData[k].y, tileData[k].z};
            acceleration = bodyBodyInteraction(position, kPosition, tileData[k].w, acceleration);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
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
    positions[3*i + 0] = position.x;
    positions[3*i + 1] = position.y;
    positions[3*i + 2] = position.z;
    velocities[3*i + 0] = velocity.x;
    velocities[3*i + 1] = velocity.y;
    velocities[3*i + 2] = velocity.z;

    // Update VBO
    int positionIndex = i;
    int velocityIndex = positionIndex + nBodies;// get_local_size(0) * get_num_groups(0);
    pvbo[positionIndex].x = position.x;
    pvbo[positionIndex].y = position.y;
    pvbo[positionIndex].z = position.z;
    pvbo[positionIndex].w = 1.f;

    pvbo[velocityIndex].x = velocity.x;
    pvbo[velocityIndex].y = velocity.y;
    pvbo[velocityIndex].z = velocity.z;
    pvbo[velocityIndex].w = 1.f;

}