// =================================
// Created by Cubolink on 20-05-2023
// =================================

#pragma OPENCL EXTENSION cl_khr_fp64: enable
float3 bodyBodyInteraction(float3 iBody, float4 jBody, float3 ai)
{
    float3 r = {};
    r.x = jBody.x - iBody.x;
    r.y = jBody.y - iBody.y;
    r.z = jBody.z - iBody.z;

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    float dist = sqrt(distSqr);
    float distCube = distSqr * dist;

    if (distCube < 1.f) return ai;

    float s = jBody.w / distCube;

    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

__kernel void nBodiesKernelLocal1D(
    __global float4 *pvbo,
    __global float *positions,
    __global float *velocities,
    __global float *futurePositions,
    __global float *futureVelocities,
    __global float *masses,
    __local float4 *tileData,
    int nBodies) {

    float dt = 0.001f;

    // Warning: float3 are 16-bit in openCL, ie: they are as float4.
    // This may complicate things
    int i = get_global_id(0);
    int local_i = get_local_id(0);

    float3 position = vload3(i, positions);  // loads 12 bytes from positions[3 * i]
    float3 velocity = vload3(i, velocities);  // {velocities[3 * i], velocities[3 * i + 1], velocities[3 * i + 2]};
    float3 acceleration = {.0f, .0f, .0f};

    for (int tile = 0; tile * get_local_size(0) < nBodies; tile++) {
        // Copy global to local memory, only one tile in an iteration
        // Each thread copy one part of the tile
        int j = tile * get_local_size(0) + get_local_id(0);

        float3 jPosition = vload3(j, positions);
        tileData[local_i] = (float4) {jPosition.x, jPosition.y, jPosition.z, masses[j]};

        barrier(CLK_LOCAL_MEM_FENCE);

        // Each thread computes the acceleration of its body inteacting with all bodies in the tile
        for (int k = 0; k < get_local_size(0); k++)
        {
            acceleration = bodyBodyInteraction(position, tileData[k], acceleration);
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
    vstore3(position, i, futurePositions);  // stores the 12 bytes from position into positions[3*i]
    vstore3(velocity, i, futureVelocities);  // stores the 12 bytes from velocity into velocities[3*i]

    // Update VBO
    int positionIndex = i;
    int velocityIndex = positionIndex + nBodies;// get_local_size(0) * get_num_groups(0);
    pvbo[positionIndex] = (float4) {position.x, position.y, position.z, 1.f};
    pvbo[velocityIndex] = (float4) {velocity.x, velocity.y, velocity.z, 1.f};

}

__kernel void nBodiesKernelLocal2D(
    __global float4 *pvbo,
    __global float *positions,
    __global float *velocities,
    __global float *futurePositions,
    __global float *futureVelocities,
    __global float *masses,
    __local float4 *tileData,
    int nBodies) {

    float dt = 0.001f;

    // Warning: float3 are 16-bit in openCL, ie: they are as float4.
    // This may complicate things
    int x = get_global_id(0);
    int y = get_global_id(1);

    int width = get_global_size(0);
    int height = get_global_size(1);

    unsigned int i = y * width + x;
    int local_i = get_local_id(1) * get_local_size(0) + get_local_id(0);

    float3 position = vload3(i, positions);  // loads 12 bytes from positions[3 * i]
    float3 velocity = vload3(i, velocities);  // {velocities[3 * i], velocities[3 * i + 1], velocities[3 * i + 2]};
    float3 acceleration = {.0f, .0f, .0f};

    for (int tile = 0; tile * get_local_size(0) * get_local_size(1) < nBodies; tile++) {
        // Copy global to local memory, only one tile in an iteration
        // Each thread copy one part of the tile
        int j = tile * get_local_size(0) * get_local_size(1) + local_i;

        float3 jPosition = vload3(j, positions);
        tileData[local_i] = (float4) {jPosition.x, jPosition.y, jPosition.z, masses[j]};

        barrier(CLK_LOCAL_MEM_FENCE);

        // Each thread computes the acceleration of its body inteacting with all bodies in the tile
        for (int k = 0; k < get_local_size(0) * get_local_size(1); k++)
        {
            acceleration = bodyBodyInteraction(position, tileData[k], acceleration);
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
    vstore3(position, i, futurePositions);  // stores the 12 bytes from position into positions[3*i]
    vstore3(velocity, i, futureVelocities);  // stores the 12 bytes from velocity into velocities[3*i]

    // Update VBO
    int positionIndex = i;
    int velocityIndex = positionIndex + nBodies;// get_local_size(0) * get_num_groups(0);
    pvbo[positionIndex] = (float4) {position.x, position.y, position.z, 1.f};
    pvbo[velocityIndex] = (float4) {velocity.x, velocity.y, velocity.z, 1.f};

}

__kernel void nBodiesKernelGlobal1D(
        __global float4 *pvbo,
        __global float *positions,
        __global float *velocities,
        __global float *futurePositions,
        __global float *futureVelocities,
        __global float *masses,
        int nBodies)
{

    float dt = 0.001f;

    // Warning: float3 are 16-bit in openCL, ie: they are as float4.
    // This may complicate things
    int i = get_global_id(0);

    float3 position = vload3(i, positions);  // loads 12 bytes from positions[3 * i]
    float3 velocity = vload3(i, velocities);  // {velocities[3 * i], velocities[3 * i + 1], velocities[3 * i + 2]};
    float3 acceleration = {.0f, .0f, .0f};

    for (int k = 0; k < nBodies; k++)
    {
        float3 jPosition = vload3(k, positions);
        float4 jData = (float4) {jPosition.x, jPosition.y, jPosition.z, masses[k]};
        acceleration = bodyBodyInteraction(position, jData, acceleration);
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
    vstore3(position, i, futurePositions);  // stores the 12 bytes from position into positions[3*i]
    vstore3(velocity, i, futureVelocities);  // stores the 12 bytes from velocity into velocities[3*i]

    // Update VBO
    int positionIndex = i;
    int velocityIndex = positionIndex + nBodies;// get_local_size(0) * get_num_groups(0);
    pvbo[positionIndex] = (float4) {position.x, position.y, position.z, 1.f};
    pvbo[velocityIndex] = (float4) {velocity.x, velocity.y, velocity.z, 1.f};

}

__kernel void nBodiesKernelGlobal2D(
    __global float4 *pvbo,
    __global float *positions,
    __global float *velocities,
    __global float *futurePositions,
    __global float *futureVelocities,
    __global float *masses,
    int nBodies)
{

    float dt = 0.001f;

    // Warning: float3 are 16-bit in openCL, ie: they are as float4.
    // This may complicate things
    int x = get_global_id(0);
    int y = get_global_id(1);

    int width = get_global_size(0);
    int height = get_global_size(1);

    unsigned int i = y * width + x;

    float3 position = vload3(i, positions);  // loads 12 bytes from positions[3 * i]
    float3 velocity = vload3(i, velocities);  // {velocities[3 * i], velocities[3 * i + 1], velocities[3 * i + 2]};
    float3 acceleration = {.0f, .0f, .0f};

    for (int k = 0; k < nBodies; k++)
    {
        float3 jPosition = vload3(k, positions);
        float4 jData = (float4) {jPosition.x, jPosition.y, jPosition.z, masses[k]};
        acceleration = bodyBodyInteraction(position, jData, acceleration);
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
    vstore3(position, i, futurePositions);  // stores the 12 bytes from position into positions[3*i]
    vstore3(velocity, i, futureVelocities);  // stores the 12 bytes from velocity into velocities[3*i]

    // Update VBO
    int positionIndex = i;
    int velocityIndex = positionIndex + nBodies;// get_local_size(0) * get_num_groups(0);
    pvbo[positionIndex] = (float4) {position.x, position.y, position.z, 1.f};
    pvbo[velocityIndex] = (float4) {velocity.x, velocity.y, velocity.z, 1.f};

}