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
    __global float *positions,
    __global float *velocities,
    __global float *masses,
    int nBodies) {
    // Warning: float3 are 16-bit in openCL, ie: they are as float4.
    // This may complicate things
    int i = get_global_id(0);
    int local_i = get_local_id(0);

    float dt = 0.001;
    float3 position = (float3) positions[3*i];
    float3 velocity = (float3) velocities[3*i];
    float3 acceleration = {.0f, .0f, .0f};

    for (int k = 0; k < 3 * nBodies; k += 3)
    {
        acceleration = bodyBodyInteraction(position, positions[k], masses[k], acceleration);
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

}