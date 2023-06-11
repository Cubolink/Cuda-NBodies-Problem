//
// Created by Cubolink on 31-05-2023.
//

#include <iostream>
#include <fstream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "data-loader.h"
#include "particle-timer.h"

struct float3
{
    float x;
    float y;
    float z;
};

// Simulation parameters
float scaleFactor = 1.5f;

// Simulation data storage loaded from file
float3* dataPositions = nullptr;
float3* dataVelocities = nullptr;
float* dataMasses = nullptr;

// Device side data buffers
cl::Buffer dPositions;
cl::Buffer dVelocities;
cl::Buffer dFuturePositions;
cl::Buffer dFutureVelocities;
cl::Buffer dMasses;

// OpenCL stuff
cl::CommandQueue queue;
cl::Program program;
cl::Kernel nBodiesKernelLocal1D;
cl::Kernel nBodiesKernelLocal2D;
cl::Kernel nBodiesKernelGlobal1D;
cl::Kernel nBodiesKernelGlobal2D;
cl::Kernel *nBodiesKernel;  // pointer to the used kernel
cl::Context context;

ParticleTimer* particleTimer;

// Number of particles to be loaded from file
#define NUM_BODIES 16384

// OpenCL work-group size
#define GROUP_SIZE 256

// Memory configuration
//#define GLOBAL_MEMORY
#define LOCAL_MEMORY

// Block configuration
#define ONE_DIM_BLOCK
// #define TWO_DIM_BLOCK

// OpenCL number of groups
int clNumGroups;
// Number of particles our kernel will work after padding, if any
int clNumBodies;
// variables when 2D memory
int groupSideSize;
int clNumBodiesX;
int clNumBodiesY;


// Clamp macro
#define LIMIT(x,min,max) { if ((x)>(max)) (x)=(max); if ((x)<(min)) (x)=(min); }

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_GPU
#endif

std::string load_program(const std::string& input) {
    std::ifstream stream(input.c_str());
    if(!stream.is_open()) {
        std::cout << "Cannot open the file: " << input << std::endl;
        exit(1);
    }
    return {std::istreambuf_iterator<char>(stream),
            (std::istreambuf_iterator<char>())};

}

// Utility function for grid dimensions calculation
int closestDivisorToSquareRoot(int n) {
    int sqrtN = std::sqrt(n);
    int closestDivisor = 1;

    for (int i = sqrtN; i >= 1; --i) {
        if (n % i == 0) {
            closestDivisor = i;
            break;
        }
    }

    return closestDivisor;
}

void initOpenCL() {
    // Round up in case NUM_BODIES is not a multiple of GROUP_SIZE
    clNumGroups = (NUM_BODIES + GROUP_SIZE - 1) / GROUP_SIZE;

    // Number of particles. If clNumGroups was rounded, then there's padding and clNumBodies > NUM_BODIES
    clNumBodies = clNumGroups * GROUP_SIZE;
    int paddedBodies = (clNumBodies - NUM_BODIES);

    // 2D block calculations
    #ifdef TWO_DIM_BLOCK
    groupSideSize = (int) sqrt(GROUP_SIZE);
		if (groupSideSize * groupSideSize != GROUP_SIZE) {
			std::cout << "BLOCK_SIZE has to be a perfect square" << std::endl;
			exit(1);
		}

		int gridSide = closestDivisorToSquareRoot(clNumGroups);

    clNumBodiesX = gridSide * groupSideSize; clNumBodiesY = clNumBodies / clNumBodiesX;
    #endif

    // Define the timer and select the kernel
    #ifdef GLOBAL_MEMORY
        #ifdef ONE_DIM_BLOCK
            particleTimer = new ParticleTimer(clNumBodies, "global-1d", GROUP_SIZE, NUM_BODIES);
            nBodiesKernel = &nBodiesKernelGlobal1D;
        #else
            particleTimer = new ParticleTimer(clNumBodies, "global-2d", GROUP_SIZE, NUM_BODIES);
            nBodiesKernel = &nBodiesKernelGlobal2D;
        #endif
    #elif defined LOCAL_MEMORY
        #ifdef ONE_DIM_BLOCK
            particleTimer = new ParticleTimer(clNumBodies, "local-1d", GROUP_SIZE, NUM_BODIES);
            nBodiesKernel = &nBodiesKernelLocal1D;
        #else
            particleTimer = new ParticleTimer(clNumBodies, "local-2d", GROUP_SIZE, NUM_BODIES);
            nBodiesKernel = &nBodiesKernelLocal2D;
        #endif
    #endif

    // create a context
    cl::Platform clPlatform = cl::Platform::getDefault();
    context = cl::Context(DEVICE);

    // get the command queue
    queue = cl::CommandQueue(context);

    // load in kernel source, creating a program object for the context
    program = cl::Program(context, load_program("../../../source/nbodies-opencl/kernel.cl"), true);

    // create the kernel functor
    nBodiesKernelLocal1D = cl::Kernel(program, "nBodiesKernelLocal1D");
    nBodiesKernelLocal2D = cl::Kernel(program, "nBodiesKernelLocal2D");
    nBodiesKernelGlobal1D = cl::Kernel(program, "nBodiesKernelGlobal1D");
    nBodiesKernelGlobal2D = cl::Kernel(program, "nBodiesKernelGlobal2D");

    // Init device data, copying data from host for positions, velocities and masses
    dPositions = cl::Buffer(context, CL_MEM_READ_WRITE, clNumBodies*sizeof(float3));
    dVelocities = cl::Buffer(context, CL_MEM_READ_WRITE, clNumBodies*sizeof(float3));
    dFuturePositions = cl::Buffer(context, CL_MEM_READ_WRITE, clNumBodies*sizeof(float3));
    dFutureVelocities = cl::Buffer(context, CL_MEM_READ_WRITE, clNumBodies*sizeof(float3));
    dMasses = cl::Buffer(context, CL_MEM_READ_WRITE, clNumBodies*sizeof(float));

    queue.enqueueWriteBuffer(dPositions, CL_TRUE, 0, NUM_BODIES*sizeof(float3), dataPositions);
    queue.enqueueWriteBuffer(dVelocities, CL_TRUE, 0, NUM_BODIES*sizeof(float3), dataVelocities);
    queue.enqueueWriteBuffer(dMasses, CL_TRUE, 0, NUM_BODIES*sizeof(float), dataMasses);
    queue.enqueueFillBuffer(dPositions, 0, NUM_BODIES*sizeof(float3), paddedBodies * sizeof(float3));
    queue.enqueueFillBuffer(dVelocities, 0, NUM_BODIES*sizeof(float3), paddedBodies * sizeof(float3));
    queue.enqueueFillBuffer(dMasses, 0, NUM_BODIES*sizeof(float), paddedBodies * sizeof(float));
    queue.finish();
    //dPositions = cl::Buffer(context, dataPositions, dataPositions+3*NUM_BODIES, false);
    //dVelocities = cl::Buffer(context, dataVelocities, dataVelocities+3*NUM_BODIES, false);
    //dMasses = cl::Buffer(context, dataMasses, dataMasses+NUM_BODIES, true);

}

void runSimulation() {  // runOpenCl

    // Prepare the kernel
    #ifdef ONE_DIM_BLOCK
        cl::NDRange global(clNumBodies);  // Total number of work items
        cl::NDRange local(GROUP_SIZE);  // Work items in each work-group
    #elif defined TWO_DIM_BLOCK
        cl::NDRange global(clNumBodiesX, clNumBodiesY);
        cl::NDRange local(groupSideSize, groupSideSize);

    #endif
    nBodiesKernel->setArg(0, dPositions);
    nBodiesKernel->setArg(1, dVelocities);
    nBodiesKernel->setArg(2, dFuturePositions);
    nBodiesKernel->setArg(3, dFutureVelocities);
    nBodiesKernel->setArg(4, dMasses);
    #ifdef GLOBAL_MEMORY
        nBodiesKernel->setArg(5, clNumBodies);
    #elif defined LOCAL_MEMORY
        nBodiesKernel->setArg(5, GROUP_SIZE * sizeof(cl_float4), nullptr);  // tileData
        nBodiesKernel->setArg(6, clNumBodies);
    #endif

    queue.enqueueNDRangeKernel(*nBodiesKernel, cl::NullRange, global, local);
    // Start timer iteration and run the kernel
    particleTimer->startIteration();
    (*nBodiesKernel)();
    queue.finish();

    particleTimer->endIteration();

    // Update positions and velocities for next iteration
    queue.enqueueCopyBuffer(dFuturePositions, dPositions, 0, 0, clNumBodies * 3 * sizeof(float));
    queue.enqueueCopyBuffer(dFutureVelocities, dVelocities, 0, 0, clNumBodies * 3 * sizeof(float));

    queue.finish();
}

int main(int argc, char** argv)
{
    // Fill host containers
    dataPositions = new float3[NUM_BODIES];
    dataVelocities = new float3[NUM_BODIES];
    dataMasses = new float[NUM_BODIES];
    loadData("../../../data/dubinski.tab", NUM_BODIES, (float*) dataPositions, (float*) dataVelocities, dataMasses, scaleFactor);

    // OpenCL setup
    initOpenCL();

    while (true) {
        runSimulation();
    }

    delete dataPositions;
    delete dataVelocities;
    delete dataMasses;

    return 0;
}