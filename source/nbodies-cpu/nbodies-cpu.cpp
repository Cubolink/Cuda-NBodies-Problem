#include <iostream>

#include <cstdlib>
#include <cstdio>

#include "data-loader.h"
#include "data-structs.h"
#include "particle-timer.h"
#include "simulation.h"

// Number of particles to be rendered
#define NUM_BODIES 4096

// Simulation parameters
float scaleFactor = 1.5f;

// Simulation data
float3 *dataPositions = nullptr;
float3 *dataVelocities = nullptr;
float *dataMasses = nullptr;

float3 *futurePositions = nullptr;
float3 *futureVelocities = nullptr;

ParticleTimer* timer = new ParticleTimer(NUM_BODIES, "", 0, NUM_BODIES);

// ======================
//          Main 
// ======================
int main(int argc, char** argv)
{
	// Data loading
	dataPositions = new float3[NUM_BODIES];
	dataVelocities = new float3[NUM_BODIES];
	dataMasses = new float[NUM_BODIES];
	loadData("../../../data/dubinski.tab", NUM_BODIES, (float*) dataPositions, (float*) dataVelocities, dataMasses, scaleFactor);

	// Locate data for future positions and velocities
	futurePositions = new float3[NUM_BODIES];
	futureVelocities = new float3[NUM_BODIES];

	while (true) {
		cpuComputeNBodies(dataPositions, dataVelocities, dataMasses, futurePositions, futureVelocities, NUM_BODIES, timer);
	}

	return 0;
}