// ==================================
// Modified from NVIDIA CUDA examples
// ==================================

#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <math.h>
#include <cuda_runtime_api.h>

#include "data-loader.h"
#include "particle-timer.h"

// Number of particles to be loaded from file
#define NUM_BODIES 16384

// Block size
#define BLOCK_SIZE 256

// Memory configuration
// #define GLOBAL_MEMORY
#define LOCAL_MEMORY

// Block configuration
#define ONE_DIM_BLOCK
// #define TWO_DIM_BLOCK

// Number of CUDA blocks after padding
int numBlocks;

// Number of particles after padding
int numBodies;

// 2D dimensions
dim3 blockSize;
dim3 gridSize;

// Simulation parameters
float scaleFactor = 1.5f;

// Simulation data storage loaded from file
float* dataPositions = nullptr;
float* dataVelocities = nullptr;
float* dataMasses = nullptr;

float* hPositions = nullptr;
float* hVelocities = nullptr;
float* hMasses = nullptr;

float* dPositions = nullptr; // Device side particles positions
float* dVelocities = nullptr; // Device side particles velocities
float* dFuturePositions = nullptr; // Device side particles future positions
float* dFutureVelocities = nullptr; // Device side particles future velocities
float* dMasses = nullptr; // Device side particles masses

// Timer
ParticleTimer* particleTimer;

// Clamp macro
#define LIMIT(x,min,max) { if ((x)>(max)) (x)=(max); if ((x)<(min)) (x)=(min); }

// Forward declarations
void initCUDA();
void runCuda();

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

// Initialize CUDA data
void initCUDA()
{	
	// Round up in case NUM_BODIES is not a multiple of BLOCK_SIZE
	numBlocks = (NUM_BODIES + BLOCK_SIZE - 1) / BLOCK_SIZE;

	// Number of particles after padding, if there exists padding
	numBodies = numBlocks * BLOCK_SIZE;

	// 2D block calculations
	#ifdef TWO_DIM_BLOCK
		int blockSide = (int) sqrt(BLOCK_SIZE);
		if (blockSide * blockSide != BLOCK_SIZE) {
			std::cout << "BLOCK_SIZE has to be a perfect square" << std::endl;
			exit(1);
		}

		int gridSide = closestDivisorToSquareRoot(numBlocks);

		blockSize.x = blockSide; blockSize.y = blockSide;
		gridSize.x = gridSide; gridSize.y = numBlocks / gridSide;
	#endif

	// Define the timer
	#ifdef GLOBAL_MEMORY
		#ifdef ONE_DIM_BLOCK
			particleTimer = new ParticleTimer(numBodies, "global-1d", BLOCK_SIZE, NUM_BODIES);
		#else
			particleTimer = new ParticleTimer(numBodies, "global-2d", BLOCK_SIZE, NUM_BODIES);
		#endif
	#elif defined LOCAL_MEMORY
		#ifdef ONE_DIM_BLOCK
			particleTimer = new ParticleTimer(numBodies, "local-1d", BLOCK_SIZE, NUM_BODIES);
		#else
			particleTimer = new ParticleTimer(numBodies, "local-2d", BLOCK_SIZE, NUM_BODIES);
		#endif
	#endif

	hPositions = new float[numBodies * 3];
	hVelocities = new float[numBodies * 3];
	hMasses = new float[numBodies];

	// Apply padding in case of round up
	std::fill_n(hPositions, 3 * numBodies, 0.0f);
	memcpy(hPositions, dataPositions, 3 * NUM_BODIES * sizeof(float));
	std::fill_n(hVelocities, 3 * numBodies, 0.0f);
	memcpy(hVelocities, dataVelocities, 3 * NUM_BODIES * sizeof(float));
	std::fill_n(hMasses, numBodies, 0.0f);
	memcpy(hMasses, dataMasses, NUM_BODIES * sizeof(float));

	// Device particles data
	cudaMalloc((void**) &dPositions, 3 * numBodies * sizeof(float));
	cudaMalloc((void**) &dVelocities, 3 * numBodies * sizeof(float));
	cudaMalloc((void**) &dFuturePositions, 3 * numBodies * sizeof(float));
	cudaMalloc((void**) &dFutureVelocities, 3 * numBodies * sizeof(float));
	cudaMalloc((void**) &dMasses, numBodies * sizeof(float));

	// Copy initial values to GPU memory
	cudaMemcpy(dPositions, hPositions, 3 * numBodies * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dVelocities, hVelocities, 3 * numBodies * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dMasses, hMasses, numBodies * sizeof(float), cudaMemcpyHostToDevice);

}

// ========================================================================
// Start CUDA code
// ========================================================================

__device__
float3 bodyBodyInteraction(float3 iBody, float4 jData, float3 ai)
{
    float3 r{};
    r.x = jData.x - iBody.x;
    r.y = jData.y - iBody.y;
    r.z = jData.z - iBody.z;

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    float dist = sqrt(distSqr);
    float distCube = distSqr * dist;

    if (distCube < 1.f) return ai;

    float s = jData.w / distCube;

    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

__global__
void nBodiesKernelGlobal1D(float3* positions, float3* velocities, float3* futurePositions, float3* futureVelocities, float* masses, int nBodies)
{
	float dt = 0.001f;

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	float3 position = positions[x];
	float3 velocity = velocities[x];
	float3 acceleration = {.0f, .0f, .0f};

  int j;
  for (j = 0; j < nBodies; j++) {
    float3 jPosition = positions[j];
    float4 jData = make_float4(jPosition.x, jPosition.y, jPosition.z, masses[j]);
    acceleration = bodyBodyInteraction(position, jData, acceleration);
  }

	velocity.x += acceleration.x * dt;
	velocity.y += acceleration.y * dt;
	velocity.z += acceleration.z * dt;

	position.x += velocity.x * dt;
	position.y += velocity.y * dt;
	position.z += velocity.z * dt;

	futurePositions[x] = position;
	futureVelocities[x] = velocity;
}

__global__
void nBodiesKernelGlobal2D(float3* positions, float3* velocities, float3* futurePositions, float3* futureVelocities, float* masses, int nBodies)
{
	float dt = 0.001f;

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	int width = gridDim.x * blockDim.x;
	int height = gridDim.y * blockDim.y;

	unsigned int i = y * width + x;
	
	float3 position = positions[i];
	float3 velocity = velocities[i];
	float3 acceleration = {.0f, .0f, .0f};

  int j;
  for (j = 0; j < nBodies; j++) {
    float3 jPosition = positions[j];
    float4 jData = make_float4(jPosition.x, jPosition.y, jPosition.z, masses[j]);
    acceleration = bodyBodyInteraction(position, jData, acceleration);
  }

	velocity.x += acceleration.x * dt;
	velocity.y += acceleration.y * dt;
	velocity.z += acceleration.z * dt;

	position.x += velocity.x * dt;
	position.y += velocity.y * dt;
	position.z += velocity.z * dt;

	futurePositions[i] = position;
	futureVelocities[i] = velocity;
}

__global__
void nBodiesKernelLocal1D(float3* positions, float3* velocities, float3* futurePositions, float3* futureVelocities, float* masses, int nBodies)
{
	extern __shared__ float4 tileData[];

	float dt = 0.001f;

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	float3 position = positions[x];
	float3 velocity = velocities[x];
	float3 acceleration = {.0f, .0f, .0f};

	int k, tile;
	for (tile = 0; tile * blockDim.x < nBodies; tile++) {    
		int idx = tile * blockDim.x + threadIdx.x;     

		float3 jPosition = positions[idx];
		tileData[threadIdx.x] = make_float4(jPosition.x, jPosition.y, jPosition.z, masses[idx]);     

		__syncthreads();     
		
		for (k = 0; k < blockDim.x; k++) {     
			acceleration = bodyBodyInteraction(position, tileData[k], acceleration);   
		}

		__syncthreads();   
	}

	velocity.x += acceleration.x * dt;
	velocity.y += acceleration.y * dt;
	velocity.z += acceleration.z * dt;

	position.x += velocity.x * dt;
	position.y += velocity.y * dt;
	position.z += velocity.z * dt;

	futurePositions[x] = position;
	futureVelocities[x] = velocity;
}

__global__
void nBodiesKernelLocal2D(float3* positions, float3* velocities, float3* futurePositions, float3* futureVelocities, float* masses, int nBodies)
{
	extern __shared__ float4 tileData[];

	float dt = 0.001f;

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	int width = gridDim.x * blockDim.x;
	int height = gridDim.y * blockDim.y;

	unsigned int i = y * width + x;
	
	float3 position = positions[i];
	float3 velocity = velocities[i];
	float3 acceleration = {.0f, .0f, .0f};

	int k, tile;
	for (tile = 0; tile * blockDim.x * blockDim.y < nBodies; tile++) {
		int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
		int idx = tile * blockDim.x * blockDim.y + localIdx;     

		float3 jPosition = positions[idx];
		tileData[localIdx] = make_float4(jPosition.x, jPosition.y, jPosition.z, masses[idx]);     

		__syncthreads();     
		
		for (k = 0; k < blockDim.x * blockDim.y; k++) {     
			acceleration = bodyBodyInteraction(position, tileData[k], acceleration);   
		}

		__syncthreads();   
	}

	velocity.x += acceleration.x * dt;
	velocity.y += acceleration.y * dt;
	velocity.z += acceleration.z * dt;

	position.x += velocity.x * dt;
	position.y += velocity.y * dt;
	position.z += velocity.z * dt;

	futurePositions[i] = position;
	futureVelocities[i] = velocity;
}

void runCuda(void)
{
	// Start timer iteration
	particleTimer->startIteration();

	// Run the kernel
	#ifdef GLOBAL_MEMORY
		#ifdef ONE_DIM_BLOCK
			nBodiesKernelGlobal1D<<<numBlocks, BLOCK_SIZE>>>((float3*) dPositions, (float3*) dVelocities, (float3*) dFuturePositions, (float3*) dFutureVelocities, dMasses, numBodies);
		#else
			nBodiesKernelGlobal2D<<<gridSize, blockSize>>>((float3*) dPositions, (float3*) dVelocities, (float3*) dFuturePositions, (float3*) dFutureVelocities, dMasses, numBodies);
		#endif
	#elif defined LOCAL_MEMORY
		#ifdef ONE_DIM_BLOCK
			nBodiesKernelLocal1D<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float4)>>>((float3*) dPositions, (float3*) dVelocities, (float3*) dFuturePositions, (float3*) dFutureVelocities, dMasses, numBodies);
		#else
			nBodiesKernelLocal2D<<<gridSize, blockSize, BLOCK_SIZE * sizeof(float4)>>>((float3*) dPositions, (float3*) dVelocities, (float3*) dFuturePositions, (float3*) dFutureVelocities, dMasses, numBodies);	
		#endif
	#endif

	// Synchronize the device with host
	cudaDeviceSynchronize();

	// End timer iteration
	particleTimer->endIteration();

	// Update positions and velocities for next iteration
	cudaMemcpy(dPositions, dFuturePositions, 3 * numBodies * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dVelocities, dFutureVelocities, 3 * numBodies * sizeof(float), cudaMemcpyDeviceToDevice);
}

// ========================================================================
// End CUDA code
// ========================================================================

// ======================
//          Main         
// ======================
int main(int argc, char** argv)
{
	dataPositions = new float[NUM_BODIES * 3];
	dataVelocities = new float[NUM_BODIES * 3];
	dataMasses = new float[NUM_BODIES];
	// Data loading
	loadData("../../../data/dubinski.tab", NUM_BODIES, dataPositions, dataVelocities, dataMasses, scaleFactor);
    
	// CUDA setup
  initCUDA();
	
	while (true) {
		runCuda();
	}

	// Free heap memory
	free(dataPositions);
	free(dataVelocities);
	free(dataMasses);
	free(hPositions);
	free(hVelocities);
	free(hMasses);
	delete particleTimer;

  return 0;
}