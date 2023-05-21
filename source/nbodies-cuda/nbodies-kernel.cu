#ifndef _GALAXY_KERNEL_H_
#define _GALAXY_KERNEL_H_


#define BSIZE 256
#define softeningSquared 0.01f		// original plumer softener is 0.025. here the value is square of it.
#define damping 1.0f				// 0.999f
#define ep 0.67f						// 0.5f


/**
 * Calculate pull acceleration that gravity of body j induces into body i
 *
 * @param bi body i
 * @param bj body j
 * @param ai acceleration of body i
 * @return
 */
__device__ float3
bodyBodyInteraction(float4 bi, float4 bj, float3 ai)
{
    float3 r;

    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += softeningSquared;
        
   	float dist = sqrtf(distSqr);
   	float distCube = dist * dist * dist;

    // if distances too small, maintain acceleration
	if (distCube < 1.0f) return ai;
    // otherwise, update acceleration
	
    float s = bi.w / distCube;
    //float s = 1.0f / distCube;
    
    ai.x += r.x * s * ep;
    ai.y += r.y * s * ep;
    ai.z += r.z * s * ep;

    return ai;
}

__device__ float3
tile_calculation(float4 myPosition, float3 acc)
{
    // Calculate the interactions between bodies in the shared memory, not in between all bodies.
	extern __shared__ float4 shPosition[];
	
	#pragma unroll 8
    // Update over and over again the acceleration, for each interaction with this body.
	for (unsigned int i = 0; i < BSIZE; i++)
		acc = bodyBodyInteraction(myPosition, shPosition[i], acc);
		
	return acc;
}

/**
 * Updates the position and velocity of the thread's assigned body.
 * Each thread worries about a particle.
 *
 * @param pos
 * @param pdata
 * @param width
 * @param height
 * @param step
 * @param apprx
 * @param offset
 */
__global__ void 
galaxyKernel(float4* pos, float4 * pdata, unsigned int width, 
			 unsigned int height, float step, int apprx, int offset)
{
	// shared memory
	extern __shared__ float4 shPosition[];
	
	// index of my body	
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pLoc = y * width + x;
    unsigned int vLoc = width * height + pLoc;
	
    // starting index of position array
    unsigned int start = ( (width * height) / apprx ) * offset;
	
	float4 myPosition = pdata[pLoc];
	float4 myVelocity = pdata[vLoc];

	float3 acc = {0.0f, 0.0f, 0.0f};

	unsigned int idx = 0;
	unsigned int loop = ((width * height) / apprx ) / BSIZE;
	for (int i = 0; i < loop; i++)
	{
        // Each thread copy part of pdata into the shared memory shPosition
		idx = threadIdx.y * blockDim.x + threadIdx.x;
		shPosition[idx] = pdata[idx + start + BSIZE * i];

		__syncthreads();

        // After all ended copying, we get the acceleration of the body
        // obtained when interacting with all the bodies in shared memory
		acc = tile_calculation(myPosition, acc);

        // Wait until all threads computed their subsets of interactions
		__syncthreads();

        // In each iteration, the threads copy different parts of the memory into the shared memory, so then
        // they update the acceleration gained of the body when interacting with the new bodies in the shared memory
	}

    // As now, we obtained the total acceleration of the current body, so we can update the position and velocity

    // update velocity with above acc
    myVelocity.x += acc.x * step;// * 2.0f;
    myVelocity.y += acc.y * step;// * 2.0f;
    myVelocity.z += acc.z * step;// * 2.0f;
    
    myVelocity.x *= damping;
    myVelocity.y *= damping;
    myVelocity.z *= damping;
    
    // update position
    myPosition.x += myVelocity.x * step;
    myPosition.y += myVelocity.y * step;
    myPosition.z += myVelocity.z * step;
        
    __syncthreads();
    
    // update device memory
	pdata[pLoc] = myPosition;
	pdata[vLoc] = myVelocity;
    
	// update vbo
	pos[pLoc] = make_float4(myPosition.x, myPosition.y, myPosition.z, 1.0f);
	pos[vLoc] = myVelocity;
}

extern "C" 
void cudaComputeGalaxy(float4* pos, float4 * pdata, int width, int height, 
					   float step, int apprx, int offset)
{
    dim3 block(16, 16, 1);
    dim3 grid(width / block.x, height / block.y, 1);
    int sharedMemSize = BSIZE * sizeof(float4);
    galaxyKernel<<<grid, block, sharedMemSize>>>(pos, pdata, width, height, 
    											 step, apprx, offset);
}

#endif