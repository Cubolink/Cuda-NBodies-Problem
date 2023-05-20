#ifndef _GALAXY_KERNEL_H_
#define _GALAXY_KERNEL_H_


#define BSIZE 256
#define softeningSquared 0.01f		// original plumer softener is 0.025. here the value is square of it.
#define damping 1.0f				// 0.999f
#define ep 0.67f						// 0.5f

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

	if (distCube < 1.0f) return ai;
	
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
	extern __shared__ float4 shPosition[];
	
	#pragma unroll 8
	for (unsigned int i = 0; i < BSIZE; i++)
		acc = bodyBodyInteraction(myPosition, shPosition[i], acc);
		
	return acc;
}

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
		idx = threadIdx.y * blockDim.x + threadIdx.x;
		shPosition[idx] = pdata[idx + start + BSIZE * i];

		__syncthreads();
		
		acc = tile_calculation(myPosition, acc);
		
		__syncthreads();		
	}
    	
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