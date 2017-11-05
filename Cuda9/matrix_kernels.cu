#include "matrix_kernels.h"

__global__ void gpu_add(float* first, float* second, size_t sizeFirst)
{
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;

	while (threadId < sizeFirst) {
		first[threadId] = (first[threadId] +  second[threadId]);
		threadId += ( blockDim.x * gridDim.x );
	}
} 