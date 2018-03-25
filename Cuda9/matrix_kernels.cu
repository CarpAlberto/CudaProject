#include "matrix_kernels.h"

#define TILE_DIM 16
__global__ void gpu_add(float* first, float* second, size_t sizeFirst)
{
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;

	while (threadId < sizeFirst) {
		first[threadId] = (first[threadId] +  second[threadId]);
		threadId += ( blockDim.x * gridDim.x );
	}
} 


__global__ void gpu_multiply(float* A, float* B, float* C,
	int ARows, int ACols, 
	int BRows, int BCols,
	int CRows, int CCols) {

	float CValue = 0;

	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + ACols - 1) / TILE_DIM; k++) {

		if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)	
			As[threadIdx.y][threadIdx.x] = A[Row*ACols + k * TILE_DIM + threadIdx.x];
		else													
			As[threadIdx.y][threadIdx.x] = 0.0;

		if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)	
			Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
		else													
			Bs[threadIdx.y][threadIdx.x] = 0.0;
		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) 
			CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

		__syncthreads();
	}

	if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue;
}

__global__ void gpu_mull2(float* a, float* b, float* c, int n, int m,int p)
{
	int i = blockIdx.x * 32 + threadIdx.x;
	int j = blockIdx.y;

	float sum = 0.0f;
	for (int k = 0; k < p; ++k) {
		sum += b[i + n * k] * c[k + p * j];

	}
	a[i + n * j] = sum;
}

__global__ void gpu_transpose(const float* src, float* dst, int colssrc, int colsdst, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < n) {
		int cdst = tid % colsdst;
		int rdst = tid / colsdst;
		int rsrc = cdst;
		int csrc = rdst;
		dst[tid] = src[rsrc * colssrc + csrc];
		tid += stride;
	}
}
