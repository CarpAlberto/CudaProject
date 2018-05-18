#pragma once

#include <float.h>
#include <cmath>
#include "includes.h"

typedef float cudaReal;

#define CUDA_MAX_THREADS_PER_BLOCK 1024
#define CUDA_MAX_GRID_X_DIM 65535
#define CUDA_MAX_GRID_Y_DIM 65535
#define NumberThreadsPerBlockThatBestFit(threads,maxThreadsPerBlock)\
	int numberThreads = 1;\
	while (numberThreads < threads && numberThreads < maxThreadsPerBlock) numberThreads <<= 1;\
	return numberThreads;\

#define NumberBlocks(threads,blockSize) \
	int numberBlocks = threads / blockSize;\
	if (threads % blockSize != 0) numberBlocks++;\
	return numberBlocks;\

#define CUDA_VALUE(X) (X##f)
#define CUDA_EXP  expf
#define sigmoid(X) (CUDA_VALUE(1.0) / (CUDA_VALUE(1.0) + CUDA_EXP(-(X))))
#define sigmoid_derivate(x) (x * ( 1 - x ))
#define CUDA_SIGMOID_DERIVATE(OUTPUT) ((OUTPUT) * (CUDA_VALUE(1.0) - (OUTPUT)))
#define same(X, Y) (((X) > 0.0 && (Y) > 0.0) || ((X) < 0.0 && (Y) < 0.0))

#define BLOCK_SIZE 128

namespace gpuNN {
	inline bool IsInfOrNaN(float x) {
#if (defined(_MSC_VER))
		return (!isfinite(x));
#else
		return (std::isnan(x) || std::isinf(x));
#endif
	}

	inline int BestFit(int threads, int maxThreadsPerBlock = 512) {
		int nt = 1;
		while (nt < threads && nt < maxThreadsPerBlock)
			nt <<= 1;
		return nt;
	}

	inline void NoMuchThreads(dim3 & block) {
		unsigned x = BestFit(block.x);
		unsigned y = BestFit(block.y);
		unsigned z = BestFit(block.z);

		while (x * y * z > 512) {
			if (z > 1 && z >= y) {
				z >>= 1;
			}
			else if (y >= x) {
				y >>= 1;
			}
			else {
				x >>= 1;
			}
		}
		if (z < block.z) block.z = z;

		while (2 * x * y * block.z < 512) {
			if (x < block.x) {
				if (y < x && y < block.y) {
					y <<= 1;
				}
				else {
					x <<= 1;
				}
			}
			else if (y < block.y) {
				y <<= 1;
			}
			else {
				break;
			}
		}

		if (y < block.y) block.y = y;

		while (x < block.x && 2 * x * y * z < 512) {
			x <<= 1;
		}

		if (x < block.x) block.x = x;
	}
}