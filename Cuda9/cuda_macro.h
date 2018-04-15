#pragma once

#include <float.h>
#include <cmath>

typedef float cudaReal;

#define CUDA_MAX_THREADS_PER_BLOCK 512
#define CUDA_MAX_GRID_X_DIM 65535

#define NumberThreadsPerBlockThatBestFit(threads,maxThreadsPerBlock)\
	int numberThreads = 1;\
	while (numberThreads < threads && numberThreads < maxThreadsPerBlock) numberThreads <<= 1;\
	return numberThreads;\

#define NumberBlocks(threads,blockSize) \
	int numberBlocks = threads / blockSize;\
	if (threads % blockSize != 0) numberBlocks++;\
	return numberBlocks;\

#define sigmoid(x) ( 1.0 / (1 + expf(-x)))
#define sigmoid_derivate(x) (x * ( 1 - x ))
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
}