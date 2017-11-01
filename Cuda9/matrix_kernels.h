#pragma once
#include "includes.h"

	/// <summary>
	/// Performs the addition of first with second  firts += second
	/// <code> sizeFirst</code> is the size of the matrix elements.
	/// </summary>
__global__ void gpu_add(float* first, float* second, size_t sizeFirst);


