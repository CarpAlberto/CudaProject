#pragma once
#include "includes.h"

/// <summary>
/// Performs the addition of first with second  firts += second
/// <code> sizeFirst</code> is the size of the matrix elements.
/// </summary>
__global__ void gpu_add(float* first, float* second, size_t sizeFirst);

/// <summary>
/// Performs the multiplication of the matrix stored inside the <code>first</code>
/// and <code>second</code>.The result is being stored inside the matrix
/// <code>result</code>
/// </summary>
__global__ void gpu_multiply(float* A, float* B, float* C,
	int rowsa, int colsa,
	int rowsb, int colsb,
	int rowsc, int colsc);

/// <summary>
/// Perform the transpose of given matrix
/// </summary>
__global__ void gpu_transpose(const float* src, float* dst, int colssrc, int colsdst, int n);

