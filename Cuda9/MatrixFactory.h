#pragma once
#include "includes.h"
#include "matrix.h"
namespace gpuNN {
	class MatrixFactory
	{
	public:
		/// <summary>
		/// Gets a matrix based on the configuration File specified in application context
		/// </summary>
		/// <param name="rows">The rows number</param>
		/// <param name="columns">The column numbers</param>
		/// <returns>The matrix</returns>
		GenericMatrix* getMatrix(int rows, int columns);
	public:
		MatrixFactory() = default;
		~MatrixFactory() = default;


	};
}