#pragma once
#include "includes.h"
#include "matrix.h"

namespace gpuNN {

	class cudaObject;

	class MatrixFactory : NonCopyableObject,
						  NonMoveableObject
	{
	public:
		/// <summary>
		/// Gets a matrix based on the configuration File specified in application context
		/// </summary>
		/// <param name="rows">The rows number</param>
		/// <param name="columns">The column numbers</param>
		/// <returns>The matrix</returns>
		static GenericMatrix* getMatrix(size_t rows, size_t columns);

		/// <summary>
		/// Returns the matrix
		/// </summary>
		/// <param name="obj"></param>
		/// <param name="size"></param>
		/// <returns></returns>
		static GenericMatrix* getMatrix(cudaObject* obj, size_t rows, size_t cols, NeuronTypeData);
		/// <summary>
		/// Returns an matrix based on the the rhs parameter
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		/// <returns>A pointer to the new Created matrix</returns>
		static GenericMatrix* getMatrix(const GenericMatrix&);
	public:
		MatrixFactory() = delete;
		~MatrixFactory() = delete;



	};
}