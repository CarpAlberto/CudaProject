#pragma once
#include "TanhTransferFunction.h"
#include "MSEErrorFunction.h"

namespace gpuNN 
{
	/// <summary>
	/// The Function Factory
	/// </summary>
	class FunctionFactory
	{

	public:
		/// <summary>
		/// Returns the unique instance of the  Factory
		/// </summary>
		/// <returns>A pointer to the instance</returns>
		static FunctionFactory* instance();
		/// <summary>
		/// Returns the Transfer Function specified by the parameter
		/// </summary>
		/// <param name="type">The Type of the transfer function</param>
		/// <returns>A pointer to the neew object</returns>
		TransferFunction* getTransferFunction(TransferFunctionType type);
		/// <summary>
		/// Returns a pointer to the Error function provided by the <code>type</code>
		/// </summary>
		/// <param name="type">The function error type</param>
		/// <returns>A pointer to the error function</returns>
		ErrorFunction*	  getErrorFunction(FunctionErrorType type);
	};
}

