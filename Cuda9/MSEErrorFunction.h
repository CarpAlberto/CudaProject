#pragma once
#include "ErrorFunction.h"

namespace gpuNN {
	/// <summary>
	/// Mean Square  Error function
	/// </summary>
	class MSEErrorFunction : public ErrorFunction
	{
	public:
		/// <summary>
		/// Returns the error from the target and value  
		/// +
		/// </summary>
		/// <param name="value">The actual value</param>
		/// <param name="target">The target value</param>
		/// <returns>The MSE between those two values</returns>
		virtual double getError(double value, double target);
	};
}
