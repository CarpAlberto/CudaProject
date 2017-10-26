#pragma once
#include "includes.h"

namespace gpuNN {

	/// <summary>
	/// Generic Error class
	/// </summary>
	class ErrorFunction {

	public:
		/// <summary>
		/// Returns the error from the target and value
		/// </summary>
		/// <param name="value">The actual value</param>
		/// <param name="target">The target value</param>
		/// <returns>The MSE between those two values</returns>
		virtual double getError(double value, double target) = 0;
	};
}