#pragma once
#include "includes.h"
class TransferFunction {

public:
	/// <summary>
	/// Gets the value based on the input
	/// </summary>
	/// <param name="input">The input parameters</param>
	/// <returns>The transfer function applied on the input</returns>
	virtual double getValue(double input)=0;

	/// <summary>
	/// Returns the derivative of the function
	/// </summary>
	/// <param name="input">The derivative of the function</param>
	/// <returns>The derivative of the function</returns>
	virtual double getDerivative(double input)=0;
};