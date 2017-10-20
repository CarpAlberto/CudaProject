#pragma once
#include "TransferFunction.h"
namespace gpuNN 
{
	class TanhTransferFunction :
		public TransferFunction
	{
	public:
		virtual double getValue(double input);
		virtual double getDerivative(double input);
	};
}
