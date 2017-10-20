#pragma once
#include "TanhTransferFunction.h"

namespace gpuNN 
{
	class TransferFunctionFactory
	{
	public:
		TransferFunction* getTransferFunction(TransferFunctionType);
	};
}

