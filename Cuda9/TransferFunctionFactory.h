#pragma once
#include "TanhTransferFunction.h"

namespace gpuNN 
{
	class TransferFunctionFactory
	{

	public:
		static TransferFunctionFactory* instance() {
			static TransferFunctionFactory* transfer = new TransferFunctionFactory();
			return transfer;
		}
		TransferFunction* getTransferFunction(TransferFunctionType);
	};
}

