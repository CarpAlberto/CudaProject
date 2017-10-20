#include "TransferFunctionFactory.h"

using namespace gpuNN;

TransferFunction* TransferFunctionFactory::getTransferFunction(TransferFunctionType transferFunctionType) {
	switch (transferFunctionType) {
		case TransferFunctionType::TANH:
			return new TanhTransferFunction();
		default:
			break;
	}
}
