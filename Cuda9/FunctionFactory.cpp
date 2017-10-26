#include "FunctionFactory.h"

using namespace gpuNN;

FunctionFactory* FunctionFactory::instance() {
	static FunctionFactory* transfer = new FunctionFactory();
	return transfer;
}

TransferFunction* FunctionFactory::getTransferFunction(TransferFunctionType transferFunctionType) {

	switch (transferFunctionType) {
		case TransferFunctionType::TANH:
			return new TanhTransferFunction();
		default:
			ApplicationContext::instance()->getLog()->print<SeverityType::WARNING>("Invalid transfer function");
			return nullptr;
	}
}

ErrorFunction*	  FunctionFactory::getErrorFunction(FunctionErrorType type) {
	
	switch (type) {
	case FunctionErrorType::MSE:
		return new MSEErrorFunction();
	default:
		ApplicationContext::instance()->getLog()->print<SeverityType::WARNING>("Invalid error function");
		return nullptr;
	}
}