#include "TanhTransferFunction.h"
using namespace gpuNN;


double TanhTransferFunction::getValue(double input)
{
	return tanh(input);
}

double TanhTransferFunction::getDerivative(double input) {

	return 1 - (tanh(input) * tanh(input));
}