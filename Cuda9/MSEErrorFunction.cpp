#include "MSEErrorFunction.h"

using namespace gpuNN;

double MSEErrorFunction::getError(double value, double target) {

	return 0.5 * pow(abs((value - target)), 2);
}