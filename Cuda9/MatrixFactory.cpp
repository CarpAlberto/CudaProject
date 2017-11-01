#include "MatrixFactory.h"



GenericMatrix* MatrixFactory::getMatrix(int rows, int columns) {

	auto configuration = ApplicationContext::instance()->getConfiguration();
	auto mode = configuration->Value("GeneralSettings", "MODE");
	
	if (mode == "GPU"){
		return new GpuMatrix(rows, columns);
	}
	else {
		return new CpuMatrix(rows,columns);
	}
}

GenericMatrix* getMatrix(const GenericMatrix& rhs) {
	auto configuration = ApplicationContext::instance()->getConfiguration();
	auto mode = configuration->Value("GeneralSettings", "MODE");

	if (mode == "GPU") {
		return new GpuMatrix(rhs);
	}
	else {
		return new CpuMatrix(rhs);
	}
}