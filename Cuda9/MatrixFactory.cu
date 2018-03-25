#include "MatrixFactory.h"



GenericMatrix* MatrixFactory::getMatrix(size_t rows, size_t columns) {

	ApplicationConfiguration* configuration = ApplicationContext::instance()->getConfiguration().get();
	auto mode = configuration->getMode();
	
	if (mode == "GPU"){
		return new GpuMatrix(rows, columns);
	}
	else {
		return new CpuMatrix(rows,columns);
	}
}

GenericMatrix* MatrixFactory::getMatrix(const GenericMatrix& rhs) {
	auto configuration = ApplicationContext::instance()->getConfiguration();
	auto mode = configuration->getMode();

	if (mode == "GPU") {
		return new GpuMatrix(rhs);
	}
	else {
		return new CpuMatrix(rhs);
	}
}