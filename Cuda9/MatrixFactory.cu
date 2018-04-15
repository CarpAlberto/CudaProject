#include "MatrixFactory.h"



GenericMatrix* MatrixFactory::getMatrix(size_t rows, size_t columns) {

	ApplicationConfiguration* configuration = ApplicationContext::instance()->getConfiguration().get();
	auto mode = configuration->getMode();
	if (mode == "GPU")
	{
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

GenericMatrix* MatrixFactory::getMatrix(cudaObject* obj,size_t rows,size_t cols, NeuronTypeData data = NeuronTypeData::DATA)
{
	auto configuration = ApplicationContext::instance()->getConfiguration();
	auto mode = configuration->getMode();

	if (mode == "GPU") 
	{
		/*
		if(data == NeuronTypeData::DATA)
			return new GpuMatrix(obj->Data(),obj->Size(),rows,cols);
		if (data == NeuronTypeData::ACTIVATED_DATA)
			return new GpuMatrix(obj->Activated(), obj->Size(), rows, cols);
		if (data == NeuronTypeData::DERIVED_DATA)
			return new GpuMatrix(obj->Derived(), obj->Size(), rows, cols);
		*/
	}
	throwUns
}