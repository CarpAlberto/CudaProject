#pragma once

#include "includes.h"

namespace gpuNN {
	
	typedef struct ErrorSurce
	{
		cudaError gpuError;

		CpuError  cpuError;

		bool hasErrors()const;

		bool isCpuError()const;

		bool isGpuError()const;
	};
}