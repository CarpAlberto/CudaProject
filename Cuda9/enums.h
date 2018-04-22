#pragma once

#include "includes.h"

namespace gpuNN {
	
	/// <summary>
	/// The possible errors that the cpu may throw
	/// </summary>
	enum class CpuError {
		/// <summary>
		/// No error was throw from cpu
		/// </summary>
		SUCCESS=0,
		/// <summary>
		/// Memory allocation failed on cpu
		/// </summary>
		MEMORY_ALLOCATION_FAILED=1
	};
	/// <summary>
	/// Utility enum from bridging the GPU and CPU mode
	/// </summary>
	enum class Bridge {
		CPU,
		GPU
	};
	/// <summary>
	/// Logging severity
	/// </summary>
	enum class SeverityType {
		DEBUG,
		WARNING,
		CUDA_ERROR
	};
	/// <summary>
	/// Transfer Functions
	/// </summary>
	enum class TransferFunctionType {
		TANH
	};
	/// <summary>
	/// Type of Error
	/// </summary>
	enum class FunctionErrorType {
		MSE
	};

	/// <summary>
	/// Strategy usef for saving the object
	/// </summary>
	enum class IOStrategy {
		XML,
		JSON,
		ASCII,
		BINARY
	};

	enum class NeuronTypeData {
		DATA,
		ACTIVATED_DATA,
		DERIVED_DATA
	};
	typedef enum {
		RowMajor,
		ColumnMajor
	} StoringOrder;
}