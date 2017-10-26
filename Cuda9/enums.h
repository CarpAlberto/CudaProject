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
		ERROR
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

	
}