#pragma once
#include <string.h>
#include <exception>
#include "settings.h"

extern "C"
{
	/// <summary>
	/// Runs all the constructors tests from the 
	/// </summary>
	EXPORT char* run_constructor_cpu();

	/// <summary>
	/// Runs all the constructors from the gpu
	/// </summary>
	EXPORT char* run_constructor_gpu();
}