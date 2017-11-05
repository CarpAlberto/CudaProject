#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>
#include <algorithm>
#include <random>
#include <thread>
#include <mutex>
#include <map>
#include <chrono>
//#include "include/json.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <immintrin.h>
#include <curand.h>

/*Cuda imports*/

#ifndef __CUDACC_RTC__ 
#define __CUDACC_RTC__
#endif 
#ifndef __CUDACC__
#define __CUDACC__
#endif


#define threadsPerBlock 512

#include "enums.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <memory>
#include <fstream>
#include <sstream>


#include "Allocator.h"
#include "GpuAllocator.h"
#include "StackAllocator.h"
#include "PoolAllocator.h"
#include "PrintableObject.h"
#include "NonCopyableObject.h"
#include "NonMoveableObject.h"
#include "MemoryAllocationException.h"
#include "Utils.h"
#include "ApplicationContext.h"
#include "UI.h"
#include "matrix_kernels.h"

/*Forward declaration*/
using namespace gpuNN;
using namespace std::chrono;


enum class ErrorSurce;
enum class CpuError;
enum class SeverityType;
typedef unsigned char byte;


#define RC2IDX(R,C,COLS) (((R)*(COLS))+(C))

/// <summary>
/// Typedef the vectore of integers for toplogy
/// </summary>
typedef std::vector<size_t>			   Topology;
typedef std::vector<double>			   vDouble;
typedef std::vector<vDouble>		   mDouble;


//using json = nlohmann::json;






 


