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
#include <atomic>
#include <initializer_list>
#include "include\parser-library\parse.h"
#include "include/word-to-vec/word2vec.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <immintrin.h>
#include <curand.h>
#include <cctype> 
#include <clocale>
#include <chrono>
#include <ctime>
#include <windows.h>


// Undef the max and use std::max
#ifdef max
	#undef max
#endif

//#define OPTIMIZED_GPU

/*
#ifndef __CUDACC_RTC__ 
#define __CUDACC_RTC__
#endif 
#ifndef __CUDACC__
#define __CUDACC__
#endif
*/
#define throwUns throw new std::exception("Unsuported operation");

#define threadsPerBlock 512

#ifndef TILE_DIM 
	#define TILE_DIM 16
#endif

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}
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


#pragma region Forward_Declaration

using namespace gpuNN;
using namespace std::chrono;
using namespace peparse;
enum class ErrorSurce;
enum class CpuError;
enum class SeverityType;
typedef unsigned char byte;

#pragma endregion


#define RC2IDX(R,C,COLS) (((R)*(COLS))+(C))

#ifndef DUMP_FIELD
#define DUMP_FIELD(x)                                                      \
  std::cout << "" #x << ": 0x";                                            \
  std::cout << to_string<std::uint32_t>(                                   \
                   static_cast<std::uint32_t>(p->peHeader.nt.x), std::hex) \
            << endl;
#endif

#ifndef DUMP_DEC_FIELD
#define DUMP_DEC_FIELD(x)                                                  \
  std::cout << "" #x << ": ";                                              \
  std::cout << to_string<uint32_t>(                                        \
                   static_cast<std::uint32_t>(p->peHeader.nt.x), std::dec) \
            << endl;

#endif

#ifndef ASSERT
#define ASSERT(cond) \
		if(!cond) {throw new std::exception("unsuported");}
#endif

/// <summary>
/// Typedef the vectore of integers for toplogy
/// </summary>
typedef std::vector<size_t>			   Topology;
typedef std::vector<double>			   vDouble;
typedef std::vector<vDouble>		   mDouble;
typedef std::vector<std::string>       vStrings ;
typedef float cudafloat;





 


