#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>
#include <algorithm>
#include <thread>
#include <mutex>
#include <chrono>

/*Cuda imports*/
#include <cuda_runtime.h>
#include <cuda.h>
#include "enums.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <memory>
#include <fstream>
#include <sstream>

#include "Allocator.h"
#include "GpuAllocator.h"
#include "StackAllocator.h"
#include "NonCopyableObject.h"
#include "NonMoveableObject.h"
#include "MemoryAllocationException.h"
#include "Utils.h"
#include "ApplicationContext.h"


/*Forward declaration*/
using namespace gpuNN;

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




 


