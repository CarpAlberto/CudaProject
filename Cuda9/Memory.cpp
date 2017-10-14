#include "memory.h"
using namespace gpuNN;

void* Memory::allocate(size_t size, Bridge strategy)
{
	switch (strategy)
	{
		case Bridge::CPU:
			return cpuAllocator.Allocate(size);
		case Bridge::GPU:
			return gpuAllocator.Allocate(size);
		default:
			throw new MemoryAllocationException("Invalid strategy on memory management");
	}
}