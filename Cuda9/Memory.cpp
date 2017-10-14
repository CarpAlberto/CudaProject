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

void Memory::deallocate(void* ptr, Bridge strategy) {
	
	switch (strategy)
	{
		case Bridge::CPU:
			return cpuAllocator.Free(ptr);
		case Bridge::GPU:
			return gpuAllocator.Free(ptr);
		default:
			throw new MemoryAllocationException("Invalid strategy on memory management");
	}
}