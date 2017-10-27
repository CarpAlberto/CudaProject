#include "memory.h"
using namespace gpuNN;

Memory::Memory()
{

}
Memory* Memory::instance()
{
	static Memory* monitor = new Memory();
	return monitor;
}
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

void Memory::PrintMemoryUsage() {

	std::cout << "---Cpu Memory usage---" << std::endl;
	std::cout << this->cpuAllocator.getTotalMemory() << " Bytes " << std::endl;
}

void Memory::PrintLayoutMemory() {
	this->cpuAllocator.PrintMemory();
}