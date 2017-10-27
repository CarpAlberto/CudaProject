#include "GpuAllocator.h"

using namespace gpuNN;


GpuAllocator::GpuAllocator(const std::size_t totalSize):
	BaseAllocator(totalSize)
{
}

void* GpuAllocator::Allocate(const std::size_t size, const std::size_t alignment) {

	void* devPtr = 0;
	cudaError_t error = cudaMalloc(&devPtr, size);
	if (error != cudaError::cudaSuccess) {
		throw new MemoryAllocationException("Cuda Failed to allocate memory");
	}
	else 
	{
		m_offset += size;
		this->points[devPtr] = (double)size;
		return devPtr;
	}
}

void GpuAllocator::Free(void* ptr)
{
	cudaFree(&ptr);
}
void GpuAllocator::Reset()
{
	m_offset = 0;

}
void GpuAllocator::Init()
{

}

GpuAllocator::~GpuAllocator()
{
	Reset();
}
