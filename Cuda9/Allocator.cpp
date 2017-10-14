#include "Allocator.h"

using namespace gpuNN;


BaseAllocator::BaseAllocator(const std::size_t totalSize) {
	m_totalSize = totalSize;
	m_used = 0;
}

BaseAllocator::~BaseAllocator() {
	m_totalSize = 0;
}