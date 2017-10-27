#include "PoolAllocator.h"
#include <algorithm>
using namespace gpuNN;

PoolAllocator::~PoolAllocator() {
	free(m_start_ptr);
}

void PoolAllocator::Init() {
	m_start_ptr = malloc(m_totalSize);
	this->Reset();
}

PoolAllocator::PoolAllocator(const std::size_t totalSize, const std::size_t chunkSize)
	: BaseAllocator(totalSize) {
	this->m_chunkSize = chunkSize;
	Init();
}

void *PoolAllocator::Allocate(const std::size_t allocationSize, const std::size_t alignment) {

	Node * freePosition = m_freeList.pop();
	m_used += m_chunkSize;
	this->m_peek = std::max(this->m_peek, m_used);
	return (void*)freePosition;

}

void PoolAllocator::Free(void * ptr) {
	m_used -= m_chunkSize;
	m_freeList.push((Node *)ptr);
}


void PoolAllocator::Reset() {

	m_used = 0;
	this->m_peek = 0;
	const int nChunks = m_totalSize / m_chunkSize;
	for (int i = 0; i < nChunks; ++i) {
		std::size_t address = (std::size_t) m_start_ptr + i * m_chunkSize;
		m_freeList.push((Node *)address);
	}
}

size_t PoolAllocator::getTotalMemory() {
	return this->m_used;
}

void PoolAllocator::PrintMemory() {
	
}



