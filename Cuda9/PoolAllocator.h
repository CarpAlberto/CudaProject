#pragma once
#include "includes.h"
#include "NonCopyableObject.h"
#include "NonMoveableObject.h"

template<typename T>
union PoolChunk 
{
	T value;
	PoolChunk<T>* nextPoolChunk;
	PoolChunk() {}
	~PoolChunk() {};
};

template<typename T>
class PoolAllocator : public NonCopyableObject,
							 NonMoveableObject
{
private:
	static const size_t POOL_SIZE = 1024;
	size_t m_size = 0;
	PoolChunk<T>* m_data=nullptr;
	PoolChunk<T>* m_head = nullptr;
public:
	explicit PoolAllocator(size_t size = POOL_SIZE) 
	: m_size(size)
	{
		m_data = new PoolChunk<T>[size];
		m_head = m_data;
		for (size_t i = 0; i < m_size-1; i++) {
			m_data[i].nextPoolChunk = std::addressof(m_data[i + 1]);
		}
		m_data[m_size - 1].nextPoolChunk = nullptr;
	}
	
	~PoolAllocator()
	{
		delete[] m_data;
		m_data = nullptr;
		m_head = nullptr;
	}
	
	template<typename... arguments>
	T* allocate(arguments&& ... args) {
		if (m_head == nullptr)
			return nullptr;
		PoolChunk<T>* poolChunk = m_head;
		m_head = m_head->nextPoolChunk;
		T* retVal = new (std::addressof(poolChunk->value)) 
			T(std::forward<arguments>(args)...);
		return retVal;
	}

	void deallocate(T* data) {
		data->~T();
		PoolChunk<T>* poolChunk = reinterpret_cast<PoolChunk<T>*>(data);
		poolChunk->nextPoolChunk = m_head;
		m_head = poolChunk;
	}
};

