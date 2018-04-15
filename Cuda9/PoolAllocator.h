#pragma once
#include "includes.h"
#include "NonCopyableObject.h"
#include "NonMoveableObject.h"

namespace gpuNN {

	template <class T>
	class StackLinkedList {
	public:
		struct Node {
			T data;
			Node* next;
		};
		Node* head;
	public:
		StackLinkedList();
		void push(Node * newNode);
		Node* pop();
	private:
		StackLinkedList(StackLinkedList &stackLinkedList);
	};
	
	class PoolAllocator : public BaseAllocator {

	private:
		struct  FreeHeader {
		};
		typedef StackLinkedList<FreeHeader>::Node Node;
		StackLinkedList<FreeHeader> m_freeList;
		void * m_start_ptr = nullptr;
		std::size_t m_chunkSize;
	public:
		PoolAllocator(const std::size_t totalSize = 1677721600,
			const std::size_t chunkSize = 1024);
		virtual ~PoolAllocator();
		virtual void* Allocate(const std::size_t size, const std::size_t alignment = 0) override;
		virtual void Free(void* ptr) override;
		virtual void Init() override;
		virtual void Reset();
		size_t getTotalMemory();
		void PrintMemory();
		    
	private:
		PoolAllocator(PoolAllocator &poolAllocator);
	};

	template <class T>
	StackLinkedList<T>::StackLinkedList() {
	}

	template <class T>
	void StackLinkedList<T>::push(Node * newNode) {
		newNode->next = head;
		head = newNode;
	}

	template <class T>
	typename StackLinkedList<T>::Node* StackLinkedList<T>::pop() {
		Node * top = head;
		head = head->next;
		return top;
	}

}