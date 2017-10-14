#ifndef ALLOCATOR_H
#define ALLOCATOR_H
#include <cstddef> 

namespace gpuNN {
	/// <summary>
	/// Base Allocator for all the allocators
	/// </summary>
	class BaseAllocator {

	protected:
		/// <summary>
		/// The total size of the allocated memory
		/// </summary>
		std::size_t m_totalSize;
		/// <summary>
		/// The used memory
		/// </summary>
		std::size_t m_used;
		std::size_t m_peek;
	public:
		BaseAllocator() {}
		/// <summary>
		/// The constructor that takes the total size as parameter
		/// </summary>
		/// <param name="totalSize">The total size of the memory</param>
		BaseAllocator(const std::size_t totalSize);
		/// <summary>
		/// Destructor 
		/// </summary>
		virtual ~BaseAllocator();
		/// <summary>
		/// Allocates size data and returns a pointer to the begining of the memory area
		/// </summary>
		/// <param name="size"></param>
		/// <param name="alignment"></param>
		/// <returns></returns>
		virtual void* Allocate(const std::size_t size, const std::size_t alignment = 0) = 0;
		/// <summary>
		/// Free the memory given by the <code>ptr</code>
		/// </summary>
		/// <param name="ptr">The given pointer</param>
		virtual void Free(void* ptr) = 0;
		/// <summary>
		/// Performs the necessary initialization
		/// </summary>
		virtual void Init() = 0;

	};
}

#endif /* ALLOCATOR_H */