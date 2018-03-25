#pragma once
#include "includes.h"


namespace gpuNN {
	
	/// /// <summary>
	/// The Stack Allocator that 
	/// </summary>
	class StackAllocator : public BaseAllocator
	{
	protected:
		void* m_start_ptr = nullptr;
		/// <summary>
		/// The current offset of the memory
		/// </summary>
		std::size_t m_offset;
	public: 
		/// <summary>
		/// The constructor
		/// </summary>
		/// <param name="totalSize">Total Size of memory</param>
		StackAllocator(const std::size_t totalSize = 50000000);
		/// <summary>
		/// The destructor of the memory
		/// </summary>
		virtual ~StackAllocator();
		/// <summary>
		/// Allocates the memory using the Stack strategy
		/// </summary>
		/// <param name="size">The size to be allocated</param>
		/// <param name="alignment">The memory alignament</param>
		/// <returns>A pointer to the newly allocated memory</returns>
		virtual void* Allocate(const std::size_t size, const std::size_t alignment = 8) override;
		/// <summary>
		/// Free up the memorygiven by the <code>ptr</code> parameters
		/// </summary>
		/// <param name="ptr"></param>
		virtual void Free(void* ptr);
		/// <summary>
		/// Performs the necessary initialization
		/// </summary>
		virtual void Init() override;
		/// <summary>
		/// Reset the data
		/// </summary>
		virtual void Reset();
		/// <summary>
		/// Returns the total memory
		/// </summary>
		/// <returns></returns>
		size_t getTotalMemory();
		/// <summary>
		/// Display the memory
		/// </summary>
		void PrintMemory();

		size_t getUnusedMemory();
	private:
		struct AllocationHeader {
			char padding;
		};

	};
}
