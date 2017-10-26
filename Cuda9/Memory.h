#pragma once
#include "includes.h"

namespace gpuNN{
	
	class Memory {
	
	public:
		Memory() {}
	public:
		/// <summary>
		/// The unique instance of the memory object
		/// </summary>
		/// <returns></returns>
		static Memory* instance()
		{
			static Memory* monitor = new Memory();
			return monitor;
		}
		/// /// <summary>
		/// Abstraction of malloc and cudaMalloc.
		/// </summary>
		/// <param name="pointer">The pointer where the memory will be stored</param>
		/// <param name="size">The size of allocation</param>
		/// <returns>The result of the operation</returns>
		void* allocate(size_t size,Bridge);
		/// <summary>
		/// Dealocates the memory
		/// </summary>
		/// <param name="ptr"></param>
		/// <param name=""></param>
		void deallocate(void* ptr, Bridge mode);
		/// <summary>
		/// Prints the memory usage
		/// </summary>
		void PrintMemoryUsage();
		/// <summary>
		/// Display the layout of the memory
		/// </summary>
		void PrintLayoutMemory();
	private:
		/// <summary>
		/// Custom allocator for cpu
		/// </summary>
		StackAllocator cpuAllocator;
		/// <summary>
		/// Custom allocator for gpu
		/// </summary>
		GpuAllocator gpuAllocator;

	};
}
