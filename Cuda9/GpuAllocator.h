#pragma once
#include "includes.h"

namespace gpuNN {
	
	class BaseAllocator;

	/// /// <summary>
	/// Manage the GPU allocation-
	/// </summary>
	class GpuAllocator : public BaseAllocator
	{
	protected:
		size_t m_offset;
		std::unordered_map<void*, double> points;
	public:
		GpuAllocator() = default;
		/// <summary>
		/// The constructor
		/// </summary>
		/// <param name="totalSize">Total Size of memory</param>
		GpuAllocator(const std::size_t totalSize);
		/// <summary>
		/// The destructor of the memory
		/// </summary>
		virtual ~GpuAllocator();
		/// <summary>
		/// Allocates the memory using the Stack strategy
		/// </summary>
		/// <param name="size">The size to be allocated</param>
		/// <param name="alignment">The memory alignament</param>
		/// <returns>A pointer to the newly allocated memory</returns>
		virtual void* Allocate(const std::size_t size, const std::size_t alignment = 0) override;
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
		/// Resets the mem-ory
		/// </summary>
		virtual void Reset();
	};

}