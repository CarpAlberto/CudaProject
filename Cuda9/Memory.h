#pragma once
#include "includes.h"

namespace gpuNN{
	
	/*Deprecated.Use Memory Manager for future Allocation*/
	class Memory {
	
	public:
		float LastSessionTime = 0;
	public:
		/// <summary>
		/// Default Constructor
		/// </summary>
		Memory();
	public:
		/// <summary>
		/// The unique instance of the memory object
		/// </summary>
		/// <returns></returns>
		static Memory* instance();
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
		PoolAllocator cpuAllocator;
		/// <summary>
		/// Custom allocator for gpu
		/// </summary>
		GpuAllocator gpuAllocator;

	};

	template <class Type> 
	class BaseMemoryManager {
	
	protected:
		Type * mData;
		size_t mSize;

	protected:
		void Reset() {
			this->mData = nullptr;
			this->mSize = 0;
		}

		BaseMemoryManager() {
			this->Reset();
		}

	public:

		virtual void Allocate(size_t size) = 0;

		virtual void Delete() = 0;

		virtual void CopyFromDevice(Type * data, size_t size) = 0;

		virtual void CopyFromHost(Type * data, size_t size) = 0;

		size_t Size() const {
			return this->mSize;
		}

		size_t SizeInBytes() const {
			return this->mSize * sizeof(Type);
		}

		Type * Data() const {
			return this->mData;
		}

		size_t Resize(size_t size) {
			
			if (size != this->mSize) {
				this->Delete();
				this->Allocate(size);
			}

			return this->mSize;
		}

		void TransferOwnerShipFrom(BaseMemoryManager<Type> & other) {
			if (this != &other) {
				Delete();
				this->mData = other.mData;
				this->mSize = other.mSize;
				other.Reset();
			}
		}
	};

	template <class Type> class HostMemoryManager :
		public BaseMemoryManager<Type> {
	public:
		virtual void Allocate(size_t size) {
			
			if (size > 0) 
			{
				this->mData = new (std::nothrow) Type[size];
				this->mSize = (this->mData != nullptr) ? size : 0;
			}
			else 
			{
				this->Reset();
			}
		}

		virtual void Delete() 
		{
			if (this->mSize > 0)
				delete[] this->mData;
			this->Reset();
		}

		virtual void CopyFromDevice(Type * data, size_t size) {
			
			this->Resize(size);
			if (this->mSize > 0) {
				cudaMemcpy(this->mData, data, this->SizeInBytes(), cudaMemcpyDeviceToHost);
			}
		}

		virtual void CopyFromHost(Type * data, size_t size) 
		{
			this->Resize(size);
			if (this->mSize > 0) {
				memcpy(this->mData, data, this->SizeInBytes());
			}
		}

		~HostMemoryManager() {
			this->Delete();
		}
	};

	template <class Type> class DeviceMemoryManager :
			public BaseMemoryManager<Type> {
	public:
		virtual void Allocate(size_t size) {
			if (size > 0 && cudaMalloc((void **) &(this->mData), size * sizeof(Type)) == cudaSuccess) {
				this->mSize = size;
			}
			else {
				this->Reset();
			}
		}

		virtual void Delete() {
			if (this->mSize > 0) 
				cudaFree(this->mData);
			this->Reset();
		}

		virtual void CopyFromDevice(Type * data, size_t size) {
			this->Resize(size);

			if (this->mSize > 0) {
				cudaMemcpy(this->mData, data, this->SizeInBytes(), cudaMemcpyDeviceToDevice);
			}
		}

		virtual void CopyFromHost(Type * data, size_t size) {
			this->Resize(size);
			if (this->mSize > 0) {
				cudaMemcpy(this->mData, data, this->SizeInBytes(), cudaMemcpyHostToDevice);
			}
		}

		~DeviceMemoryManager() {
			this->Delete();
		}
	};


}
