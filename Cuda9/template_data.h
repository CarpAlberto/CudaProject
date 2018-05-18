#pragma once
#include "cuda_macro.h"
#include "Memory.h"
#include <cublas.h>


/*
Noel Lopes is a Professor at the Polytechnic of Guarda, Portugal
and a Researcher at the CISUC - University of Coimbra, Portugal
Copyright (C) 2009-2015 Noel de Jesus Mendonça Lopes

This file is part of GPUMLib.

GPUMLib is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

namespace gpuNN
{
	template <class Type> class CudaMatrix;
	
	template <class Type> class DeviceMatrix;
	
	template <class Type> class GpuArray;
	
	template <class Type> class CpuArray;
	
	template <class Type> class CudaArray;

	template <class Type> class Array 
	{
		friend class CudaArray<Type>;
	private:
		BaseMemoryManager<Type> * memory;

	protected:
		Array(BaseMemoryManager<Type> & memory) {
			this->memory = &memory;
		}

	public:
		void Free() {
			memory->Delete();
		}

		size_t Size() const {
			return memory->Size();
		}

		Type * Data() const {
			return memory->Data();
		}

		size_t Resize(size_t size) {
			return memory->Resize(size);
		}
	};

	template <class Type> 
	class CpuArray :public Array<Type> {
	private:
		HostMemoryManager<Type> hostMem;

	public:
		CpuArray() : Array<Type>(hostMem) {}

		explicit CpuArray(size_t size) : Array<Type>(hostMem) {
			hostMem.Allocate(size);
		}

		CpuArray(const GpuArray<Type> & originalArray) : Array<Type>(hostMem) {
			hostMem.CopyFromDevice(originalArray.Data(), originalArray.Size());
		}

		CpuArray(const CpuArray<Type> & originalArray) :Array<Type>(hostMem) {
			hostMem.CopyFromHost(originalArray.Data(), originalArray.Size());
		}

		CpuArray<Type> & operator = (const GpuArray<Type> & originalArray) {
			hostMem.CopyFromDevice(originalArray.Data(), originalArray.Size());
			return *this;
		}
		CpuArray<Type> & operator = (const CpuArray<Type> & originalArray) {
			hostMem.CopyFromHost(originalArray.Data(), originalArray.Size());
			return *this;
		}
		void TransferOwnerShipFrom(CpuArray<Type> & other) {
			if (this != &other)
				hostMem.TransferOwnerShipFrom(other.hostmem);
		}

		CpuArray(CpuArray<Type> && temporaryArray) : Array<Type>(hostMem) {
			hostMem.TransferOwnerShipFrom(temporaryArray.hostMem);
		}
		CpuArray<Type> & operator = (CpuArray<Type> && temporaryArray) {
			hostMem.TransferOwnerShipFrom(temporaryArray.hostMem);
			return *this;
		}
		Type & operator [] (size_t element) {
			assert(element < this->Size());
			return this->Data()[element];
		}
		Type operator [] (size_t element) const {
			assert(element < this->Size());
			return this->Data()[element];
		}
	};

	template <class Type> class CudaArray {
	
	private:
		CpuArray<Type> h;
		GpuArray<Type> d;
		void TransferOwnerShipFrom(Array<Type> & other) {
			if (this != other) {
				h.TransferOwnerShipFrom(other.h);
				d.TransferOwnerShipFrom(other.d);
			}
		}

	public:
		CudaArray() {}

		explicit CudaArray(size_t size) : h(size), d(size) {}

		CudaArray(const GpuArray<Type> & originalArray) : h(originalArray), d(originalArray) {}

		CudaArray(const CpuArray<Type> & originalArray) : h(originalArray), d(originalArray) {}

		CudaArray(const CudaArray<Type> & other) : h(other.h), d(other.d) {}

		CudaArray<Type> & operator = (const CudaArray<Type> & originalArray) {
			h = originalArray.h;
			d = originalArray.d;

			return *this;
		}

		CudaArray<Type> & operator = (const GpuArray<Type> & originalArray) {
			h = d = originalArray;

			return *this;
		}

		CudaArray<Type> & operator = (const CpuArray<Type> & originalArray) {
			d = h = originalArray;

			return *this;
		}
		
		CudaArray(CudaArray<Type> && temporaryArray) {
			TransferOwnerShipFrom(temporaryArray);
		}

		CudaArray<Type> & operator = (const CudaArray<Type> && temporaryArray) {
			TransferOwnerShipFrom(temporaryArray);
			return *this;
		}

		Type & operator [] (size_t element) {
			return h[element];
		}

		Type operator [] (size_t element) const {
			return h[element];
		}

		size_t Length() const {
			return d.Length();
		}

		Type * HostData() const {
			return h.Data();
		}

		Type * DeviceData() const {
			return d.Data();
		}

		size_t ResizeWithoutPreservingData(size_t size) {
			size_t he = h.ResizeWithoutPreservingData(size);
			size_t de = d.ResizeWithoutPreservingData(size);

			return (he > de) ? de : he;
		}

		void UpdateDevice() {
			d = h;
		}

		void UpdateHost() {
			h = d;
		}

		GpuArray<Type> & GetGpuArray() {
			return d;
		}

		CpuArray<Type> & GetCpuArray() {
			return h;
		}

		//! Disposes the array
		void Dispose() {
			d.Dispose();
			h.Dispose();
		}

		void DisposeDevice() {
			UpdateHost();
			d.Dispose();
		}

		void DisposeHost() {
			UpdateDevice();
			h.Dispose();
		}
	};

	template <class Type> 
	class GpuArray : public Array < Type > {
	private:
		DeviceMemoryManager<Type> deviceMem;

	public:
		GpuArray() : Array<Type>(deviceMem) {}

		explicit GpuArray(size_t size) : Array<Type>(deviceMem) {
			deviceMem.Allocate(size);
		}

		GpuArray(const CpuArray<Type> & originalArray) : Array<Type>(deviceMem) {
			deviceMem.CopyFromHost(originalArray.Data(), originalArray.Length());
		}

		GpuArray(const GpuArray<Type> & originalArray) : Array<Type>(deviceMem) {
			deviceMem.CopyFromDevice(originalArray.Data(), originalArray.Size());
		}

		GpuArray(const Type * originalArray, size_t size) : Array<Type>(deviceMem) {
			deviceMem.CopyFromHost(originalArray, size);
		}

		GpuArray<Type> & operator = (const GpuArray<Type> & originalArray) {
			deviceMem.CopyFromDevice(originalArray.Data(), originalArray.Size());
			return *this;
		}

		GpuArray<Type> & operator = (const CpuArray<Type> & originalArray) {
			deviceMem.CopyFromHost(originalArray.Data(), originalArray.Size());
			return *this;
		}

		void TransferOwnerShipFrom(GpuArray<Type> & other) {
			if (this != &other) deviceMem.TransferOwnerShipFrom(other.deviceMem);
		}

		GpuArray(GpuArray<Type> && temporaryArray) : Array<Type>(deviceMem) {
			deviceMem.TransferOwnerShipFrom(temporaryArray.deviceMem);
		}

		GpuArray<Type> & operator = (GpuArray<Type> && temporaryArray) {
			deviceMem.TransferOwnerShipFrom(temporaryArray.deviceMem);
			return *this;
		}
	};


	template <class Type> class BaseMatrix {
		
		friend class CudaMatrix<Type>;
	protected:
		StoringOrder m_storingOrder;
		size_t m_rows;
		size_t m_columns;

	private:
		BaseMemoryManager<Type> * bMemory;

		void CompleteAssign(const BaseMatrix<Type> & other) {
			
			if (bMemory->Size() == other.Elements()) {
				this->m_rows = other.m_rows;
				this->m_columns = other.m_columns;
				this->m_storingOrder = other.m_storingOrder;
			}
		}

	public:

		size_t ResizeWithoutPreservingData(size_t rows, size_t columns) {
			size_t newElements = rows * columns;

			if (bMemory->Resize(newElements) == newElements) {
				this->m_rows = rows;
				this->m_columns = columns;
			}
			else {
				this->m_rows = 0;
				this->m_columns = 0;
			}

			return Elements();
		}

	protected:
		BaseMatrix(BaseMemoryManager<Type> & mem, StoringOrder storingOrder = RowMajor) {
			this->bMemory = &mem;
			this->m_storingOrder = storingOrder;
			this->m_rows = 0;
			this->m_columns = 0;
		}

		void AssignHostMatrix(const BaseMatrix<Type> & other) {
			bMemory->CopyFromHost(other.Data(), other.Elements());
			CompleteAssign(other);
		}

		void AssignDeviceMatrix(const BaseMatrix<Type> & other) {
			bMemory->CopyFromDevice(other.Data(), other.Elements());
			CompleteAssign(other);
		}

	public:
		void Dispose() {
			bMemory->Dispose();
			m_rows = m_columns = 0;
		}

		size_t Rows() const {
			return m_rows;
		}

		size_t Columns() const {
			return m_columns;
		}

		Type * Data() const {
			return bMemory->Data();
		}

		size_t Elements() const {
			return bMemory->Size();
		}
		bool IsRowMajor() const {
			return (m_storingOrder == RowMajor);
		}

		void ReplaceByTranspose() {
			size_t newRows = m_columns;
			m_columns = m_rows;
			m_rows = newRows;
			m_storingOrder = (IsRowMajor()) ? ColumnMajor : RowMajor;
		}

		void TransferOwnerShipFrom(BaseMatrix<Type> & other) {
			if (this != &other) {
				bMemory->TransferOwnerShipFrom(*(other.bMemory));

				m_storingOrder = other.m_storingOrder;
				m_rows = other.m_rows;
				m_columns = other.m_columns;

				other.m_rows = 0;
				other.m_columns = 0;
			}
		}
	};

	template <class Type> 
	class HostMatrix : public BaseMatrix < Type > {
	private:
		HostMemoryManager<Type> hostMem;

		size_t Index(size_t row, size_t column) const {
			
			assert(row < this->m_rows && column < this->m_columns);

			return (this->IsRowMajor()) ? row * this->m_columns + column : column * this->m_rows + row;
		}

	public:
		HostMatrix(StoringOrder storingOrder = RowMajor) : BaseMatrix<Type>(hostMem, storingOrder) {}

		HostMatrix(size_t rows, size_t columns, StoringOrder storingOrder = RowMajor) : BaseMatrix<Type>(hostMem, storingOrder) {
			this->ResizeWithoutPreservingData(rows, columns);
		}

		HostMatrix(const HostMatrix<Type> & other) : BaseMatrix<Type>(hostMem) {
			this->AssignHostMatrix(other);
		}

		HostMatrix(const DeviceMatrix<Type> & other) : BaseMatrix<Type>(hostMem) {
			this->AssignDeviceMatrix(other);
		}

		HostMatrix<Type> & operator = (const HostMatrix<Type> & other) {
			this->AssignHostMatrix(other);
			return *this;
		}
		
		HostMatrix<Type> & operator = (const DeviceMatrix<Type> & other) {
			this->AssignDeviceMatrix(other);
			return *this;
		}

		HostMatrix(HostMatrix<Type> && temporaryMatrix) : BaseMatrix<Type>(hostMem) {
			this->TransferOwnerShipFrom(temporaryMatrix);
		}

		HostMatrix<Type> & operator = (HostMatrix<Type> && temporaryMatrix) {
			this->TransferOwnerShipFrom(temporaryMatrix);
			return *this;
		}

		HostMatrix<Type> Transpose() {
			HostMatrix<Type> transpose(*this);
			transpose.ReplaceByTranspose();

			return transpose;
		}

		Type & operator()(size_t row, size_t column) {
			return this->Data()[Index(row, column)];
		}

		Type operator()(size_t row, size_t column) const {
			return this->Data()[Index(row, column)];
		}

		void Print()
		{
			
		}
	};

	template <class Type> class DeviceMatrix : public BaseMatrix < Type > {
	private:
		DeviceMemoryManager<Type> deviceMem;

	public:

		DeviceMatrix(StoringOrder storingOrder = RowMajor) : BaseMatrix<Type>(deviceMem, storingOrder) {}

		DeviceMatrix(size_t rows, size_t columns, StoringOrder storingOrder = RowMajor) : BaseMatrix<Type>(deviceMem, storingOrder) {
			this->ResizeWithoutPreservingData(rows, columns);
		}

		DeviceMatrix(const DeviceMatrix<Type> & other) : BaseMatrix<Type>(deviceMem) {
			this->AssignDeviceMatrix(other);
		}

		DeviceMatrix(const HostMatrix<Type> & other) : BaseMatrix<Type>(deviceMem) {
			this->AssignHostMatrix(other);
		}

		DeviceMatrix<Type> & operator = (const HostMatrix<Type> & other) {
			this->AssignHostMatrix(other);
			return *this;
		}

		DeviceMatrix<Type> & operator = (const DeviceMatrix<Type> & other) {
			this->AssignDeviceMatrix(other);
			return *this;
		}
		DeviceMatrix(DeviceMatrix<Type> && temporaryMatrix) : BaseMatrix<Type>(deviceMem) {
			this->TransferOwnerShipFrom(temporaryMatrix);
		}

		DeviceMatrix<Type> & operator = (DeviceMatrix<Type> && temporaryMatrix) {
			this->TransferOwnerShipFrom(temporaryMatrix);
			return *this;
		}

		DeviceMatrix<Type> Transpose() {
			HostMatrix<Type> transpose(*this);
			transpose.ReplaceByTranspose();

			return transpose;
		}
	};

	template <> class DeviceMatrix<cudafloat> : public BaseMatrix < cudafloat > {
	private:
		DeviceMemoryManager<cudafloat> deviceMem;

	public:
	
		DeviceMatrix(StoringOrder storingOrder = RowMajor) : BaseMatrix<cudafloat>(deviceMem, storingOrder) {}


		DeviceMatrix(size_t rows, size_t columns, StoringOrder storingOrder = RowMajor) : BaseMatrix<cudafloat>(deviceMem, storingOrder) {
			this->ResizeWithoutPreservingData(rows, columns);
		}
		DeviceMatrix(const DeviceMatrix<cudafloat> & other) : BaseMatrix<cudafloat>(deviceMem) {
			this->AssignDeviceMatrix(other);
		}
		DeviceMatrix(const HostMatrix<cudafloat> & other) : BaseMatrix<cudafloat>(deviceMem) {
			this->AssignHostMatrix(other);
		}

		DeviceMatrix<cudafloat> & operator = (const HostMatrix<cudafloat> & other) {
			this->AssignHostMatrix(other);
			return *this;
		}
		DeviceMatrix<cudafloat> & operator = (const DeviceMatrix<cudafloat> & other) {
			this->AssignDeviceMatrix(other);
			return *this;
		}

		DeviceMatrix(DeviceMatrix<cudafloat> && temporaryMatrix) : BaseMatrix<cudafloat>(deviceMem) {
			this->TransferOwnerShipFrom(temporaryMatrix);
		}

		DeviceMatrix<cudafloat> & operator = (DeviceMatrix<cudafloat> && temporaryMatrix) {
			this->TransferOwnerShipFrom(temporaryMatrix);
			return *this;
		}

		DeviceMatrix<cudafloat> Transpose() {
			HostMatrix<cudafloat> transpose(*this);
			transpose.ReplaceByTranspose();

			return transpose;
		}

		void MultiplyBySelfTranspose(DeviceMatrix<cudafloat> & C, cudafloat alpha = 1, cudafloat beta = 0) {
			assert(C.m_rows <= INT_MAX && C.m_rows == m_rows && C.m_columns == m_rows);

			if (C.IsRowMajor()) {
				assert(beta == 0);
				C.m_storingOrder = ColumnMajor;
			}

			size_t ldAB = IsRowMajor() ? this->m_columns : this->m_rows;
			
			cublasSgemm(this->IsRowMajor() ? 'T' : 'N', this->IsRowMajor() ? 'N' : 'T', 
				(int)C.m_rows, (int)C.m_columns, (int)m_columns, alpha, 
				this->Data(), (int)ldAB, this->Data(), (int)ldAB, beta, C.Data(), (int)C.m_rows);
		}

		static void Multiply(DeviceMatrix<cudafloat> & A, DeviceMatrix<cudafloat> & B, 
			DeviceMatrix<cudafloat> & C, cudafloat alpha = 1, cudafloat beta = 0) {
			
			assert(A.m_columns <= INT_MAX && C.m_rows <= INT_MAX && C.m_columns <= INT_MAX);
			assert(A.m_columns == B.m_rows && C.m_rows == A.m_rows && C.m_columns == B.m_columns);

			if (C.IsRowMajor()) {
				assert(beta == 0);
				C.m_storingOrder = ColumnMajor;
			}

			cublasSgemm(A.IsRowMajor() ? 'T' : 'N', B.IsRowMajor() ? 'T' : 'N', 
				(int)C.m_rows, (int)C.m_columns, (int)A.m_columns, alpha, A.Data(),
				(int)(A.IsRowMajor() ? A.m_columns : A.m_rows), B.Data(),
				(int)(B.IsRowMajor() ? B.m_columns : B.m_rows), beta, C.Data(), (int)C.m_rows);
		}
	};

	template <class Type> class DeviceAccessibleVariable {
	private:
		Type * value;
	public:

		DeviceAccessibleVariable() {
			cudaMallocHost((void**)&value, sizeof(Type));
		}
		DeviceAccessibleVariable(const Type initialValue) {
			cudaMallocHost((void**)&value, sizeof(Type));
			*value = initialValue;
		}
		~DeviceAccessibleVariable() {
			cudaFreeHost(value);
		}
		Type & Value() {
			return *value;
		}
		Type * Data() {
			return value;
		}
		void UpdateValue(Type * deviceValue) {
			cudaMemcpy(value, deviceValue, sizeof(Type), cudaMemcpyDeviceToHost);
		}

		void UpdateValueAsync(Type * deviceValue, cudaStream_t stream) {
			cudaMemcpyAsync(value, deviceValue, sizeof(Type), cudaMemcpyDeviceToHost, stream);
		}
	};

	class CudaStream {
	private:
		cudaStream_t stream;

	public:
		CudaStream() {
			cudaStreamCreate(&stream);
		}
		~CudaStream() {
			cudaStreamDestroy(stream);
		}
		operator cudaStream_t () {
			return stream;
		}
	};

	using RealGpuArray = GpuArray<cudaReal>;
	using RealCpuArray = CpuArray<cudaReal>;
	using IntCpuArray = CpuArray<int>;
	using RealDeviceMatrix = DeviceMatrix<cudaReal>;
	using RealHostMatrix = HostMatrix<cudaReal>;
	using TopologyOptimized = CpuArray<int>;
}