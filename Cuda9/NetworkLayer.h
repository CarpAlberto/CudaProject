#pragma once
#include "includes.h"
#include "Neuron.h"
#include "MatrixFactory.h"
#include "Memory.h"

namespace gpuNN {

	typedef GenericMatrix*			 PtrMatrix;
	typedef std::vector<PtrMatrix>   VectorPtrMatrix;


	class cudaObject {
		
	protected:
		cudafloat* data;
		cudafloat* activatedData;
		cudafloat* derivedData;
		size_t	   data_size;
	public:
		cudaObject(cudafloat* data,size_t allSize);
		void   Print(UIInterface* rhs);
		float* HostData();
		float* HostDataActivated();
		float* HostDataDerived();
		float* Data();
		float* Activated();
		float* Derived();
		void   SetData(float* data, size_t dataSize);
		size_t Size();
		void SetSize(size_t size);
		void Activate();
		void Derive();
	};

	class NetworkLayer {

		typedef std::vector<Neuron*> InternalLayer;
	
	public:
		NetworkLayer() = default;
		/// <summary>
		/// Creates a new layer
		/// </summary>
		/// <param name="numOutputs">The number of outputs</param>
		NetworkLayer(int numOutputs,TransferFunction* transfer);
		/// <summary>
		/// Add a new neuron in the layer
		/// </summary>
		/// <param name="neuron"></param>
		void Push(Neuron* neuron);
		/// <summary>
		/// Returns the @code index entry from the vector
		/// </summary>
		/// <param name="index">The index parameter</param>
		/// <returns></returns>
		Neuron* operator[](size_t index) const;
		/// <summary>
		/// The destructor of the layer
		/// </summary>
		virtual ~NetworkLayer();
		/// <summary>
		/// Returns the number of the neuron inside the layer
		/// </summary>
		/// <returns></returns>
		size_t Size() const;
		/// <summary>
		/// Sets the Neuron index value with the value provided by the parameter <code>value</code>
		/// </summary>
		/// <param name="index">The index of the neuron</param>
		/// <param name="value">The value to be setted</param>
		void SetValue(size_t index, double value);
		/// <summary>
		/// Returns a vector based on a neuron vector
		/// </summary>
		/// <returns>The vector</returns>
		vDouble toVector();
		/// <summary>
		/// Returns the matrix based on the neurons
		/// </summary>
		/// <returns></returns>
		PtrMatrix toMatrix();
		/// <summary>
		/// Returns a matrix from the activated value
		/// </summary>
		/// <returns></returns>
		PtrMatrix toMatrixActivated();
		/// <summary>
		/// Returns a matrix from derivde value
		/// </summary>
		/// <returns></returns>
		PtrMatrix toMatrixDerived();

		virtual cudaObject* getData() { throw new std::exception(""); };

	protected:
		std::string                    m_layer_name;
		InternalLayer				   m_neurons;

	};

	class OptimizedGpuNetworkLayer : NetworkLayer {

	protected:
		cudaObject* obj;
		TransferFunction* transfer;
	public:
		OptimizedGpuNetworkLayer(int numOutputs, TransferFunction* transfer);
		virtual cudaObject* getData();
	};
}