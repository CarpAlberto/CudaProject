#pragma once
#include "includes.h"
#include "Neuron.h"
#include "matrix.h"

namespace gpuNN {

	typedef GenericMatrix*			 PtrMatrix;
	typedef std::vector<PtrMatrix>   VectorPtrMatrix;

	class NetworkLayer {

		typedef std::vector<Neuron*> InternalLayer;
	public:
		/// <summary>
		/// Creates a new layer
		/// </summary>
		/// <param name="numOutputs">The number of outputs</param>
		NetworkLayer(int numOutputs);
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
		Neuron* operator[](int index) const;
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
		void SetValue(int index, double value);
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

	protected:
		std::string                    m_layer_name;
		InternalLayer				   m_neurons;

	};
}