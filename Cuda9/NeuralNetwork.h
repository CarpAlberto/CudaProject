#pragma once
#include "memory.h"
#include "NetworkLayer.h"
#include "InputLayer.h"
#include "matrix.h"
#include "includes.h"

namespace gpuNN {

	class NeuralNetwork
	{
	
	protected:
		/// <summary>
		/// Internals layers
		/// </summary>
		std::vector<std::shared_ptr<NetworkLayer>> m_layers;
		/// <summary>
		/// The weights of neurons
		/// </summary>
		VectorPtrMatrix m_weights;
		/// <summary>
		/// The topology of the network
		/// </summary>
		Topology m_topology;
		/// <summary>
		/// The input of the Neural Network
		/// </summary>
		vDouble m_input;
		/// <summary>
		/// The learning rate
		/// </summary>
		double learningRate;
		/// <summary>
		/// The Bias of the NN
		/// </summary>
		double bias;

	public:
		/// <summary>
		/// Init a new neural network with a specific topology
		/// </summary>
		/// <param name="rhs">The righ hand side topology</param>
		NeuralNetwork(Topology& rhs,TransferFunctionType type = TransferFunctionType::TANH);
		/// <summary>
		/// Destroy the Neural network
		/// </summary>
		~NeuralNetwork();
		/// <summary>
		/// Forward a vector of inputs inside the Neural Network
		/// </summary>
		/// <param name="inputValue">The input vector</param>
		void feedForward();
		/// <summary>
		/// Returns a matrix from the output value stored in neurons
		/// <param name="index">The index of the layers</param>
		/// </summary>
		/// <returns></returns>
		PtrMatrix getNeuronAsMatrix(size_t index) const;
		/// <summary>
		/// Returns a matrix from derived value stored in neurons
		/// <param name="index">The index of the layers</param>
		/// </summary>
		/// <returns></returns>
		PtrMatrix getNeuronActivatedValueAsMatrix(size_t index) const;
		/// <summary>
		/// Returns a matrix from derived value stored in neurons
		/// </summary>
		/// <returns></returns>
		PtrMatrix getNeuronDerivedValueAsMatrix (size_t index) const;
		/// <summary>
		/// Returns the weights matrix
		/// </summary>
		/// <param name=""></param>
		/// <returns></returns>
		PtrMatrix getWeightsMatrix(size_t) const;
		/// <summary>
		/// Sets the neuron value
		/// </summary>
		/// <param name="indexLayer">The index of the layer inside the Neural Network</param>
		/// <param name="indexNeuron">The index of neuron inside layer</param>
		/// <param name="value">The value to be setted</param>
		void setNeuronValue(size_t indexLayer, size_t indexNeuron, double value);
		/// <summary>
		/// Set's the current input of the Neural Network
		/// </summary>
		/// <param name="input">The input value</param>
		void SetCurrentInput(const vDouble& input);
	};
}

