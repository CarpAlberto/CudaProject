#pragma once
#include "memory.h"
#include "NetworkLayer.h"
#include "InputLayer.h"
#include "matrix.h"

namespace gpuNN {

	class NeuralNetwork
	{
		/// <summary>
		/// Typedef the vectore of integers for toplogy
		/// </summary>
		typedef std::vector<size_t> Topology;

	protected:
		/// <summary>
		/// Internals layers
		/// </summary>
		std::vector<std::unique_ptr<NetworkLayer>> m_layers;
		/// <summary>
		/// The weights of neurons
		/// </summary>
		std::vector<std::unique_ptr<GenericMatrix>> m_weights;
		/// <summary>
		/// The topology of the network
		/// </summary>
		Topology m_topology;

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
		void feedForward(const std::vector<double>& inputValue);
	};
}

