#pragma once
#include "memory.h"
#include "NetworkLayer.h"
#include "InputLayer.h"
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
public:
	/// <summary>
	/// Init a new neural network with a specific topology
	/// </summary>
	/// <param name=""></param>
	NeuralNetwork(Topology&);
	/// <summary>
	/// Destroy the Neural network
	/// </summary>
	~NeuralNetwork();
};

