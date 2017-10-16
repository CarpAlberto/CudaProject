#pragma once
#include "NetworkLayer.h"
class NeuralNetwork
{
protected:
	std::vector<std::unique_ptr<NetworkLayer>> m_layers;
public:
	NeuralNetwork();
	~NeuralNetwork();
};

