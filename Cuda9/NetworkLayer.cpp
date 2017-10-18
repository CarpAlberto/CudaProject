#include "NetworkLayer.h"
using namespace gpuNN;

void NetworkLayer::Push(Neuron&& neuron) {
	this->m_neurons.push_back(std::make_unique<Neuron>(neuron));
}