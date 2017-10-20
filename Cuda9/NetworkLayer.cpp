#include "NetworkLayer.h"
using namespace gpuNN;

void NetworkLayer::Push(Neuron&& neuron) {
	this->m_neurons.push_back(std::make_shared<Neuron>(neuron));
}

Neuron* NetworkLayer::operator[](int index) const{
	return this->m_neurons[index].get();
}

size_t NetworkLayer::Size() const {
	return this->m_neurons.size();
}

NetworkLayer::NetworkLayer(int numOutputs) {
	for (int i = 0; i < numOutputs; i++) {
		this->m_neurons.push_back(std::make_shared<Neuron>(0.0));
	}
}

void NetworkLayer::SetValue(int index, double value) {
	this->m_neurons[index].get()->SetOutputValue(value);
}