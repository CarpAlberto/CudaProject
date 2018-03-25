#include "NetworkLayer.h"
#include "MatrixFactory.h"
using namespace gpuNN;

void NetworkLayer::Push(Neuron* neuron) {
	this->m_neurons.push_back(neuron);
}

Neuron* NetworkLayer::operator[](size_t index) const{
	return this->m_neurons[index];
}

size_t NetworkLayer::Size() const {
	return this->m_neurons.size();
}

NetworkLayer::NetworkLayer(int numOutputs, TransferFunction* transfer) {

	//TODO may cause memory leaks
	for (int i = 0; i < numOutputs; i++) {
		this->m_neurons.push_back(new Neuron(numOutputs,transfer));
	}
}

void NetworkLayer::SetValue(size_t index, double value) {
	this->m_neurons[index]->SetOutputValue(value);
}

vDouble NetworkLayer::toVector() {
	vDouble returnVector;
	for (auto i = 0; i < this->m_neurons.size(); ++i) {
		auto activatedValue = this->m_neurons[i]->getActivatedValue();
		returnVector.push_back(activatedValue);
	}
	return returnVector;
}

PtrMatrix NetworkLayer::toMatrix() {
	auto m = MatrixFactory::getMatrix(1, this->m_neurons.size());
	for (size_t i = 0; i < this->m_neurons.size(); ++i) {
		float value = (float)(this->m_neurons[i]->getOutputValue());
		m->Set(0,i, value);
	}
	return m;
}

PtrMatrix NetworkLayer::toMatrixActivated() {
	auto m = MatrixFactory::getMatrix(1, this->m_neurons.size());
	for (size_t i = 0; i < this->m_neurons.size(); ++i) {
		auto value = (float)(this->m_neurons[i]->getActivatedValue());
		m->Set(0, i, value);
	}
	return m;
}

PtrMatrix NetworkLayer::toMatrixDerived() {
	auto m = MatrixFactory::getMatrix(1, this->m_neurons.size());
	for (auto i = 0; i < this->m_neurons.size(); ++i) {
		auto value = this->m_neurons[i]->getDerivedValue();
		m->Set(0, i,(float)value);
	}
	return m;
}

NetworkLayer::~NetworkLayer() {
	for (auto it : this->m_neurons) {
		delete it;
	}
}