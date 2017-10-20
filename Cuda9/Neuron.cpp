#include "Neuron.h"
using namespace gpuNN;


Neuron::Neuron( double value)
{
	this->m_outputValue = value;
	this->index = index;
}
   

Neuron::~Neuron()
{
	m_outputWeights.clear();
}

double Neuron::randomWeights() {
	return rand() % RAND_MAX;
}

void Neuron::SetOutputValue(double mValue) {
	this->m_outputValue = mValue;
}

double Neuron::getOutputValue()const {
	return this->m_outputValue;
}