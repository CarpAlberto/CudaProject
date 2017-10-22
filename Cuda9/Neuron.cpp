#include "Neuron.h"
using namespace gpuNN;


Neuron::Neuron( double value)
{
	this->m_outputValue = value;
	this->transferFunction = TransferFunctionFactory::instance()->
		getTransferFunction(TransferFunctionType::TANH);
	Activate();
	Derive();
}
   
Neuron::~Neuron()
{
	
}

double Neuron::randomWeights() {
	return rand() % RAND_MAX;
}

void Neuron::SetOutputValue(double mValue) {
	this->m_outputValue = mValue;
	Activate();
	Derive();
}

double Neuron::getOutputValue()const {
	return this->m_outputValue;
}

void Neuron::Activate() {
	this->m_activatedValue = this->transferFunction->getValue(this->m_outputValue);
}

void Neuron::Derive() {
	this->m_derivedValue = this->transferFunction->getDerivative(this->m_outputValue);
}

double Neuron::getActivatedValue() const {
	return this->m_activatedValue;
}

double Neuron::getDerivedValue() const {
	return this->m_derivedValue;
}