#include "NeuralNetwork.h"
using namespace gpuNN;


NeuralNetwork::NeuralNetwork(Topology& topology, TransferFunctionType transferFunction)
{
	this->m_topology = topology;
	auto size = topology.size();
	for (auto i = 0; i < size; i++) {
		/*Add a new layer*/
		auto layer = std::make_shared<NetworkLayer>(topology[i]);
		this->m_layers.push_back(layer);
	}
	for (auto i = 0; i < topology.size() - 1; i++) {
		auto matrix = new CpuMatrix(topology[i], topology[i + 1], 1);
		matrix->SetRandom();
		this->m_weights.push_back(matrix);
	}
}

NeuralNetwork::~NeuralNetwork()
{

}

PtrMatrix NeuralNetwork::getNeuronAsMatrix(size_t index) const {
	return this->m_layers[index].get()->toMatrix();
}

PtrMatrix NeuralNetwork::getNeuronActivatedValueAsMatrix(size_t index) const {
	return this->m_layers[index].get()->toMatrixActivated();
}

PtrMatrix NeuralNetwork::getNeuronDerivedValueAsMatrix(size_t index) const {
	return this->m_layers[index].get()->toMatrixDerived();
}

PtrMatrix NeuralNetwork::getWeightsMatrix(size_t index) const {
	return this->m_weights[index];
}

void NeuralNetwork::setNeuronValue(size_t indexLayer, size_t indexNeuron, double value) {
	this->m_layers[indexLayer]->SetValue(indexNeuron, value);
}

void NeuralNetwork::feedForward() {
	
	
	for (auto i = 0; i < this->m_layers.size() - 1; ++i) {
		auto neuronMatrix = this->getNeuronAsMatrix(i);
		auto weightsMatrix = this->getWeightsMatrix(i);

		if (i != 0) {
			neuronMatrix = this->getNeuronActivatedValueAsMatrix(i);
		}
		//Perform the multiplication
		(*neuronMatrix).Print();
		(*weightsMatrix).Print();
	  CpuMatrix multipliedMatrix = (*neuronMatrix) * (*weightsMatrix);
	  multipliedMatrix.Print();
	  for (auto index = 0; index < multipliedMatrix.getCols(); index++) 
	  {
			this->setNeuronValue(i + 1, index, multipliedMatrix.Get(0, index, 0));
	  }
	}
			
}

void NeuralNetwork::SetCurrentInput(const vDouble& input) {
	this->m_input = input;

	for (auto i = 0; i < input.size(); i++) {
		this->m_layers[0]->SetValue(i, input[i]);
	}
}

void NeuralNetwork::Print() {
	for (int i = 0; i < this->m_layers.size(); i++) {
		std::cout << "Layer:" << i << std::endl;
		if (i == 0) {
			auto m = this->m_layers[i].get()->toMatrix();
			m->Print();
		}
		else {
			auto m = this->m_layers[i].get()->toMatrixActivated();
			m->Print();
		}
		if (i < this->m_layers.size() - 1) {
			std::cout << "Weight Matrix" << i << std::endl;
			this->getWeightsMatrix(i)->Print();
		}
		std::cout << "================" << std::endl;
	} 
}