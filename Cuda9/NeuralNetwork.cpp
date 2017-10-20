#include "NeuralNetwork.h"
using namespace gpuNN;


NeuralNetwork::NeuralNetwork(Topology& topology, TransferFunctionType transferFunction)
{
	this->m_topology = topology;
	auto size = topology.size();
	for (auto i = 0; i < size; i++) {
		/*Add a new layer*/
		this->m_layers.push_back(std::make_unique<NetworkLayer>());
	}
	for (auto i = 0; i < topology.size() - 1; i++) {
		this->m_weights.push_back(std::make_unique<GenericMatrix>
			(new CpuMatrix(topology[i], topology[i + 1], 1)));
	}
}

NeuralNetwork::~NeuralNetwork()
{

}

void Neuron::FeedForward(const NetworkLayer& previousLayer) {
	
	double sum = 0.0;

	for (size_t n = 0; n < previousLayer.Size(); ++n) {
		sum += previousLayer[n]->getOutputValue() * previousLayer[n]->m_outputWeights[0].weight;
	}
	this->m_outputValue = this->transferFunction->getValue(sum);
}

void NeuralNetwork::feedForward(const std::vector<double>& inputValue) {
	for (auto i = 0; i < inputValue.size(); i++) {
		(*this->m_layers[0])[i]->SetOutputValue(inputValue[i]);
	}
	//Forward propagation
	for (auto i = 1; i < this->m_layers.size(); ++i) {
		for (auto n = 0; n < (m_layers[i]).get()->Size(); ++n) {
			(*this->m_layers[i])[n]->
		}
	}
 }