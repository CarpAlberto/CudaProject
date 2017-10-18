#include "NeuralNetwork.h"
using namespace gpuNN;


NeuralNetwork::NeuralNetwork(Topology& topology)
{
	auto size = topology.size();

	/*Add the input layer*/

	for (auto i = 0; i < size; i++) {
		this->m_layers.push_back(std::make_unique<NetworkLayer>());
		auto numOutputs = i == topology.size() - 1 ? 0 : topology[i + 1];
		for (auto neuronNumber = 0; neuronNumber <= topology[i]; ++neuronNumber) {
			this->m_layers.back()->Push(Neuron(numOutputs));
			ApplicationContext::instance()->getLog().get()->print<gpuNN::SeverityType::DEBUG>
				("Neuron has been created");
		}
	}
}


NeuralNetwork::~NeuralNetwork()
{

}
