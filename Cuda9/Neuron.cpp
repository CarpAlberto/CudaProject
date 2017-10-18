#include "Neuron.h"
using namespace gpuNN;


Neuron::Neuron( size_t numOutputs)
{
	/* Creates the number of connections */
	for (size_t i = 0; i < numOutputs; i++) {
		/*Add a new connection*/
		this->m_outputWeights.push_back(Connection());
		/*Associate a random weight*/
		this->m_outputWeights.back().weight = randomWeights();
	}
}


Neuron::~Neuron()
{
}
