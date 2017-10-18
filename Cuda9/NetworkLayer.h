#pragma once
#include "includes.h"
#include "Neuron.h"

namespace gpuNN {

	class NetworkLayer {
	
		typedef std::vector<std::unique_ptr<Neuron>> InternalLayer;
	public:
		void Push(Neuron&& neuron);
		virtual ~NetworkLayer();
	protected:
		std::string                    m_layer_name;
		InternalLayer				   m_neurons;

	};
}