#pragma once
#include "vector.h"
#include <vector>
namespace gpuNN {

	/// <summary>
	/// Typedefs the connection as an vector of two integers
	/// </summary>
	typedef struct
	{
		/// <summary>
		/// The internal weights
		/// </summary>
		double weight;
		/// <summary>
		/// The difference between the current neuron and the next one
		/// </summary>
		double deltaWeight;

	}Connection;

	class Neuron
	{
		private:
			/// <summary>
			/// Return a random weights
			/// </summary>
			/// <returns></returns>
			static double randomWeights();
			/// <summary>
			/// The output of the neuron
			/// </summary>
			double m_outputValue;
			/// <summary>
			/// The output wieghts and the diference between the weights (delta)
			/// </summary>
			std::vector<Connection> m_outputWeights;
		public:
			/// <summary>
			/// Initiate a new Neuron with the whole number of outputs
			/// </summary>
			/// <param name="numOutputs"></param>
			Neuron(size_t numOutputs);
			/// <summary>
			/// Destroy the neuron
			/// </summary>
			~Neuron();
	};
}

