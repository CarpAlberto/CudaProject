#pragma once
#include "vector.h"
#include <vector>
#include "TransferFunction.h"

namespace gpuNN {

	class NetworkLayer;

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

	} Connection;

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
			/// The index inside the layer
			/// </summary>
			size_t index;
			/// <summary>
			/// The transfer function
			/// </summary>
			TransferFunction* transferFunction;
		public:
			/// <summary>
			/// Initiate a new Neuron with the whole number of outputs
			/// </summary>
			/// <param name="numOutputs"></param>
			Neuron(double value);
			/// <summary>
			/// Destroy the neuron
			/// </summary>
			~Neuron();
			/// <summary>
			/// Sets the value of the value element
			/// </summary>
			/// <param name="mValue">The value to be setted</param>
			void SetOutputValue(double mValue);
			/// <summary>
			/// Feeds the data inside the NN based on the previous layer data
			/// </summary>
			/// <param name="previousLayer">The previous layer data</param>
			void FeedForward(const NetworkLayer& previousLayer);
			/// <summary>
			/// Returns the output value
			/// </summary>
			/// <returns>The output value</returns>
			double getOutputValue()const;
	};
}

