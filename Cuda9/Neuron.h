#pragma once
#include "vector.h"
#include <vector>
#include "TransferFunction.h"

namespace gpuNN {

	class NetworkLayer;

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
			/// The activated value
			/// </summary>
			double m_activatedValue;
			/// <summary>
			/// The derived value
			/// </summary>
			double m_derivedValue;
			/// <summary>
			/// The index inside the layer
			/// </summary>
			/// <summary>
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
			/// Returns the output value
			/// </summary>
			/// <returns>The output value</returns>
			double getOutputValue()const;
			/// <summary>
			/// Activate the current value
			/// </summary>
			void Activate();
			/// <summary>
			/// Derive the value
			/// </summary>
			void Derive();
			/// <summary>
			/// Returns the activated value
			/// </summary>
			/// <returns>The activated value</returns>
			double getActivatedValue() const;
			/// <summary>
			/// Returns the derived value
			/// </summary>
			/// <returns>The derived value</returns>
			double getDerivedValue() const;


	};
}

