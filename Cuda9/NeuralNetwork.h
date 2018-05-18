#pragma once
#include "includes.h"
#include "memory.h"
#include "NetworkLayer.h"
#include "INeuralNetwork.h"


namespace gpuNN {

	class NeuralNetwork : public IPrintableObject,public INeuralNetwork
	{
	
	protected:
		/// <summary>
		/// Internals layers
		/// </summary> 
		#ifndef OPTIMIZED_GPU
			std::vector<std::shared_ptr<NetworkLayer>> m_layers;
		#else
			std::vector<std::shared_ptr<OptimizedGpuNetworkLayer>> m_layers;
		#endif
		/// <summary>
		/// The weights of neurons
		/// </summary>
		VectorPtrMatrix m_weights;
		/// <summary>
		/// The matrix of gradients
		/// </summary>
		VectorPtrMatrix m_gradients;
		/// <summary>
		/// The topology of the network
		/// </summary>
		Topology m_topology;
		/// <summary>
		/// The input of the Neural Network
		/// </summary>
		vDouble m_input;
		/// <summary>
		/// The input of the Neural Network
		/// </summary>
		vDouble m_target;
		/// <summary>
		/// The learning rate
		/// </summary>
		double m_learningRate;
		/// <summary>
		/// The Bias of the NN
		/// </summary>
		double m_bias;
		/// <summary>
		/// The momentum of the NN
		/// </summary>
		double m_momentum;
		/// <summary>
		/// The error of the NN
		/// </summary>
		double m_error;
		/// <summary>
		/// All the errors
		/// </summary>
		vDouble m_errors;
		/// <summary>
		/// The historical errors
		/// </summary>
		vDouble m_historicalErrors;
		/// <summary>
		/// The transfer function used in NN
		/// </summary>
		TransferFunction* m_TransferFunction;
		/// <summary>
		/// The error function
		/// </summary>
		ErrorFunction* m_ErrorFunction;

	public:
		/// <summary>
		/// Init a new neural network with a specific topology
		/// </summary>
		/// <param name="rhs">The righ hand side topology</param>
		NeuralNetwork(Topology& rhs,
			double bias,
			double learningRate,
			double momentum,
			TransferFunctionType type   = TransferFunctionType::TANH,
			FunctionErrorType errorType = FunctionErrorType::MSE);
		/// <summary>
		/// Destroy the Neural network
		/// </summary>
		~NeuralNetwork();
		/// <summary>
		/// Forward a vector of inputs inside the Neural Network
		/// </summary>
		/// <param name="inputValue">The input vector</param>
		void FeedForward();
		/// <summary>
		/// Returns a matrix from the output value stored in neurons
		/// <param name="index">The index of the layers</param>
		/// </summary>
		/// <returns></returns>
		PtrMatrix getNeuronAsMatrix(size_t index) const;
		/// <summary>
		/// Returns neurons as anm array of data
		/// </summary>
		/// <returns></returns>
		cudaObject* getNeuronAsData(size_t index)const;
		/// <summary>
		/// Returns a matrix from derived value stored in neurons
		/// <param name="index">The index of the layers</param>
		/// </summary>
		/// <returns></returns>
		PtrMatrix getNeuronActivatedValueAsMatrix(size_t index) const;
		/// <summary>
		/// Returns a matrix from derived value stored in neurons
		/// </summary>
		/// <returns></returns>
		PtrMatrix getNeuronDerivedValueAsMatrix (size_t index) const;
		/// <summary>
		/// Returns the weights matrix
		/// </summary>
		/// <param name=""></param>
		/// <returns></returns>
		PtrMatrix getWeightsMatrix(size_t) const;
		/// <summary>
		/// Sets the neuron value
		/// </summary>
		/// <param name="indexLayer">The index of the layer inside the Neural Network</param>
		/// <param name="indexNeuron">The index of neuron inside layer</param>
		/// <param name="value">The value to be setted</param>
		void setNeuronValue(size_t indexLayer, size_t indexNeuron, double value);
		/// <summary>
		/// Set's the current input of the Neural Network
		/// </summary>
		/// <param name="input">The input value</param>
		void SetCurrentInput(const vDouble& input);
		/// <summary>
		/// Sets the target of the neural network
		/// </summary>
		/// <param name="target">The target of the neural network</param>
		void SetCurrentTarget(const vDouble& target);
		/// <summary>
		/// Prints the UI interface
		/// </summary>
		void Print(UIInterface*) const override;
		/// <summary>
		/// Returns the total error
		/// </summary>
		/// <returns>Returns the total error</returns>
		double getTotalError() const;
		/// <summary>
		/// Returns the total errors
		/// </summary>
		/// <returns></returns>
		vDouble getTotalErrors() const;
		/// <summary>
		/// Build the errors
		/// </summary>
		void setErrors();
		/// <summary>
		/// Perform the back propagation
		/// </summary>
		void BackPropagation();
		/// <summary>
		/// Prints the output of the Neural Network
		/// </summary>
		void PrintOutput();
		/// <summary>
		/// Prints the target of the Neural Network
		/// </summary>
		void PrintTarget();
		/// <summary>
		/// Train the neural network
		/// </summary>
		/// <param name="noEpock"></param>
		void Train(int noEpock);
		/// <summary>
		/// Save the Object
		/// </summary>
		void Save(const std::string&,IOStrategy strategy);

		void Load(const std::string& filename, IOStrategy strategy) {}

		void NeuralNetwork::SetCurrentInput(const RealHostMatrix& input);
	};

}

