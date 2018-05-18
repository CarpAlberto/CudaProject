#pragma once
#include "ml.h"

namespace gpuNN {
	/// <summary>
	/// An optimized implementation of the ANN
	/// </summary>
	class OptimizedLayer {

		//Public for easy access
	public:
		/// <summary>
		/// The weights of the layer
		/// </summary>
		RealGpuArray vWeights;
		/// <summary>
		/// The best weights of the layer
		/// </summary>
		RealGpuArray vBestWeights;
		/// <summary>
		/// The learn rate
		/// </summary>
		RealGpuArray vLearnRate;
		/// <summary>
		/// Array last delta
		/// </summary>
		RealGpuArray vLastDelta;
		/// <summary>
		/// Array last delta with no momentum
		/// </summary>
		RealGpuArray vLastDeltaNoMomentum;
		/// <summary>
		/// Array of outputs
		/// </summary>
		RealGpuArray vOutputs;
		/// <summary>
		/// Array of local gradeints
		/// </summary>
		RealGpuArray vLocalGradients;

		/// <summary>
		/// Number of neurons
		/// </summary>
		int numberNeurons;
		/// <summary>
		/// Number of next layer neurons
		/// </summary>
		int numberNextLayerNeurons;
		/// <summary>
		/// Number of connection
		/// </summary>
		int numberConnection;
		/// <summary>
		/// Number of inputh with no bias
		/// </summary>
		int numberInputNoBias;

		cudafloat * floatDestinationInputs;
		cudafloat * floatDestinationOutputs;
		cudafloat * floatRootMeanSquare;


		dim3 dimInputsNeurons;
		dim3 dimOutputsNeurons;
		int inputsBlockSize;
		int sharedMemFire;
		int sharedMemGradients;
		bool isOutputLayer;

	public:
		/// <summary>
		/// Creates random values for weights between values given by the min and max values
		/// </summary>
		/// <param name="minValue"></param>
		/// <param name="maxValue"></param>
		/// <param name="initialLearningRate"></param>
		void RandWeights(cudafloat minValue, cudafloat maxValue, cudafloat initialLearningRate);
		
		/// <summary>
		/// Activates the layer given the stream
		/// </summary>
		/// <param name="stream">The given stream</param>
		void Activate(cudaStream_t stream);

		/// <summary>
		/// Initializie the layer with the nerons inputs and 
		/// </summary>
		/// <param name="neurons">The given neurons</param>
		/// <param name="inputs">The given inputs</param>
		/// <param name="nextLayerNeurons">The neurons of the next inputs</param>
		/// <param name="initialLearningRate">The initial learning rate</param>
		/// <param name="layerInputs"></param>
		/// <param name="isOutputLayer">True if it is an output layer</param>
		void InitLayer(int neurons, int inputs, int nextLayerNeurons,
			cudafloat initialLearningRate, float * layerInputs, 
			bool isOutputLayer);

		void UpdateInput(int neurons, int inputs, int nextLayerNeurons,
			cudafloat initialLearningRate, float * layerInputs,bool outputLayer);

		/// <summary>
		/// Build the gradients values
		/// </summary>
		/// <param name="stream">The stream</param>
		/// <param name="rms">The rms</param>
		/// <param name="nextLayer">The next layer</param>
		void BuildGradients(cudaStream_t stream, float * rms, OptimizedLayer & nextLayer);

		/// <summary>
		/// Build the weights values
		/// </summary>
		/// <param name="stream"></param>
		/// <param name="patternsBlockSize"></param>
		/// <param name="rms"></param>
		/// <param name="momentum"></param>
		/// <param name="u"></param>
		/// <param name="d"></param>
		/// <param name="maxStepSize"></param>
		void BuildWeights(cudaStream_t stream, int patternsBlockSize, float * rms, float momentum, 
			float u, float d, float maxStepSize);

		OptimizedLayer() = default;

	public:
		static int totalNeuronsWithSelectiveActivation;
		static int patterns;
	};
}