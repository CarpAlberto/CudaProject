#pragma once
#include "OptimizedLayer.h"
#include "INeuralNetwork.h"

namespace gpuNN {
	class OptimizedNeuralNetwork : public INeuralNetwork {

	protected:
		RealDeviceMatrix vInputs;
		RealDeviceMatrix vOutputs;
		int maxNumberWeigths;
		float initialLearningRate;
		float momentum;
		float u;
		float d;
		float maxStepSize;
		GpuArray<float> d_rms;
		GpuArray<float> d_rmsOut;
		DeviceAccessibleVariable<cudafloat> rms;
		CudaStream streamKernels;
		CudaStream streamRMS;
		int patternsBlockSize;
		cudafloat numberPatternsNeurons;
		int epoch;
		GpuArray<int>		 d_numberWeightsLayer;
		GpuArray<cudafloat*>  d_weightsLayers;
		GpuArray<cudafloat *> d_bestWeightsLayers;
		GpuArray<cudafloat *> d_learnRatesLayers;
		GpuArray<cudafloat *> d_lastDeltaLayers;
		GpuArray<cudafloat *>    d_lastDeltaWithoutLMlayers;
		CpuArray<OptimizedLayer> mlayers;

	public:
		OptimizedNeuralNetwork(IntCpuArray& layers, RealHostMatrix& inputs,
			RealHostMatrix & desiredOutputs, float initialLearningRate);

		OptimizedNeuralNetwork() = default;

		void RandomizeValue(float min, float max);

		void Activate();

		float GetMomentum() const;

		void SetMomentum(float value);

		int  GetEpoch() const;

		float GetRMS();

		virtual void Train(int epochs);

		void SetCurrentInput(const vDouble& input) {};

		void SetCurrentTarget(const vDouble& target) {};

		int GetNumberInputs() const;

		int GetNumberOutputs() const;

		int GetNumbersLayers() const;

		int GetNumberNeurons(int layer) const;

		/// <summary>
		/// Returns the weights associated with the given layer
		/// </summary>
		/// <param name="layer">The given layer</param>
		/// <returns>The associated weights</returns>
		RealCpuArray GetWeights(int layer) const;

		virtual void Save(const std::string& filename, IOStrategy strategy);
	};
}