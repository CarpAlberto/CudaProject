#pragma once
#include "template_data.h"

namespace gpuNN 
{
	template <int blockSize> __device__ __forceinline__ void SumBeforeWarp(cudafloat * s) {
		if (blockSize >= 1024) {
			if (threadIdx.x < 512) s[threadIdx.x] += s[threadIdx.x + 512];
			__syncthreads();
		}

		if (blockSize >= 512) {
			if (threadIdx.x < 256) s[threadIdx.x] += s[threadIdx.x + 256];
			__syncthreads();
		}

		if (blockSize >= 256) {
			if (threadIdx.x < 128) s[threadIdx.x] += s[threadIdx.x + 128];
			__syncthreads();
		}

		if (blockSize >= 128) {
			if (threadIdx.x < 64) s[threadIdx.x] += s[threadIdx.x + 64];
			__syncthreads();
		}
	}

	template <int blockSize> __device__ __forceinline__ void SumWarp(volatile cudafloat * s) {
		if (blockSize >= 64)
			s[threadIdx.x] += s[threadIdx.x + 32];
		if (blockSize >= 32)
			s[threadIdx.x] += s[threadIdx.x + 16];
		if (blockSize >= 16)
			s[threadIdx.x] += s[threadIdx.x + 8];
		if (blockSize >= 8)
			s[threadIdx.x] += s[threadIdx.x + 4];
		if (blockSize >= 4)
			s[threadIdx.x] += s[threadIdx.x + 2];
		if (blockSize >= 2)
			s[threadIdx.x] += s[threadIdx.x + 1];
	}

	__global__ void kSigmoid(const int nThreads, float const *input, float *output);

	__global__ void kSigmoid_d(const int nThreads, float const *input, float *output);

	__global__ void kDot(const int nThreads, const float *m1, const float *m2, float *output,
		const int m1_rows, const int m1_columns, const int m2_columns);

	__global__ void kTanh(const int nThreads, float const *input, float *output);

	__global__ void kTanhDerivative(const int nThreads, float const *input, float *output);

	__global__ void cuda_activate_output_layer(float * inputs, float * weights,
		int mOffset, float * expected_outputs,
		float * outputs, float * localGradient, float * rms);

	__global__ void cuda_activate_layer(float * inputs, float * weights, int mOffset, float * outputs);

	template <int blockSize> __global__ 
		void cuda_activate_layerTemplate(float * inputs, float * weights, int mOffset, float * outputs, int numInputs)
	{
		extern __shared__ cudafloat iw[];

		iw[threadIdx.x] = CUDA_VALUE(0.0);
		for (int i = threadIdx.x; i <= numInputs; i += blockDim.x) 
		{
			cudafloat i_w = weights[blockIdx.x * (numInputs + 1) + i];
			if (i > 0) 
				i_w *= inputs[blockIdx.y * numInputs + (i - 1)];
			iw[threadIdx.x] += i_w;
		}
		__syncthreads();

		SumBeforeWarp<blockSize>(iw);

		if (threadIdx.x < 32) {
			SumWarp<blockSize>(iw);

			if (threadIdx.x == 0) {
				cudafloat output = sigmoid(iw[0]);
				outputs[blockIdx.y * gridDim.x + blockIdx.x] = output;
			}
		}
	}

	void cuda_activate_layerWrapper(cudaStream_t stream, dim3 & gridSize, int blockSize,
		float * inputs, float * weights, int mOffset, float * outputs, int numInputs);

	__global__ void cuda_calculate_gradients(float * rmsF,
		float * outputs, float* weights, int mOffset,
		float * localGradientNextLayer, int neuronsNextLayer,
		int neurons, float * localGradient);


	template <int blockSize>
	__global__ void cuda_correct_weights(float * rmsF, float * inputs, float * localGradient, float * weights,
		float * learningRate, float * lastDeltaWithoutLearningMomentum, float * lastDelta,
		float maxStepSize, float u, float d, float momentum, int numberPatterns)
	{

		extern __shared__ cudafloat deltas[];
		deltas[threadIdx.x] = 0.0;

		for (int p = threadIdx.x; p < numberPatterns; p += blockDim.x)
		{
			float delta = localGradient[p * gridDim.y + blockIdx.y];
			if (blockIdx.x > 0)
				delta *= inputs[p * (gridDim.x - 1) + (blockIdx.x - 1)];

			deltas[threadIdx.x] += delta;
		}
		__syncthreads();

		SumBeforeWarp<blockSize>(deltas);

		if (threadIdx.x < 32) {
			SumWarp<blockSize>(deltas);
			if (threadIdx.x == 0) {
				int connection = blockIdx.y * gridDim.x + blockIdx.x;

				float delta = deltas[0] / numberPatterns;
				float learnRate = learningRate[connection];

				float factor = same(lastDeltaWithoutLearningMomentum[connection], delta) ? u : d;
				learnRate *= factor;
				if (learnRate > maxStepSize)
					learnRate = maxStepSize;
				learningRate[connection] = learnRate;

				lastDeltaWithoutLearningMomentum[connection] = delta;

				delta += momentum * lastDelta[connection];
				lastDelta[connection] = delta;

				float w = weights[connection] + (learnRate * delta);
				if (!isfinite(w)) {
					lastDelta[connection] = 0.0;
					lastDeltaWithoutLearningMomentum[connection] = 0.0;
				}
				else {
					weights[connection] = w;
				}
			}
		}

	}

	void cuda_correct_weights_Wrapper(cudaStream_t stream, dim3 & gridSize, int blockSize, float * rmsF, float * inputs, float * localGradient, float * weights,
		float * learningRate, float * lastDeltaWithoutLearningMomentum, float * lastDelta,
		float maxStepSize, float u, float d, float momentum, int numberPatterns);

	template <int blockSize>
	__global__ void cuda_calculate_errors(float* rms, float* rmsF, int patternsNo,
		float numberPatternsNeurons)
	{

		extern __shared__ cudafloat shared_rms[];
		shared_rms[threadIdx.x] = 0.0;

		for (int p = threadIdx.x; p < patternsNo; p += blockDim.x)
			shared_rms[threadIdx.x] += rms[p];
		__syncthreads();

		SumBeforeWarp<blockSize>(shared_rms);

		if (threadIdx.x < 32) {
			SumWarp<blockSize>(shared_rms);

			if (threadIdx.x == 0) {
				cudafloat fRMS = sqrtf(shared_rms[0] / numberPatternsNeurons) / 2.0;
				if (!isfinite(fRMS))
					fRMS = numberPatternsNeurons;
				*rmsF = fRMS;
			}
		}
	}

	void cuda_Calculate_errorsWrapper(cudaStream_t stream, int blockSize,
		float* rms, float* rmsF, int patternsNo,
		float numberPatternsNeurons);
}




