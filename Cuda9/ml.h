#pragma once
#include "template_data.h"

namespace gpuNN 
{
	__device__ void cuda_sum(volatile float * s);

	__device__ void cuda_sum_before(float * s);

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

	__global__ void cuda_calculate_gradients(float * rmsF,
		float * outputs, float* weights, int mOffset,
		float * localGradientNextLayer, int neuronsNextLayer,
		int neurons, float * localGradient);

	__global__ void cuda_correct_weights(float * rmsF, float * inputs, float * localGradient, float * weights,
		float * learningRate, float * lastDeltaWithoutLearningMomentum, float * lastDelta,
		float maxStepSize, float u, float d, float momentum, int numberPatterns);

	__global__ void cuda_calculate_errors(float* rms, float* rmsF, int patternsNo, float numberPatternsNeurons);

}




