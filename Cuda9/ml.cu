#include "ml.h"
#include "cuda_macro.h"

namespace gpuNN
{

	__device__ void cuda_sum_weights(int connection, float * inputs, float * weights)
	{
		extern __shared__ cudafloat tempWeights[];

		tempWeights[connection] = weights[connection];

		if (threadIdx.x > 0)
			tempWeights[connection] *= inputs[blockIdx.x * (blockDim.x - 1) + (threadIdx.x - 1)];

		__syncthreads();

		int sumElements = blockDim.x;

		for (int sumUpTo = (sumElements >> 1); sumElements > 1; sumUpTo = (sumElements >> 1)) {

			int nextNumberElemSum = sumUpTo;
			if (sumElements & 1) 
				nextNumberElemSum++;

			if (threadIdx.x < sumUpTo)
				tempWeights[connection] += tempWeights[connection + nextNumberElemSum];

			sumElements = nextNumberElemSum;
			__syncthreads();
		}
	}

	__global__ void kSigmoid(const int nThreads, float const *input, float *output) {
		
		for (int i = blockIdx.x * blockDim.x + threadIdx.x;
			i < nThreads;
			i += blockDim.x * gridDim.x)
		{
			output[i] = 1.0 / (1.0 + std::exp(-input[i]));
		}
	}

	__global__ void kTanh(const int nThreads, float const *input, float *output) {
		
		for (int i = blockIdx.x * blockDim.x + threadIdx.x;
			i < nThreads;
			i += blockDim.x * gridDim.x)
		{
			output[i] = tanh(input[i]);
		}
	}

	__global__ void kTanhDerivative(const int nThreads, float const *input, float *output)
	{
		
		for (int i = blockIdx.x * blockDim.x + threadIdx.x;
			i < nThreads;
			i += blockDim.x * gridDim.x)
		{
			output[i] = 1 - (tanh(input[i]) * tanh(input[i]));
		}
	}

	__global__ void kSigmoid_d(const int nThreads, float const *input, float *output) {

		for (int i = blockIdx.x * blockDim.x + threadIdx.x;
			i < nThreads;
			i += blockDim.x * gridDim.x)
		{
			output[i] = input[i] * (1 - input[i]);
		}
	}

	__global__ void kDot(const int nThreads, const float *m1, const float *m2, float *output,
		const int m1_rows, const int m1_columns, const int m2_columns) {

		for (int i = blockIdx.x * blockDim.x + threadIdx.x;
			i < nThreads;
			i += blockDim.x * gridDim.x)
		{
			int r = (int)i / m2_columns;
			int c = i % m2_columns;
			float t_output = 0.f;

			for (int k = 0; k < m1_columns; ++k) {
				t_output += m1[r * m1_columns + k] * m2[k * m2_columns + c];
			}

			output[i] = t_output;
		}
	}

	__global__ void cuda_activate_output_layer(float * inputs, float * weights,
		int mOffset, float * expected_outputs,float * outputs, float * gradients, float * rootMeanSquare)
	{
		
		int connection = threadIdx.y * blockDim.x + threadIdx.x;
		cuda_sum_weights(connection, inputs, weights);
		
		extern __shared__ float tempWeights[];
		float * sharedRootMeanSquare = (tempWeights + (blockDim.x * blockDim.y));
		
		if (threadIdx.x == 0)
		{
			int n = blockIdx.x * blockDim.y + threadIdx.y;
			float output = sigmoid(tempWeights[connection]);

			float outn = output;
			float error = (expected_outputs[n] - output);
			
			outputs[n] = outn;
			gradients[n] = error *sigmoid_derivate(output);

			sharedRootMeanSquare[threadIdx.y] = error * error;
		}

		if (blockDim.y > 1)
		{
			__syncthreads();

			if (threadIdx.x == 0 && (threadIdx.y & 1) == 0 && threadIdx.y + 1 < blockDim.y)
				sharedRootMeanSquare[threadIdx.y] += sharedRootMeanSquare[threadIdx.y + 1];
			__syncthreads();

			int nextInterval;
			for (int interval = 2; interval < blockDim.y; interval = nextInterval)
			{
				nextInterval = interval << 1;

				if (threadIdx.x == 0 && (threadIdx.y & (nextInterval - 1)) == 0 && threadIdx.y + interval < blockDim.y)
					sharedRootMeanSquare[threadIdx.y] += sharedRootMeanSquare[threadIdx.y + interval];
				__syncthreads();
			}
		}

		if (threadIdx.y == 0 && threadIdx.x == 0)
			rootMeanSquare[blockIdx.x] = sharedRootMeanSquare[0];
	}

	__global__ void cuda_activate_layer(float * inputs, float * weights,int mOffset, float * outputs)
	{
		
		extern __shared__ float tempWeights[];
		int connection = threadIdx.y * blockDim.x + threadIdx.x;
		cuda_sum_weights(connection, inputs, weights);

		if (threadIdx.x == 0) {
			int n = blockIdx.x * blockDim.y + threadIdx.y;
			cudafloat output = sigmoid(tempWeights[connection]);
			outputs[n] = output;
		}
	}

	__global__ void cuda_calculate_gradients(float * rmsF,
		float * outputs, float* weights, int mOffset,
		float * localGradientNextLayer, int neuronsNextLayer,
		int neurons, float * localGradient)
	{
		extern __shared__ cudafloat lg[];

		float * lgNextLayer = (lg + (blockDim.y * blockDim.x));

		int threadId = (threadIdx.y * blockDim.x + threadIdx.x);

		for (int neuron = threadIdx.y; neuron < neurons + threadIdx.y; neuron += blockDim.y) {
			lg[threadId] = 0;

			for (int outputNeuron = threadIdx.x; outputNeuron < neuronsNextLayer + threadIdx.x;
				outputNeuron += blockDim.x) {
				if (threadIdx.y == 0 && outputNeuron < neuronsNextLayer) {
					lgNextLayer[threadIdx.x] = localGradientNextLayer[blockIdx.x * neuronsNextLayer + outputNeuron];
				}
				__syncthreads();

				if (outputNeuron < neuronsNextLayer && neuron < neurons) {
					int connection = outputNeuron * (neurons + 1) + neuron + 1;
					lg[threadId] += weights[connection] * lgNextLayer[threadIdx.x];
				}
				__syncthreads();
			}

			int numberElemSum = blockDim.x;
			for (int sumUpTo = (numberElemSum >> 1); numberElemSum > 1; sumUpTo = (numberElemSum >> 1)) {
				int nextNumberElemSum = sumUpTo;
				if (numberElemSum & 1) 
					nextNumberElemSum++;

				if (threadIdx.x < sumUpTo) 
					lg[threadId] += lg[threadId + nextNumberElemSum];

				numberElemSum = nextNumberElemSum;

				__syncthreads();
			}

			if (threadIdx.x == 0 && neuron < neurons) {

				int n = blockIdx.x * neurons + neuron;
				cudafloat Fh = outputs[n];
				cudafloat lgn = lg[threadId];
				localGradient[n] = lgn * CUDA_SIGMOID_DERIVATE(Fh);
			}
		} 
	}


	__global__ void cuda_calculate_weights_block128(float * rmsF, float * inputs, float * localGradient, float * selectiveNeuronsWeights,
		float * selectiveNeuronsBias, float * learningRateWeights, float * learningRateBias, float * lastDeltaWithoutLearningMomentumWeights,
		float * lastDeltaWithoutLearningMomentumBias, float * lastDeltaWeights, float * lastDeltaBias,
		float u, float d, float r, float maxStepSize,
		float momentum, int numberPatterns)
	{
		
		extern __shared__ float deltasWeights[];
		float * deltasBias = (deltasWeights + blockDim.x);
		deltasBias[threadIdx.x] = 0.0;
		deltasWeights[threadIdx.x] = 0.0;
		for (int p = threadIdx.x; p < numberPatterns; p += blockDim.x) {
			int n = p * gridDim.x + blockIdx.x;
			float i = inputs[n];
			if (!isfinite(i)) {
				float delta = localGradient[n];
				deltasBias[threadIdx.x] += delta;
				deltasWeights[threadIdx.x] += delta * i;
			}
		}
		__syncthreads();

		if (threadIdx.x < 64)
		{
			deltasBias[threadIdx.x] += deltasBias[threadIdx.x + 64];
			deltasWeights[threadIdx.x] += deltasWeights[threadIdx.x + 64];
		}
		__syncthreads();

		if (threadIdx.x < 32)
		{
			volatile float * _deltasBias = deltasBias;
			volatile float * _deltasWeights = deltasWeights;

			// Perform the unroll looping
			_deltasBias[threadIdx.x] += _deltasBias[threadIdx.x + 32];
			_deltasWeights[threadIdx.x] += _deltasWeights[threadIdx.x + 32];

			_deltasBias[threadIdx.x] += _deltasBias[threadIdx.x + 16];
			_deltasWeights[threadIdx.x] += _deltasWeights[threadIdx.x + 16];

			_deltasBias[threadIdx.x] += _deltasBias[threadIdx.x + 8];
			_deltasWeights[threadIdx.x] += _deltasWeights[threadIdx.x + 8];

			_deltasBias[threadIdx.x] += _deltasBias[threadIdx.x + 4];
			_deltasWeights[threadIdx.x] += _deltasWeights[threadIdx.x + 4];

			_deltasBias[threadIdx.x] += _deltasBias[threadIdx.x + 2];
			_deltasWeights[threadIdx.x] += _deltasWeights[threadIdx.x + 2];

			_deltasBias[threadIdx.x] += _deltasBias[threadIdx.x + 1];
			_deltasWeights[threadIdx.x] += _deltasWeights[threadIdx.x + 1];

			if (threadIdx.x == 0)
			{
				float deltaB = deltasBias[0] / numberPatterns;
				float deltaW = deltasWeights[0] / numberPatterns;

				float learnRateB = learningRateBias[blockIdx.x];
				float learnRateW = learningRateWeights[blockIdx.x];

				float factorB = same(lastDeltaWithoutLearningMomentumBias[blockIdx.x], deltaB) ? u : d;
				float factorW = same(lastDeltaWithoutLearningMomentumWeights[blockIdx.x], deltaW) ? u : d;

				learnRateB *= factorB;
				learnRateW *= factorW;

				if (learnRateB > maxStepSize) learnRateB = maxStepSize;
				if (learnRateW > maxStepSize) learnRateW = maxStepSize;

				learningRateBias[blockIdx.x] = learnRateB;
				learningRateWeights[blockIdx.x] = learnRateW;

				lastDeltaWithoutLearningMomentumBias[blockIdx.x] = deltaB;
				lastDeltaWithoutLearningMomentumWeights[blockIdx.x] = deltaW;

				deltaB += momentum * lastDeltaBias[blockIdx.x];
				deltaW += momentum * lastDeltaWeights[blockIdx.x];

				lastDeltaBias[blockIdx.x] = deltaB;
				lastDeltaWeights[blockIdx.x] = deltaW;

				float wb = selectiveNeuronsBias[blockIdx.x] + (learnRateB * deltaB);
				float w = selectiveNeuronsWeights[blockIdx.x] + (learnRateW * deltaW);

				if (!isfinite(wb)) 
				{
					lastDeltaBias[blockIdx.x] = 0.0;
					lastDeltaWithoutLearningMomentumBias[blockIdx.x] = 0.0;
				}
				else
				{
					selectiveNeuronsBias[blockIdx.x] = wb;
				}

				if (!isfinite(w)) {
					lastDeltaWeights[blockIdx.x] = 0.0;
					lastDeltaWithoutLearningMomentumWeights[blockIdx.x] = 0.0;
				}
				else {
					selectiveNeuronsWeights[blockIdx.x] = w;
				}
			}
		}
	}


	void cuda_activate_layerWrapper(cudaStream_t stream, dim3 & gridSize, int blockSize,
		float * inputs, float * weights, int mOffset, float * outputs, int numInputs) {

		switch (blockSize) {
		case 1:
			cuda_activate_layerTemplate<1> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >(inputs, weights, mOffset, outputs, numInputs);
			break;
		case 2:
			cuda_activate_layerTemplate<2> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >(inputs, weights, mOffset, outputs, numInputs);
			break;
		case 4:
			cuda_activate_layerTemplate<4> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >(inputs, weights, mOffset, outputs, numInputs);
			break;
		case 8:
			cuda_activate_layerTemplate<8> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >(inputs, weights, mOffset, outputs, numInputs);
			break;
		case 16:
			cuda_activate_layerTemplate<16> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >(inputs, weights, mOffset, outputs, numInputs);
			break;
		case 32:
			cuda_activate_layerTemplate<32> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >(inputs, weights, mOffset, outputs, numInputs);
			break;
		case 64:
			cuda_activate_layerTemplate<64> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >(inputs, weights, mOffset, outputs, numInputs);
			break;
		case 128:
			cuda_activate_layerTemplate<128> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >(inputs, weights, mOffset, outputs, numInputs);
			break;
		case 256:
			cuda_activate_layerTemplate<256> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >(inputs, weights, mOffset, outputs, numInputs);
			break;
		case 512:
			cuda_activate_layerTemplate<512> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >(inputs, weights, mOffset, outputs, numInputs);
			break;
		case 1024:
			cuda_activate_layerTemplate<1024> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >(inputs, weights, mOffset, outputs, numInputs);
			break;
		case 2048:
			cuda_activate_layerTemplate<2048> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >(inputs, weights, mOffset, outputs, numInputs);
			break;
		}
	}

	void cuda_Calculate_errorsWrapper(cudaStream_t stream, int blockSize,
		float* rms, float* rmsF, int patternsNo,
		float numberPatternsNeurons)
	{
		switch (blockSize) {
		case 1024:
			cuda_calculate_errors<1024> << <1, blockSize, blockSize * sizeof(cudafloat), stream >> >
				(rms, rmsF, patternsNo, numberPatternsNeurons);
			break;
		case 512:
			cuda_calculate_errors<512> << <1, blockSize, blockSize * sizeof(cudafloat), stream >> >
				(rms, rmsF, patternsNo, numberPatternsNeurons);
			break;

		case 256:
			cuda_calculate_errors<256> << <1, blockSize, blockSize * sizeof(cudafloat), stream >> >
				(rms, rmsF, patternsNo, numberPatternsNeurons);
			break;

		case 128:
			cuda_calculate_errors<128> << <1, blockSize, blockSize * sizeof(cudafloat), stream >> >
				(rms, rmsF, patternsNo, numberPatternsNeurons);
			break;

		case 64:
			cuda_calculate_errors<64> << <1, blockSize, blockSize * sizeof(cudafloat), stream >> >
				(rms, rmsF, patternsNo, numberPatternsNeurons);
			break;

		case 32:
			cuda_calculate_errors<32> << <1, blockSize, blockSize * sizeof(cudafloat), stream >> >
				(rms, rmsF, patternsNo, numberPatternsNeurons);
			break;

		case 16:
			cuda_calculate_errors<16> << <1, blockSize, blockSize * sizeof(cudafloat), stream >> >
				(rms, rmsF, patternsNo, numberPatternsNeurons);
			break;

		case 8:
			cuda_calculate_errors<8> << <1, blockSize, blockSize * sizeof(cudafloat), stream >> >
				(rms, rmsF, patternsNo, numberPatternsNeurons);
			break;

		case 4:
			cuda_calculate_errors<4> << <1, blockSize, blockSize * sizeof(cudafloat), stream >> >
				(rms, rmsF, patternsNo, numberPatternsNeurons);
			break;

		case 2:
			cuda_calculate_errors<2> << <1, blockSize, blockSize * sizeof(cudafloat), stream >> >
				(rms, rmsF, patternsNo, numberPatternsNeurons);
			break;

		case 1:
			cuda_calculate_errors<1> << <1, blockSize, blockSize * sizeof(cudafloat), stream >> >
				(rms, rmsF, patternsNo, numberPatternsNeurons);
			break;
		}
	}

	void cuda_correct_weights_Wrapper(cudaStream_t stream, dim3 & gridSize, int blockSize,
		float * rmsF, float * inputs, float * localGradient, float * weights,
		float * learningRate, float * lastDeltaWithoutLearningMomentum, float * lastDelta,
		float maxStepSize, float u, float d, float momentum, int numberPatterns)
	{
		switch (blockSize) {

			case 1024:
			cuda_correct_weights<1024> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >
				(rmsF, inputs, localGradient, weights, learningRate,
					lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, momentum, numberPatterns);
			break;
			case 512:
			cuda_correct_weights<512> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >
				(rmsF, inputs, localGradient, weights, learningRate,
					lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, momentum, numberPatterns);
			break;
			case 256:
			cuda_correct_weights<256> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >
				(rmsF, inputs, localGradient, weights, learningRate,
					lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, momentum, numberPatterns);
			break;
			case 128:
				cuda_correct_weights<128> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >
					(rmsF, inputs, localGradient, weights, learningRate,
						lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, momentum, numberPatterns);
				break;
			case 64:
				cuda_correct_weights<64> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >
					(rmsF, inputs, localGradient, weights, learningRate,
						lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, momentum, numberPatterns);
				break;
			case 32:
				cuda_correct_weights<32> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >
					(rmsF, inputs, localGradient, weights, learningRate,
						lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, momentum, numberPatterns);
				break;
			case 16:
				cuda_correct_weights<16> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >
					(rmsF, inputs, localGradient, weights, learningRate,
						lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, momentum, numberPatterns);
					break;
			case 8:
			   cuda_correct_weights<8> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >
				(rmsF, inputs, localGradient, weights, learningRate,
					lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, momentum, numberPatterns);
			break;

			case 4:
				cuda_correct_weights<4> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> >
					(rmsF, inputs, localGradient, weights, learningRate,
						lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, momentum, numberPatterns);
				break;
			case 2:
				cuda_correct_weights<2> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >> > 
					(rmsF, inputs, localGradient, weights, learningRate,
						lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, momentum, numberPatterns);
				break;
			case 1:
				cuda_correct_weights<1> << <gridSize, blockSize, blockSize * sizeof(cudafloat), stream >>>
					(rmsF,inputs, localGradient, weights, learningRate, 
						lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d,momentum, numberPatterns);
				break;
		}
	}
}