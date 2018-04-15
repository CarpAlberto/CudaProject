#include "OptimizedLayer.h"

int OptimizedLayer::totalNeuronsWithSelectiveActivation = 0;
int OptimizedLayer::patterns;

void OptimizedLayer::RandWeights(cudafloat min, cudafloat max, cudafloat learningRate) {

	RealCpuArray learnRate(this->numberConnection);
	RealCpuArray delta(this->numberConnection);
	RealCpuArray weights(this->numberConnection);

	for (int iterator = 0; iterator < numberConnection; iterator++) {
		weights[iterator] = (max - min) * ((cudafloat)rand() / RAND_MAX) + min;
		learnRate[iterator] = learningRate;
		delta[iterator] = 0;
	}
	this->vBestWeights = this->vWeights = weights;
	this->vLearnRate = learnRate;
	this->vLastDelta = this->vLastDeltaNoMomentum= delta;
}

void OptimizedLayer::Activate(cudaStream_t stream)
{
	dim3 neuronsPatterns;
	neuronsPatterns.x = this->numberNeurons;
	int processed;
	if (this->isOutputLayer) 
	{
		processed = 0;
		do 
		{
			int size = (patterns > CUDA_MAX_GRID_X_DIM) ? CUDA_MAX_GRID_X_DIM : patterns;
			
			float* vInputFloat = this->floatDestinationInputs + (processed * numberInputNoBias);
			float* vOutputFloat = vOutputs.Data() + (processed * this->numberNeurons);
			
			cuda_activate_output_layer <<<size, dimInputsNeurons, sharedMemFire, stream >>>
				(vInputFloat,vWeights.Data(),0,this->floatDestinationOutputs + (processed * this->numberNeurons),
				vOutputFloat,
				vLocalGradients.Data() + (processed * this->numberNeurons),
				this->floatRootMeanSquare + processed);

			processed += size;
		} while (processed < patterns);
	}
	else
	{
		processed = 0;
		do {
			int size = (patterns > CUDA_MAX_GRID_X_DIM) ? CUDA_MAX_GRID_X_DIM : patterns;
			cuda_activate_layer <<< size, dimInputsNeurons, sharedMemFire, stream >>>
				(this->floatDestinationInputs + (processed * this->numberInputNoBias),
				vWeights.Data(),
				0, vOutputs.Data() + (processed * this->numberNeurons));
			processed += size;
		} while (processed < patterns);
	}

}

void OptimizedLayer::InitLayer(int neurons, int inputs, int nextLayerNeurons,cudafloat initialLearningRate, float * layerInputs,
	bool isOutputLayer)
{
	/*Setup the informations*/
	this->numberNeurons = neurons;
	this->numberNextLayerNeurons = nextLayerNeurons;
	this->numberConnection = inputs * neurons;
	this->numberInputNoBias = inputs - 1;

	/*Randomize the weights*/
	this->RandWeights(-1, 1, initialLearningRate);

	inputsBlockSize = 1;
	while (inputsBlockSize < CUDA_MAX_THREADS_PER_BLOCK && inputsBlockSize < inputs)
		inputsBlockSize <<= 1;

	this->floatDestinationInputs = layerInputs;
	this->vOutputs.Resize(neurons * patterns);
	this->vLocalGradients.Resize(neurons * patterns);

	dimInputsNeurons.x = inputs;
	dimInputsNeurons.y = neurons;

	dimOutputsNeurons.x = nextLayerNeurons;
	dimOutputsNeurons.y = neurons;

	sharedMemFire = this->numberConnection * sizeof(float);
	sharedMemGradients = (dimOutputsNeurons.x * (dimOutputsNeurons.y + 1)) * sizeof(float);

	this->isOutputLayer = isOutputLayer;
}


void OptimizedLayer::BuildGradients(cudaStream_t stream, float * rootMeanSquare, OptimizedLayer & nextLayer)
{
	int processed = 0;
	do 
	{
		int patternsToProcess = (patterns > CUDA_MAX_GRID_X_DIM) ? CUDA_MAX_GRID_X_DIM : patterns;
		
		/*Build the floats from the cuda obj*/
		auto ActualOutputsFloat  = vOutputs.Data() + (processed * this->numberNeurons);
		auto NextLayerFloat		 = nextLayer.vLocalGradients.Data() + (processed * dimOutputsNeurons.x);
		auto LocalGradientsFloat = this->vLocalGradients.Data() + (processed * this->numberNeurons);
		
		/*Perform the kernel code*/
		cuda_calculate_gradients<<<patternsToProcess, dimOutputsNeurons, sharedMemGradients,stream>>>
			(rootMeanSquare, ActualOutputsFloat,this->vWeights.Data(),
			0,NextLayerFloat,
			this->numberNextLayerNeurons,
			this->numberNeurons, 
			LocalGradientsFloat);
		processed += patternsToProcess;

	} while (processed < patterns);
}

void OptimizedLayer::BuildWeights(cudaStream_t stream, int patternsBlockSize, float * rms, float momentum,
	float u, float d, float maxStepSize){
	
	cuda_correct_weights<<<dimInputsNeurons, BLOCK_SIZE, BLOCK_SIZE * sizeof(float),stream>>>(rms, 
		this->floatDestinationInputs, vLocalGradients.Data(),
		vWeights.Data(), vLearnRate.Data(),
		vLastDeltaNoMomentum.Data(), vLastDelta.Data(), maxStepSize,
		u, d, momentum, OptimizedLayer::patterns);
}

