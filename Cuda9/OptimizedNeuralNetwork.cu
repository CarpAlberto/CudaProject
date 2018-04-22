#include "OptimizedNeuralNetwork.h"

using namespace gpuNN;

OptimizedNeuralNetwork::OptimizedNeuralNetwork(IntCpuArray& sizeLayers, RealHostMatrix& trainInputPatterns,
	RealHostMatrix & trainDesiredOutputPatterns, float initialLearningRate)
{
	/*Init the values*/
	int nsamples = (int)trainInputPatterns.Rows();
	int ninputs = (int)trainInputPatterns.Columns();
	this->vInputs = trainInputPatterns;
	this->vOutputs = trainDesiredOutputPatterns;

	/*Some initialization*/
	momentum = 0.7;
	u = 1.2;
	d = 0.8;
	maxStepSize = 10.0;

	OptimizedLayer::patterns = nsamples;
	d_rmsOut.Resize(1);
	this->initialLearningRate = initialLearningRate;
	int numberLayers = (int)sizeLayers.Size() - 1;
	mlayers.Resize(numberLayers);
	int outputLayer = numberLayers - 1;
	int inputsWithoutBias = sizeLayers[0];

	/*Perform static assertion*/
	assert(OptimizedLayer::patterns > 0 && OptimizedLayer::patterns ==
		(int)trainDesiredOutputPatterns.Rows());
	assert(initialLearningRate > 0.0);
	assert(numberLayers > 0);
	assert(inputsWithoutBias > 0 && inputsWithoutBias == (int)trainInputPatterns.Columns());


	float * layerInputs = vInputs.Data();	

	for (int l = 0; l < numberLayers; l++) {
		int neurons = sizeLayers[l + 1];
		assert(neurons > 0);

		bool isOutputLayer = (l == outputLayer) ? true : false;

		int nextLayerNeurons = (isOutputLayer) ? 0 : sizeLayers[l + 2];

		mlayers[l].InitLayer(neurons, inputsWithoutBias + 1, nextLayerNeurons, initialLearningRate,
			layerInputs, isOutputLayer);

		layerInputs = mlayers[l].vOutputs.Data();
		inputsWithoutBias = neurons;
	}
	
	CpuArray<int>		  numberWeightsLayer(numberLayers);
	CpuArray<cudafloat *> weightsLayers(numberLayers);
	CpuArray<cudafloat *> bestWeightsLayers(numberLayers);
	CpuArray<cudafloat *> learnRatesLayers(numberLayers);
	CpuArray<cudafloat *> lastDeltaLayers(numberLayers);
	CpuArray<cudafloat *> lastDeltaWithoutLMlayers(numberLayers);

	maxNumberWeigths = 0;
	int ll = 0;

	for (int l = 0; l < numberLayers; l++){
		int connections = mlayers[l].numberConnection;
		if (connections > maxNumberWeigths)
			maxNumberWeigths = connections;
		numberWeightsLayer[ll] = connections;
		weightsLayers[ll] = mlayers[l].vWeights.Data();
		bestWeightsLayers[ll] = mlayers[l].vBestWeights.Data();
		learnRatesLayers[ll] = mlayers[l].vLearnRate.Data();
		lastDeltaLayers[ll] = mlayers[l].vLastDelta.Data();
		lastDeltaWithoutLMlayers[ll] = mlayers[l].vLastDeltaNoMomentum.Data();
		ll++;
	}

	d_numberWeightsLayer = numberWeightsLayer;
	d_weightsLayers = weightsLayers;
	d_bestWeightsLayers = bestWeightsLayers;
	d_learnRatesLayers = learnRatesLayers;
	d_lastDeltaLayers = lastDeltaLayers;
	d_lastDeltaWithoutLMlayers = lastDeltaWithoutLMlayers;

	int sizeRMSvector = (mlayers[outputLayer].numberConnection > CUDA_MAX_THREADS_PER_BLOCK)
		? OptimizedLayer::patterns * mlayers[outputLayer].numberNeurons :
		OptimizedLayer::patterns;
	d_rms.Resize(sizeRMSvector);
	mlayers[outputLayer].floatDestinationOutputs = vOutputs.Data();
	mlayers[outputLayer].floatRootMeanSquare = d_rms.Data();
	mlayers[outputLayer].sharedMemFire += mlayers[outputLayer].numberNeurons * 
		sizeof(float);

	CpuArray<cudafloat> h_bestRMS(1);
	h_bestRMS[0] = 1.0;
	rms.Value() = h_bestRMS[0];

	//Other stuff
	patternsBlockSize = 1;
	while (patternsBlockSize < CUDA_MAX_THREADS_PER_BLOCK &&
		patternsBlockSize < OptimizedLayer::patterns) patternsBlockSize <<= 1;
	numberPatternsNeurons = (cudafloat)OptimizedLayer::patterns * (cudafloat)mlayers[outputLayer].numberNeurons;
	epoch = 0;
}

void OptimizedNeuralNetwork::RandomizeValue(float min, float max)
{
	/*Randomize Each Layer*/
	int nLayers = (int)mlayers.Size();
	for (int layer = 0; layer < nLayers; layer++) 
		mlayers[layer].RandWeights(min, max, initialLearningRate);
	epoch = 0;
}

void OptimizedNeuralNetwork::Train(int noEpochs)
{
	for (auto i = 0; i < noEpochs; i++) 
	{
		int numLayers = (int)mlayers.Size();
		int nSpaceLayers = 0;

		this->Activate();
		if (cudaStreamQuery(streamRMS) == cudaSuccess) 
		{
			cuda_calculate_errors<<<1,BLOCK_SIZE,BLOCK_SIZE* sizeof(float), streamRMS >>>(d_rms.Data(),
				d_rmsOut.Data(), (int)d_rms.Size(), numberPatternsNeurons);
			
			rms.UpdateValueAsync(d_rmsOut.Data(), streamRMS);
		}
		cudafloat * rms =  nullptr;
		for (int l = numLayers - 2; l >= 0; l--) {
			mlayers[l].BuildGradients(streamKernels, rms, mlayers[l + 1]);
		}

		for (int l = numLayers - 1; l >= 0; l--) {
			mlayers[l].BuildWeights(streamKernels, patternsBlockSize, rms, momentum, u, d, maxStepSize);
		}
		float gRms = this->GetRMS();
		std::cout << "Epock : " << this->GetEpoch() << "  ";
		std::cout << "RMS : " << gRms << std::endl;
		
		if (gRms < this->minRms) {
			break;
		}
		epoch++;
	}
}

float OptimizedNeuralNetwork::GetMomentum() const
{
	return this->momentum;
}

void OptimizedNeuralNetwork::SetMomentum(float value)
{
	this->momentum = value;
}

float OptimizedNeuralNetwork::GetRMS()
{
	cudaDeviceSynchronize();
	this->Activate();
	cuda_calculate_errors <<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), streamRMS >> >(d_rms.Data(),
		d_rmsOut.Data(), (int)d_rms.Size(), numberPatternsNeurons);
	rms.UpdateValue(d_rmsOut.Data());
	return rms.Value();
}

void OptimizedNeuralNetwork::Activate()
{
	int numLayers = (int)this->mlayers.Size();
	for (int l = 0; l < numLayers; l++)
		this->mlayers[l].Activate(streamKernels);
}

int OptimizedNeuralNetwork::GetNumberInputs() const
{
	return this->mlayers[0].numberInputNoBias;
}

int OptimizedNeuralNetwork::GetNumberOutputs() const
{
	return mlayers[mlayers.Size() - 1].numberNeurons;
}

int OptimizedNeuralNetwork::GetEpoch() const
{
	return this->epoch;
}

int OptimizedNeuralNetwork::GetNumbersLayers() const
{
	return this->mlayers.Size();
}

int OptimizedNeuralNetwork::GetNumberNeurons(int layer) const
{
	return this->mlayers[layer].numberNeurons;
}

RealCpuArray OptimizedNeuralNetwork::GetWeights(int layer) const
{
	return RealCpuArray(this->mlayers[layer].vWeights);
}

void OptimizedNeuralNetwork::Save(const std::string& filename, IOStrategy strategy)
{
	if (strategy != IOStrategy::ASCII) {
		throw new std::exception("Unsuported strategy save");
	}
	std::ofstream os(filename, std::ios_base::out);
	size_t index = 0;

	/*Save the number of inputs*/
	os << this->GetNumberInputs();

	int numLayers = this->GetNumbersLayers();
	
	for (int l = 0; l < numLayers; l++) {
		os << "-";
		os << (this->GetNumberNeurons(l));
	}
	os << std::endl;

	for (int l = 0; l < numLayers; l++) 
	{
		auto weights = this->GetWeights(l);
		size_t numWeights = weights.Size();
		
		for (size_t w = 0; w < numWeights; w++) {
			os << (weights[w]) << std::endl;
			os << "0.0" << std::endl; //delta
			os << "0.0" << std::endl; //delta no momentum
			os << this->initialLearningRate << std::endl;
		}
	}
	os.close();
}

void OptimizedNeuralNetwork::Load(const std::string& filename, IOStrategy strategy)
{
	std::ifstream is(filename);

	if (strategy != IOStrategy::ASCII) {
		throw new std::exception("Unsuported strategy save");
	}
}