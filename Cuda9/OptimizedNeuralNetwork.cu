#include "OptimizedNeuralNetwork.h"

using namespace gpuNN;

OptimizedNeuralNetwork::OptimizedNeuralNetwork(IntCpuArray& sizeLayers, 
	RealHostMatrix& trainInputPatterns,
	RealHostMatrix & trainDesiredOutputPatterns, 
	float initialLearningRate, float minRms)
{
	/*Init the values*/
	int nsamples = (int)trainInputPatterns.Rows();
	int ninputs = (int)trainInputPatterns.Columns();
	this->vInputs = trainInputPatterns;
	this->vOutputs = trainDesiredOutputPatterns;
	this->sizeLayers = sizeLayers;

	/*Some initialization*/
	momentum = 0.4;
	u = 1.2;
	d = 0.8;
	maxStepSize = 5;
	d_rmsOut.Resize(1);

	OptimizedLayer::patterns = nsamples;
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
	assert(inputsWithoutBias > 0 && inputsWithoutBias == 
		(int)trainInputPatterns.Columns());


	float * layerInputs = vInputs.Data();	

	for (int l = 0; l < numberLayers; l++) {
		int neurons = sizeLayers[l + 1];
		assert(neurons > 0);

		bool isOutputLayer = (l == outputLayer) ? true : false;

		int nextLayerNeurons = (isOutputLayer) ? 0 : sizeLayers[l + 2];

		mlayers[l].InitLayer(neurons, inputsWithoutBias + 1, 
			nextLayerNeurons, initialLearningRate,
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

	for (int l = 0; l < numberLayers; l++)
	{
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
	mlayers[outputLayer].sharedMemFire += mlayers[outputLayer].numberNeurons * sizeof(float);

	rms.Value() = 1.0;

	//Other stuff
	patternsBlockSize = 1;
	while (patternsBlockSize < CUDA_MAX_THREADS_PER_BLOCK &&
		patternsBlockSize < OptimizedLayer::patterns) patternsBlockSize <<= 1;
	
	numberPatternsNeurons = (cudafloat)OptimizedLayer::patterns * 
		(cudafloat)mlayers[outputLayer].numberNeurons;
	epoch = 0;
	this->minRms = minRms;
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
			
			cuda_Calculate_errorsWrapper(streamRMS, patternsBlockSize, d_rms.Data(),
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
		/*if (gRms < this->minRms) {
			epoch = 0;
			break;
		}*/
		epoch++;
	}
	epoch = 0;
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
		
	cuda_Calculate_errorsWrapper(streamKernels, patternsBlockSize, d_rms.Data(),
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
		}
	}
	os.close();
}

OptimizedNeuralNetwork OptimizedNeuralNetwork::Load(const std::string& filename,
	IOStrategy strategy)
{
	std::ifstream is(filename);

	if (strategy != IOStrategy::ASCII) {
		throw new std::exception("Unsuported strategy save");
	}

	std::vector<int> layers;
	std::string line;
	std::getline(is, line);
	auto Topology = Utils::Split(line, '-');
	CpuArray<int> vTopology;
	vTopology.Resize(4);
	int i = 0;
	for (auto& item : Topology) {
		vTopology[i] = (stoi(item));
		i++;
	}

	RealHostMatrix input(1, stoi(Topology[0]));

	for (auto i = 0; i < Topology.size(); i++) {
		input(0, i) = 0;
	}
	RealHostMatrix outputs(1, 1);
	outputs(0, 0) = 1.0f;

    OptimizedNeuralNetwork n(vTopology, input, outputs, 0.4,0.001);
	RealCpuArray array;
	for (int l = 0; l < Topology.size()-1; l++)
	{
		auto numWeights = (vTopology[l] + 1) * vTopology[l + 1];
		array.Resize(numWeights);
		float data;
		for (size_t w = 0; w < numWeights; w++) {
			is >> (data);
		}
		n.SetWeights(array, l);
	}

	return std::move(n);
}

void OptimizedNeuralNetwork::SetWeights(RealCpuArray data, int index)
{
	this->mlayers[index].vWeights = data;
}

void OptimizedNeuralNetwork::SetCurrentInput(const RealHostMatrix& input)
{
	int nsamples = (int)input.Rows();
	int ninputs = (int)input.Columns();
	this->vInputs = input;
	FeedForward();
}

void OptimizedNeuralNetwork::UpdateOutput(const RealHostMatrix& output)
{
	this->vOutputs = output;
	int numberLayers = (int)sizeLayers.Size() - 1;
	int outputLayer = numberLayers - 1;
	mlayers[outputLayer].floatDestinationOutputs = vOutputs.Data();
	FeedForward();
}

void OptimizedNeuralNetwork::OptimizedNeuralNetwork::FeedForward()
{
	int numberLayers = (int)sizeLayers.Size() - 1;
	mlayers.Resize(numberLayers);
	int outputLayer = numberLayers - 1;
	int inputsWithoutBias = sizeLayers[0];

	float * layerInputs = vInputs.Data();

	for (int l = 0; l < numberLayers; l++) {
		int neurons = sizeLayers[l + 1];
		assert(neurons > 0);
		bool isOutputLayer = (l == outputLayer) ? true : false;
		int nextLayerNeurons = (isOutputLayer) ? 0 : sizeLayers[l + 2];
		mlayers[l].UpdateInput(neurons, inputsWithoutBias + 1,
			nextLayerNeurons, initialLearningRate,
			layerInputs, isOutputLayer);
		layerInputs = mlayers[l].vOutputs.Data();
		inputsWithoutBias = neurons;
	}
}
HostMatrix<cudafloat> OptimizedNeuralNetwork::GetOutputs(HostMatrix<cudafloat> & inputs)
{
	int patterns = (int)inputs.Rows();
	int numberLayers = (int)this->mlayers.Size();
	DeviceMatrix<cudafloat> d_inputs(inputs);

	CpuArray< DeviceMatrix<cudafloat>*> layerOutputs;
	layerOutputs.Resize(numberLayers);
	
	for (int l = 0; l < numberLayers; l++) {
		layerOutputs[l] = new DeviceMatrix<cudafloat>(patterns, mlayers[l].numberNeurons);
	}
	cudafloat * layerInputs = d_inputs.Data();
	int ninputs = (int)d_inputs.Columns();

	for (int l = 0; l < numberLayers; l++) {
		
		if (mlayers[l].numberConnection < CUDA_MAX_THREADS_PER_BLOCK)
		{
			int processed = 0;
			do {
				int size = (patterns > CUDA_MAX_GRID_X_DIM) ?
					CUDA_MAX_GRID_X_DIM : patterns;

				cuda_activate_layer << <size, this->mlayers[l].dimInputsNeurons, this->mlayers[l].sharedMemFire,
					this->streamKernels >> > (
						layerInputs,
						this->mlayers[l].vWeights.Data(),
						0,
						layerOutputs[l]->Data()
						);
				processed += size;
			} while (processed < patterns);	
		}
		else
		{
			dim3 dimNeuronsPatterns;
			dimNeuronsPatterns.x = mlayers[l].numberNeurons;
			int processed = 0;
			do {
				int patternsToProcess = (patterns > CUDA_MAX_GRID_Y_DIM) ? CUDA_MAX_GRID_Y_DIM : patterns;
				dimNeuronsPatterns.y = patternsToProcess;

				cuda_activate_layerWrapper(streamKernels, dimNeuronsPatterns, mlayers[l].inputsBlockSize,
					layerInputs,
					this->mlayers[l].vWeights.Data(),
					0,
					layerOutputs[l]->Data(),
					this->mlayers[l].numberInputNoBias);
				processed += patternsToProcess;
			} while (processed < patterns);
		}
		layerInputs = layerOutputs[l]->Data();

	}

	HostMatrix<cudafloat> outputs(*(layerOutputs[numberLayers - 1]));
	for (int l = 0; l < numberLayers; l++) {
		delete layerOutputs[l];
	}
	return outputs;
}