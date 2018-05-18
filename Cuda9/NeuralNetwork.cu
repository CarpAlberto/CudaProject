#include "NeuralNetwork.h"
#include "MatrixFactory.h"
using namespace gpuNN;


NeuralNetwork::NeuralNetwork(Topology& topology,
	double bias,
	double learningRate,
	double momentum,
	TransferFunctionType transferType, FunctionErrorType errorType)
{
	this->m_topology		 = topology;
	size_t size				 = topology.size();
	this->m_bias			 = bias;
	this->m_learningRate	 = learningRate;
	this->m_momentum		 = momentum;

	this->m_TransferFunction = FunctionFactory::instance()->getTransferFunction(transferType);
	this->m_ErrorFunction	 = FunctionFactory::instance()->getErrorFunction(errorType);

	for (auto i = 0; i < size; i++) {
		/*Add a new layer*/
		#ifndef OPTIMIZED_GPU
			auto layer = std::make_shared<NetworkLayer>(topology[i],this->m_TransferFunction);
		#else	
			auto layer = std::make_shared<OptimizedGpuNetworkLayer>(topology[i], this->m_TransferFunction);
		#endif
		this->m_layers.push_back(std::move(layer));
	}

	for (auto i = 0; i < topology.size() - 1; i++) {
		auto matrix = MatrixFactory::getMatrix(topology[i], topology[i + 1]);
		matrix->SetRandom();
		this->m_weights.push_back(std::move(matrix));
	}
}

NeuralNetwork::~NeuralNetwork()
{
}

PtrMatrix NeuralNetwork::getNeuronAsMatrix(size_t index) const 
{
	#ifndef OPTIMIZED_GPU
		return this->m_layers[index].get()->toMatrix();
	#else
		return nullptr;
	#endif
}

cudaObject* NeuralNetwork::getNeuronAsData(size_t index)const
{
	return this->m_layers[index].get()->getData();
}

PtrMatrix NeuralNetwork::getNeuronActivatedValueAsMatrix(size_t index) const {
	
	#ifndef OPTIMIZED_GPU
		return this->m_layers[index].get()->toMatrixActivated();
	#else
		return nullptr;
	#endif;
}

PtrMatrix NeuralNetwork::getNeuronDerivedValueAsMatrix(size_t index) const {
	#ifndef OPTIMIZED_GPU
		return this->m_layers[index].get()->toMatrixDerived();
	#else
		return nullptr;
	#endif	
}

PtrMatrix NeuralNetwork::getWeightsMatrix(size_t index) const {
	return this->m_weights[index];
}

void NeuralNetwork::setNeuronValue(size_t indexLayer, size_t indexNeuron, double value) {
	#ifndef OPTIMIZED_GPU
		this->m_layers[indexLayer]->SetValue(indexNeuron, value);
	#endif	
}

void NeuralNetwork::FeedForward() {
	
	for (auto i = 0; i < this->m_layers.size() - 1; ++i) 
	{

			GenericMatrix* weightsMatrix = this->getWeightsMatrix(i);
			GenericMatrix* neuronMatrix = nullptr;
		
			#ifndef OPTIMIZED_GPU
				neuronMatrix = this->getNeuronAsMatrix(i);
				if (i != 0) {
					neuronMatrix = this->getNeuronActivatedValueAsMatrix(i);
				}
			#else

			auto cudaObject			 = this->getNeuronAsData(i);
			auto cudaObjectNextLayer = this->getNeuronAsData(i+1);
				
			if (i > 0){
				neuronMatrix = MatrixFactory::getMatrix(cudaObject, 1, cudaObject->Size(), 
					NeuronTypeData::ACTIVATED_DATA);
			}
			else{
				neuronMatrix = MatrixFactory::getMatrix(cudaObject, 1, cudaObject->Size(),
					NeuronTypeData::DATA);
			}
			
			#endif	
			//neuronMatrix->Print(ApplicationContext::instance()->getGUI().get());
			
			/*Perform the multiplication*/
			GenericMatrix& multipliedMatrix = (*neuronMatrix) * (*weightsMatrix);
			multipliedMatrix.Print(ApplicationContext::instance()->getGUI().get());
			
			#ifdef OPTIMIZED_GPU
				auto status = cudaMemcpy(cudaObjectNextLayer->Data(), multipliedMatrix.getData(),
					multipliedMatrix.getLength() * sizeof(float)
					, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
				cudaObjectNextLayer->Derive();
				cudaObjectNextLayer->Activate();

				//cudaObject->Print(ApplicationContext::instance()->getGUI().get());
				//cudaObjectNextLayer->Print(ApplicationContext::instance()->getGUI().get());

			#else		
				for (auto index = 0; index < multipliedMatrix.getCols(); index++) {
					this->setNeuronValue(i + 1, index, multipliedMatrix.Get(0, index));
				}
				delete neuronMatrix;
			#endif	
					
			delete &multipliedMatrix;
	}   
}

void NeuralNetwork::SetCurrentInput(const vDouble& input) {
	this->m_input = input;

	#ifdef OPTIMIZED_GPU
		double *d = (double*)&input[0];
		float * data = (float*)malloc(input.size() * sizeof(float));
		for (size_t i = 0; i < input.size();i++) {
			data[i] = (float)d[i];
		}
		auto cudaObj = this->m_layers[0].get()->getData();
		cudaMemcpy(cudaObj->Data(), data, input.size() * sizeof(float)
			, cudaMemcpyKind::cudaMemcpyHostToDevice);

		cudaObj->SetSize(input.size() * sizeof(float));
		cudaObj->Activate();
		cudaObj->Derive();

		cudaObj->Print(ApplicationContext::instance()->getGUI().get());
	
	#else	
		for (auto i = 0; i < input.size(); i++) {
			this->m_layers[0]->SetValue(i, input[i]);
		}
	#endif
}

void NeuralNetwork::Print( UIInterface* rhs) const
{
	for (int i = 0; i < this->m_layers.size(); i++) {
		rhs->showMessage("Layer : " + i);
		if (i == 0) 
		{
			#ifndef OPTIMIZED_GPU
				auto m = this->m_layers[i].get()->toMatrix();
			#else
				auto m = this->m_layers[i].get()->getData();
			#endif		

			m->Print(rhs);
		}
		else 
		{
			#ifndef OPTIMIZED_GPU
				auto m = this->m_layers[i].get()->toMatrixActivated();
			#else
				auto m = this->m_layers[i].get()->getData();
			#endif
			m->Print(rhs);
		}
		if (i < this->m_layers.size() - 1) {
			rhs->showMessage("Weight Matrix : " + i);
			this->getWeightsMatrix(i)->Print(rhs);
		}
		rhs->showMessage("=======================");
	} 
	rhs->showMessage("Total Error : ");
	rhs->Show(this->m_error);

}

double NeuralNetwork::getTotalError() const {
	return this->m_error;
}

vDouble NeuralNetwork::getTotalErrors() const {
	return this->m_errors;
}

void NeuralNetwork::setErrors() {

	if (this->m_target.size() == 0) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::CUDA_ERROR>
			("No target for this neural network");
		throw new std::exception("No target for this neural network");
	}
	size_t outputLayerIndex = this->m_layers.size() - 1;
	#ifdef OPTIMIZED_GPU
		size_t outputSize = this->m_layers[outputLayerIndex].get()->getData()->Size();
	#else
		auto outputSize = this->m_layers[outputLayerIndex]->Size();
	#endif
	
	if (this->m_target.size() != outputSize) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::CUDA_ERROR>
			("Target size is not the sime as layer size");
		throw new std::exception("Target size is not the sime as layer size");
	}
	this->m_error = 0.0;
	this->m_errors.resize(outputSize);

#ifdef OPTIMIZED_GPU
	auto cudaObj = (*this->m_layers[outputLayerIndex]).getData();
	float *hostData = cudaObj->HostDataActivated();
#endif

	for (auto i = 0; i < this->m_target.size(); i++) {
		/*The cost function as difference*/
		double t = this->m_target[i];
		#ifdef OPTIMIZED_GPU
			double y = hostData[i];
		#else
			double y = (*this->m_layers[outputLayerIndex])[i]->getActivatedValue();
		#endif
		this->m_errors[i] = y - t; 
		this->m_error += this->m_ErrorFunction->getError(y, t);
	}
#ifdef OPTIMIZED_GPU
	free(hostData);
#endif
	this->m_error = this->m_error;
	this->m_historicalErrors.push_back(this->m_error);
}

void NeuralNetwork::SetCurrentTarget(const vDouble& target) {
	this->m_target = target;
}

void NeuralNetwork::BackPropagation() {

#ifndef OPTIMIZED_GPU

	std::vector<GenericMatrix*> newWeights;

	auto outputLayerIndex = this->m_layers.size() - 1;

	/*Derived value from output to hidden*/
	auto derived = this->m_layers[outputLayerIndex]->toMatrixDerived();

	/*The gradients will be stored*/
	GenericMatrix* gradients = MatrixFactory::getMatrix(1, this->m_layers[outputLayerIndex]->Size());

	size_t i = 0;
	for (auto error : this->m_errors) {

		/*Get the derived value*/
		auto value = derived->Get(0, i);
		auto product = value * error;
		gradients->Set((size_t)0, i, product);
		i++;
	}
	auto lastHiddenLayerIndex = outputLayerIndex - 1;
	auto weightsOutputToHidden = this->m_weights[outputLayerIndex - 1];

	GenericMatrix& gradientsTransposed = gradients->Transpose();
	GenericMatrix& lastIndexLayerActivated = *this->m_layers[lastHiddenLayerIndex]->toMatrixActivated();

	GenericMatrix& deltaOutputToHiddenBefore = (gradientsTransposed * lastIndexLayerActivated);
	GenericMatrix& deltaOutputToHidden = deltaOutputToHiddenBefore.Transpose();

	GenericMatrix* newWeightsOutputToHidden = MatrixFactory::getMatrix(deltaOutputToHidden.getRows(),
		deltaOutputToHidden.getCols());

	for (auto i = 0; i < deltaOutputToHidden.getRows(); i++) {
		for (auto j = 0; j < deltaOutputToHidden.getCols(); j++) {
			auto originalWeight = weightsOutputToHidden->Get(i, j);
			auto deltaWeight = deltaOutputToHidden.Get(i, j);

			originalWeight = (float)(this->m_momentum * originalWeight);
			deltaWeight = (float)(this->m_learningRate * deltaWeight);
			newWeightsOutputToHidden->Set(i, j, (originalWeight - deltaWeight));
		}
	}
	newWeights.push_back(newWeightsOutputToHidden);

	//copy the gradients
	GenericMatrix* gradient = MatrixFactory::getMatrix(*gradients);

	/*Loop from (output,input] */
	for (size_t i = outputLayerIndex - 1; i > 0; i--) {

		/*Compute the delta weights*/
		GenericMatrix* activatedHidden = this->m_layers[i].get()->toMatrixActivated();
		GenericMatrix* derivedGradients = MatrixFactory::getMatrix(1, this->m_layers[i]->Size());

		GenericMatrix* weightMatrix = this->m_weights[i];
		GenericMatrix* originalWeights = this->m_weights[i - 1];

		for (auto r = 0; r < weightMatrix->getRows(); r++) {
			double sum = 0;
			for (auto c = 0; c < weightMatrix->getCols(); c++) {
				double product = gradient->Get(0, c) * weightMatrix->Get(r, c);
				sum += product;
			}
			float g = (float)(sum * activatedHidden->Get(0, r));
			derivedGradients->Set(0, r, g);
		}
		GenericMatrix * leftNeurons = nullptr;
		if (i - 1 == 0) {
			leftNeurons = this->m_layers[0]->toMatrix();
		}
		else {
			leftNeurons = this->m_layers[i - 1]->toMatrixActivated();
		}
		GenericMatrix& deriveGradientsTranspose = (*derivedGradients).Transpose();
		GenericMatrix& deltaWeightsNotTransposed = (deriveGradientsTranspose * (*leftNeurons));
		GenericMatrix& deltaWeights = deltaWeightsNotTransposed.Transpose();
		GenericMatrix* newWeightsHidden = MatrixFactory::getMatrix(deltaWeights.getRows(), deltaWeights.getCols());

		for (auto r = 0; r < newWeightsHidden->getRows(); r++) {
			for (auto c = 0; c < newWeightsHidden->getCols(); c++) {
				double w = originalWeights->Get(r, c);
				double d = deltaWeights.Get(r, c);
				auto error = (float)(w - d);
				newWeightsHidden->Set(r, c, error);
			}
		}
		newWeights.push_back(newWeightsHidden);

		delete gradient;
		gradient = MatrixFactory::getMatrix(*derivedGradients);
		delete leftNeurons;
		delete derivedGradients;
		delete &deriveGradientsTranspose;
		delete &deltaWeights;
		delete &deltaWeightsNotTransposed;
	}

	for (int i = 0; i < this->m_weights.size(); i++) {
		delete this->m_weights[i];
	}
	this->m_weights.clear();
	std::reverse(newWeights.begin(), newWeights.end());
	for (int i = 0; i < newWeights.size(); i++) {
		this->m_weights.push_back(MatrixFactory::getMatrix(*newWeights[i]));

	}
	delete &gradientsTransposed;
	delete &lastIndexLayerActivated;
	delete &deltaOutputToHiddenBefore;
	delete &deltaOutputToHidden;
	delete gradients;
	delete derived;

	newWeights.clear();

#endif
}

void NeuralNetwork::SetCurrentInput(const RealHostMatrix& input)
{
	assert(0);
}

void NeuralNetwork::PrintOutput() 
{
#ifndef OPTIMIZED_GPU
	size_t indexOfOutputLayer = this->m_layers.size() - 1;
	GenericMatrix *outputValues = this->m_layers.at(indexOfOutputLayer)->toMatrixActivated();
	for (int c = 0; c < outputValues->getCols(); c++) {
		std::cout << outputValues->Get(0, c) << "\t";
	}
	delete outputValues;
	std::cout << std::endl;

#endif
}

void NeuralNetwork::PrintTarget() {
	for (int c = 0; c < this->m_target.size(); c++) {
		std::cout << this->m_target[c]<< "\t";
	}
	std::cout << std::endl;
}

void NeuralNetwork::Train(int noEpock) {

	for (auto i = 0; i < noEpock; i++) {
		this->FeedForward();
		this->setErrors();
		this->BackPropagation();
		std::cout << "Epock " << i << std::endl;
	}
	PrintOutput();
}

void NeuralNetwork::Save(const std::string& filename, IOStrategy strategy) {

	if (strategy != IOStrategy::ASCII) {
		throw new std::exception("Unsuported strategy save");
	}
	std::ofstream os(filename,std::ios_base::out);
	size_t index = 0;
	for (auto& t : this->m_topology) {
		os << t << " ";
	}
	os << std::endl;
	os << this->m_bias << " " << this->m_learningRate << " " << this->m_momentum << std::endl;
	os << this->m_weights.size() << std::endl;
	for (auto& weight : m_weights) {
		os << weight->getRows() << " " << weight->getCols() << " " <<std::endl;
		for (auto& vec : weight->getAsMatrix()) {
			for (auto& el : vec) {
				os << el << " ";
			}
			os << std::endl;
		}
	}
	os.close();
}

