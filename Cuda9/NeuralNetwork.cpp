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
		auto layer = std::make_shared<NetworkLayer>(topology[i],this->m_TransferFunction);
		this->m_layers.push_back(layer);
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

PtrMatrix NeuralNetwork::getNeuronAsMatrix(size_t index) const {
	return this->m_layers[index].get()->toMatrix();
}

PtrMatrix NeuralNetwork::getNeuronActivatedValueAsMatrix(size_t index) const {
	return this->m_layers[index].get()->toMatrixActivated();
}

PtrMatrix NeuralNetwork::getNeuronDerivedValueAsMatrix(size_t index) const {
	return this->m_layers[index].get()->toMatrixDerived();
}

PtrMatrix NeuralNetwork::getWeightsMatrix(size_t index) const {
	return this->m_weights[index];
}

void NeuralNetwork::setNeuronValue(size_t indexLayer, size_t indexNeuron, double value) {
	this->m_layers[indexLayer]->SetValue(indexNeuron, value);
}

void NeuralNetwork::FeedForward() {
	
	for (auto i = 0; i < this->m_layers.size() - 1; ++i) {
		GenericMatrix* neuronMatrix = this->getNeuronAsMatrix(i);
		GenericMatrix* weightsMatrix = this->getWeightsMatrix(i);
		if (i != 0) {
			neuronMatrix = this->getNeuronActivatedValueAsMatrix(i);
		}
	  GenericMatrix& multipliedMatrix = (*neuronMatrix) * (*weightsMatrix);
	  for (auto index = 0; index < multipliedMatrix.getCols(); index++) {
			this->setNeuronValue(i + 1, index, multipliedMatrix.Get(0, index, 0));
	  }

	  delete neuronMatrix;
	  delete &multipliedMatrix;
	}   
}

void NeuralNetwork::SetCurrentInput(const vDouble& input) {
	this->m_input = input;

	for (auto i = 0; i < input.size(); i++) {
		this->m_layers[0]->SetValue(i, input[i]);
	}
}

void NeuralNetwork::Print( UIInterface* rhs) const
{
	for (int i = 0; i < this->m_layers.size(); i++) {
		rhs->showMessage("Layer : " + i);
		if (i == 0) {
			auto m = this->m_layers[i].get()->toMatrix();
			m->Print(rhs);
		}
		else {
			auto m = this->m_layers[i].get()->toMatrixActivated();
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
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("No target for this neural network");
		throw new std::exception("No target for this neural network");
	}
	if (this->m_target.size() != this->m_layers[this->m_layers.size() - 1].get()->Size()) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Target size is not the sime as layer size");
		throw new std::exception("Target size is not the sime as layer size");
	}
	this->m_error = 0.0;
	int outputLayerIndex = this->m_layers.size() - 1;
	auto size = this->m_layers[outputLayerIndex]->Size();
	this->m_errors.resize(size);
	for (auto i = 0; i < this->m_target.size(); i++) {
		/*The cost function as difference*/
		double t = this->m_target[i];
		double y = (*this->m_layers[outputLayerIndex])[i]->getActivatedValue();
		this->m_errors[i] = y - t; 
		this->m_error += this->m_ErrorFunction->getError(y, t);
	}
	this->m_error = this->m_error;
	this->m_historicalErrors.push_back(this->m_error);
}

void NeuralNetwork::SetCurrentTarget(const vDouble& target) {
	this->m_target = target;
}

void NeuralNetwork::BackPropagation() {

	std::vector<GenericMatrix*> newWeights;

	auto outputLayerIndex = this->m_layers.size() - 1;
	
	/*Derived value from output to hidden*/
	auto derived = this->m_layers[outputLayerIndex]->toMatrixDerived();

	/*The gradients will be stored*/
	GenericMatrix* gradients = new CpuMatrix(1, this->m_layers[outputLayerIndex]->Size(), 1);


	for (auto i = 0; i < this->m_errors.size(); i++) {		
		/*Get the derived value*/
		auto value = derived->Get(0, i, 0);
		auto error = this->m_errors[i];
		auto product = value * error;
		gradients->Set(0, i, 0, product);

	}
	auto lastHiddenLayerIndex  = outputLayerIndex - 1;
	auto weightsOutputToHidden = this->m_weights[outputLayerIndex - 1];

	GenericMatrix& gradientsTransposed		 = gradients->Transpose();
	GenericMatrix& lastIndexLayerActivated   = *this->m_layers[lastHiddenLayerIndex]->toMatrixActivated();
	GenericMatrix& deltaOutputToHiddenBefore = (gradientsTransposed * lastIndexLayerActivated);
	GenericMatrix& deltaOutputToHidden		 = deltaOutputToHiddenBefore.Transpose();
	GenericMatrix* newWeightsOutputToHidden  = new CpuMatrix(deltaOutputToHidden.getRows(),
										deltaOutputToHidden.getCols(), 1);

	for (auto i = 0; i < deltaOutputToHidden.getRows(); i++) {
		for (auto j = 0; j < deltaOutputToHidden.getCols(); j++) {
			auto originalWeight = weightsOutputToHidden->Get(i, j, 0);
			auto deltaWeight    =   deltaOutputToHidden.Get(i, j, 0);

			originalWeight = this->m_momentum * originalWeight;
			deltaWeight = this->m_learningRate * deltaWeight;
			newWeightsOutputToHidden->Set(i, j, 0,(originalWeight - deltaWeight));
		}
	}
	newWeights.push_back(newWeightsOutputToHidden);

	//copy the gradients
	GenericMatrix* gradient = new CpuMatrix(*gradients);

	/*Loop from (output,input] */
	for (size_t i = outputLayerIndex - 1; i > 0; i--) {
		/*Compute the delta weights*/
		GenericMatrix* activatedHidden	= this->m_layers[i].get()->toMatrixActivated();
		GenericMatrix* derivedGradients = new CpuMatrix(1, this->m_layers[i]->Size(), 1);
		GenericMatrix* weightMatrix = this->m_weights[i];
		GenericMatrix* originalWeights = this->m_weights[i - 1];

		for (auto r = 0; r < weightMatrix->getRows(); r++) {
			double sum = 0;
			for (auto c = 0; c < weightMatrix->getCols(); c++) {
				double product = gradient->Get(0,c, 0) * weightMatrix->Get(r, c, 0);
				sum += product;
			}
			float g = sum * activatedHidden->Get(0, r, 0);
			derivedGradients->Set(0, r, 0, g);
		}
		GenericMatrix * leftNeurons = nullptr;
		if (i - 1 == 0) {
			leftNeurons = this->m_layers[0]->toMatrix();
		}
		else{
			leftNeurons = this->m_layers[i - 1]->toMatrixActivated();
		}
		GenericMatrix& deriveGradientsTranspose = (*derivedGradients).Transpose();
		GenericMatrix& deltaWeightsNotTransposed = (deriveGradientsTranspose * (*leftNeurons));
		GenericMatrix& deltaWeights = deltaWeightsNotTransposed.Transpose();
		GenericMatrix* newWeightsHidden = new CpuMatrix(deltaWeights.getRows(),deltaWeights.getCols(),1);

		for (auto r = 0; r < newWeightsHidden->getRows(); r++) {
			for (auto c = 0; c < newWeightsHidden->getCols(); c++) {
				double w = originalWeights->Get(r, c, 0);
				double d = deltaWeights.Get(r, c, 0);
				double error = w - d;
				newWeightsHidden->Set(r, c, 0, error);
			}
		}
		newWeights.push_back(newWeightsHidden);
		delete gradient;
		gradient = new CpuMatrix(*derivedGradients);
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
		this->m_weights.push_back(new CpuMatrix(*newWeights[i]));

	}
	delete &gradientsTransposed;
	delete &lastIndexLayerActivated;
	delete &deltaOutputToHiddenBefore;
	delete &deltaOutputToHidden;
	delete gradients;
	delete derived;

	newWeights.clear();
}

void NeuralNetwork::PrintOutput() {
	int indexOfOutputLayer = this->m_layers.size() - 1;

	GenericMatrix *outputValues = this->m_layers.at(indexOfOutputLayer)->toMatrixActivated();
	for (int c = 0; c < outputValues->getCols(); c++) {
		std::cout << outputValues->Get(0, c, 0) << "\t";
	}
	delete outputValues;
	std::cout << std::endl;
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
	}
}

void NeuralNetwork::Save(const std::string& filename,IOStrategy strategy) {
	
	
	//json j = {};
	 std::vector<mDouble> weightSet; 
	for (int i = 0; i < this->m_weights.size(); i++) {

		auto ptrMatrix = this->m_weights[i];
		weightSet.push_back(ptrMatrix->getAsMatrix());

	}
	//j["weights"] = weightSet;
	//std::ofstream o(filename);
	//o << std::setw(4) << j << std::endl;* /
}