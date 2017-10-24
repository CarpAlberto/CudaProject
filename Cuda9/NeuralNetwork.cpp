#include "NeuralNetwork.h"
using namespace gpuNN;


NeuralNetwork::NeuralNetwork(Topology& topology, TransferFunctionType transferFunction)
{
	this->m_topology = topology;
	auto size = topology.size();
	for (auto i = 0; i < size; i++) {
		/*Add a new layer*/
		auto layer = std::make_shared<NetworkLayer>(topology[i]);
		this->m_layers.push_back(layer);
	}
	for (auto i = 0; i < topology.size() - 1; i++) {
		auto matrix = new CpuMatrix(topology[i], topology[i + 1], 1);
		matrix->SetRandom();
		this->m_weights.push_back(matrix);
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

void NeuralNetwork::feedForward() {
	
	
	for (auto i = 0; i < this->m_layers.size() - 1; ++i) {
		auto neuronMatrix = this->getNeuronAsMatrix(i);
		auto weightsMatrix = this->getWeightsMatrix(i);
		if (i != 0) {
			neuronMatrix = this->getNeuronActivatedValueAsMatrix(i);
		}
	  CpuMatrix multipliedMatrix = (*neuronMatrix) * (*weightsMatrix);
	//  neuronMatrix->Print();
	//  weightsMatrix->Print();
	 // multipliedMatrix.Print();
	  for (auto index = 0; index < multipliedMatrix.getCols(); index++) 
	  {
			this->setNeuronValue(i + 1, index, multipliedMatrix.Get(0, index, 0));
	  }
	}
			
}

void NeuralNetwork::SetCurrentInput(const vDouble& input) {
	this->m_input = input;

	for (auto i = 0; i < input.size(); i++) {
		this->m_layers[0]->SetValue(i, input[i]);
	}
}

void NeuralNetwork::Print() {
	for (int i = 0; i < this->m_layers.size(); i++) {
		std::cout << "Layer:" << i << std::endl;
		if (i == 0) {
			auto m = this->m_layers[i].get()->toMatrix();
			m->Print();
		}
		else {
			auto m = this->m_layers[i].get()->toMatrixActivated();
			m->Print();
		}
		if (i < this->m_layers.size() - 1) {
			std::cout << "Weight Matrix" << i << std::endl;
			this->getWeightsMatrix(i)->Print();
		}
		std::cout << "================" << std::endl;
	} 
	std::cout << "Totral Error:" << this->m_error;
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
		auto temporaryError = (*this->m_layers[outputLayerIndex])[i]->getActivatedValue() -
			this->m_target[i];
		this->m_errors[i] = temporaryError;
		this->m_error += temporaryError;
	}
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

	//std::cout << "DERIVED :" << std::endl;
	//(derived)->Print();

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

	GenericMatrix& gradientsTransposed		= gradients->Transpose();
	GenericMatrix& lastIndexLayerActivated  = *this->m_layers[lastHiddenLayerIndex]->toMatrixActivated();
	GenericMatrix& deltaOutputToHidden		= (gradientsTransposed * lastIndexLayerActivated).Transpose();


	delete &gradientsTransposed;
	delete &lastIndexLayerActivated;

	auto newWeightsOutputToHidden = new CpuMatrix(deltaOutputToHidden.getRows(), 
										deltaOutputToHidden.getCols(), 1);

	for (auto i = 0; i < deltaOutputToHidden.getRows(); i++) {
		for (auto j = 0; j < deltaOutputToHidden.getCols(); j++) {
			auto originalWeight =	weightsOutputToHidden->Get(i, j, 0);
			auto deltaWeight = deltaOutputToHidden.Get(i, j, 0);
			newWeightsOutputToHidden->Set(i, j, 0,originalWeight - deltaWeight);

		}
	}
	newWeights.push_back(newWeightsOutputToHidden);

	//copy the gradients
	GenericMatrix* gradient = new CpuMatrix(*(gradients));

	/*Loop from (output,input] */
	for (auto i = outputLayerIndex - 1; i > 0; i--) {
		/*Compute the delta weights*/
		GenericMatrix* activatedHidden	= this->m_layers[i].get()->toMatrixActivated();
		GenericMatrix* derivedGradients = new CpuMatrix(1, this->m_layers[i]->Size(), 1);
		auto weightMatrix = this->m_weights[i];
		auto originalWeights = this->m_weights[i - 1];

		for (auto r = 0; r < weightMatrix->getRows(); r++) {
			double sum = 0;
			for (auto c = 0; c < weightMatrix->getCols(); c++) {
				double product = gradient->Get(0,c, 0) * weightMatrix->Get(r, c, 0);
				sum += product;
			}
			float g = sum * activatedHidden->Get(0, r, 0);
			derivedGradients->Set(0, r, 0, g);
		}
//		std::cout << "DERIVED GRADIENTS:" << std::endl;
//		(derivedGradients)->Print();
		GenericMatrix * leftNeurons = nullptr;
		if (i - 1 == 0) {
			leftNeurons = this->m_layers[0]->toMatrix();
		}
		else{
			leftNeurons = this->m_layers[i - 1]->toMatrixActivated();
		}

	//	std::cout << "Left neurons transposed:" << std::endl;
	//	leftNeurons->Transpose().Print();
		GenericMatrix& deltaWeights = ((*derivedGradients).Transpose() * (*leftNeurons)).Transpose();

	//	std::cout << "delta weights:" << std::endl;
	//	(deltaWeights).Print();

		GenericMatrix* newWeightsHidden = new CpuMatrix(deltaWeights);

		for (auto r = 0; r < newWeightsHidden->getRows(); r++) {
			for (auto c = 0; c < newWeightsHidden->getCols(); c++) {
				double w = originalWeights->Get(r, c, 0);
				double d = deltaWeights.Get(r, c, 0);
				double error = w - d;
				newWeightsHidden->Set(r, c, 0, error);
			}
		}
		delete gradient;
		gradient = new CpuMatrix(*derivedGradients);
		newWeights.push_back(newWeightsHidden);
	}
	for (int i = 0; i < this->m_weights.size(); i++) {
		delete this->m_weights[i];
	}
	this->m_weights.clear();

	std::reverse(newWeights.begin(), newWeights.end());
	for (int i = 0; i < newWeights.size(); i++) {
		this->m_weights.push_back(new CpuMatrix(*newWeights[i]));

	}
	newWeights.clear();
}