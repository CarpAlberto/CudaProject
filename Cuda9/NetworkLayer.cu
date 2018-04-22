#include "NetworkLayer.h"
#include "ml.h"
using namespace gpuNN;

void NetworkLayer::Push(Neuron* neuron) {
	this->m_neurons.push_back(neuron);
}

Neuron* NetworkLayer::operator[](size_t index) const {
	return this->m_neurons[index];
}

size_t NetworkLayer::Size() const {
	return this->m_neurons.size();
}

NetworkLayer::NetworkLayer(int numOutputs, TransferFunction* transfer) {

	//TODO may cause memory leaks
	for (int i = 0; i < numOutputs; i++) {
		this->m_neurons.push_back(new Neuron(numOutputs, transfer));
	}
}

void NetworkLayer::SetValue(size_t index, double value) {
	this->m_neurons[index]->SetOutputValue(value);
}

vDouble NetworkLayer::toVector() {
	vDouble returnVector;
	for (auto i = 0; i < this->m_neurons.size(); ++i) {
		auto activatedValue = this->m_neurons[i]->getActivatedValue();
		returnVector.push_back(activatedValue);
	}
	return returnVector;
}

PtrMatrix NetworkLayer::toMatrix() {

	auto m = MatrixFactory::getMatrix(1, this->m_neurons.size());
	for (size_t i = 0; i < this->m_neurons.size(); ++i) {
		float value = (float)(this->m_neurons[i]->getOutputValue());
		m->Set(0, i, value);
	}
	return m;
}

PtrMatrix NetworkLayer::toMatrixActivated() {
	auto m = MatrixFactory::getMatrix(1, this->m_neurons.size());
	for (size_t i = 0; i < this->m_neurons.size(); ++i) {
		auto value = (float)(this->m_neurons[i]->getActivatedValue());
		m->Set(0, i, value);
	}
	return m;
}

PtrMatrix NetworkLayer::toMatrixDerived() {
	auto m = MatrixFactory::getMatrix(1, this->m_neurons.size());
	for (auto i = 0; i < this->m_neurons.size(); ++i) {
		auto value = this->m_neurons[i]->getDerivedValue();
		m->Set(0, i, (float)value);
	}
	return m;
}

NetworkLayer::~NetworkLayer() {
	for (auto it : this->m_neurons) {
		delete it;
	}
}

OptimizedGpuNetworkLayer::OptimizedGpuNetworkLayer(int numOutputs, TransferFunction* transfer)
{
	this->transfer = transfer;
	auto data = (float*)Memory::instance()->allocate(numOutputs * sizeof(float), gpuNN::Bridge::GPU);
	cudaMemset(data, numOutputs, numOutputs * sizeof(float));
	this->obj = new cudaObject(data, numOutputs * sizeof(float));
}

cudaObject* OptimizedGpuNetworkLayer::getData()
{
	return this->obj;
}

void gpuNN::cudaObject::Activate()
{
	int num_blocks = 1;
	int block_size = 1;
	kTanh<< <num_blocks, block_size >> >(data_size, data, activatedData);
}

void gpuNN::cudaObject::Derive()
{
	int num_blocks = 1;
	int block_size = 1;
	kTanhDerivative << <num_blocks, block_size >> >(data_size, data, derivedData);
}

void gpuNN::cudaObject::Print(UIInterface* rhs)
{
	float* f = (float*)malloc(this->data_size);

	auto status = cudaMemcpy(f, this->data,
		this->data_size
		, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	float* activ = this->HostDataActivated();

	rhs->showMessage("Data  \n");
	for (auto i = 0; i < data_size / sizeof(float); i++) {
		rhs->Show(f[i]);
		rhs->showMessage("\n");
	}
	rhs->showMessage("Activated  \n");
	for (auto i = 0; i < data_size / sizeof(float); i++) {
		rhs->Show(activ[i]);
		rhs->showMessage(" ");
	}
	rhs->showMessage("\n");
}
size_t gpuNN::cudaObject::Size()
{
	return ( this->data_size/4 );
}

float* gpuNN::cudaObject::HostData()
{
	float* f = (float*)malloc(this->data_size);

	auto status = cudaMemcpy(f, this->data,this->data_size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	return f;
}
float* gpuNN::cudaObject::HostDataActivated()
{
	float* f = (float*)malloc(this->data_size);

	auto status = cudaMemcpy(f, this->activatedData, this->data_size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	return f;
}

float*  gpuNN::cudaObject::HostDataDerived()
{
	float* f = (float*)malloc(this->data_size);

	auto status = cudaMemcpy(f, this->derivedData, this->data_size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	return f;
}
cudaObject::cudaObject(cudafloat* data,size_t allSize)
{
	this->data = data;
	this->data_size = allSize;

	this->activatedData = (float*)Memory::instance()->allocate(allSize, gpuNN::Bridge::GPU);
	this->derivedData = (float*)Memory::instance()->allocate(allSize, gpuNN::Bridge::GPU);
}

float* cudaObject::Data()
{
	return this->data;
}
void   cudaObject::SetData(float* data,size_t dataSize)
{
	this->data = data;
	this->data_size = dataSize;

	Derive();
	Activate();
}

void cudaObject::SetSize(size_t size)
{
	this->data_size = size;
}

float* cudaObject::Activated()
{
	return this->activatedData;
}
float* cudaObject::Derived()
{
	return this->derivedData;
}
