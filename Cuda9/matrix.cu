#pragma once
#include "matrix.h"
#include "Memory.h"

using namespace gpuNN;

GenericMatrix::GenericMatrix() {
	m_cols = 0;
	m_rows = 0;
	m_data = nullptr;
	m_channels = 0;
}

GenericMatrix::GenericMatrix(const GenericMatrix& rhs) {
	Clone(rhs);
}

GenericMatrix::GenericMatrix(int height, int width, int channels) {
	this->m_cols = width;
	this->m_rows = height;
	this->m_channels = channels;
	this->m_data = nullptr;
	this->Malloc();
}

void GenericMatrix::Release() {
	this->Free();
	this->m_cols = 0;
	this->m_rows = 0;
	this->m_channels = 0;
	this->m_data = nullptr;
}

GenericMatrix& GenericMatrix::operator=(const GenericMatrix& rhs)
{
	this->Free();
	this->m_cols = rhs.m_cols;
	this->m_rows = rhs.m_rows;
	this->m_channels = rhs.m_channels;
	this->Malloc();
	this->Memcpy(const_cast<GenericMatrix&>(rhs));
	return *this;
}

void GenericMatrix::SetSize(int rows, int cols, int channels) {
	this->Free();
	this->m_cols = cols;
	this->m_rows = rows;
	this->m_channels = channels;
	Malloc();
	Zeros();
}

int GenericMatrix::getLength() const {
	return this->m_cols * this->m_rows * this->m_channels;
}

void GenericMatrix::Zeros() {
	SetAll(0.0);
}

void GenericMatrix::Ones() {
	SetAll(1.0);
}

float* GenericMatrix::getData() const {
	return this->m_data;
}

float  GenericMatrix::Get(int y, int x, int channel)const {
	if (this->m_data == nullptr) {
		if (x >= this->m_cols || y >= this->m_rows || channel >= this->m_channels) {
			ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
				("Invalid position");
			throw new std::exception("Invalid position");
		}
	}
	return this->m_data[RC2IDX(y,x, this->m_cols) + channel * (this->m_rows * this->m_cols)];
}


int GenericMatrix::getCols() const {
	return this->m_cols;
}

int GenericMatrix::getRows() const {
	return this->m_rows;
}

int GenericMatrix::getChannels() const {
	return this->m_channels;
}

void GenericMatrix::Print(UIInterface * guiInterface) const {
	
	for (auto i = 0; i < this->getRows(); i++) {
		for (auto j = 0; j < this->getCols(); j++) {
			double value = this->Get(i, j, 0);
			guiInterface->Show(value);
			guiInterface->showMessage(" ");
		}
		guiInterface->showMessage("\n");
	}
}

void GenericMatrix::SetRandom() {
	auto length = getLength();
	for (auto i = 0; i < length; i++) {
		this->m_data[i] = (float)Utils::generateRandom();
	}
}
// Cpu Matrix Implementation

CpuMatrix::CpuMatrix(int height, int width, int channels)  
	: GenericMatrix(height,width,channels) {

	this->Malloc();
}

void CpuMatrix::Set(int y, int x, int channel, float val) {
	if (this->m_data == nullptr) {
		Zeros();
	}
	if (y >= this->m_rows || x >= this->m_cols || channel >= this->m_channels) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid position");
		throw new std::exception("invalid");
	}
	this->m_data[RC2IDX(y,x,this->m_cols) + channel * (this->m_rows * this->m_cols)] = val;
}

void CpuMatrix::Set(int y, int x, const VectorFloat& rhs) {
	if (this->m_data == nullptr) {
		Zeros();
	}
	if (y >= this->m_rows || x >= this->m_cols) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid position");
		throw new std::exception("invalid");
	}
	for (int i = 0; i < this->m_channels; ++i) {
		Set(y, x, i, rhs.Get(i));
	}
}

void CpuMatrix::Set(int pos, int pos_channel, float val) {
	if (this->m_data == nullptr) {
		Zeros();
	}
	if (pos >= this->m_cols * this->m_rows) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid position");
		return;
	}
	this->m_data[pos + pos_channel * (this->m_rows * this->m_cols)] = val;
}

CpuMatrix::CpuMatrix() :
	GenericMatrix()
{

}

CpuMatrix::CpuMatrix(const GenericMatrix& rhs) :
	GenericMatrix(rhs)
{
	this->m_data = nullptr;
	this->Clone(rhs);
}

void CpuMatrix::Malloc() {

	if (this->m_data == nullptr) {
	
		try
		{
			// allocates a vector of two integers
			this->m_data = (float*)Memory::instance()->allocate(this->m_cols * this->m_rows * this->m_channels *
				sizeof(float), Bridge::CPU);
		}
		catch (MemoryAllocationException exception) {
			ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
				(exception.what());
		}
		catch (...) {
			ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
				("Unknown Error");
		}
		memset(this->m_data, 0 ,m_cols * m_rows * m_channels * sizeof(float));
	}
}

void CpuMatrix::Memcpy(GenericMatrix& rhs) {
	memcpy(this->m_data,rhs.getData(), getLength() * sizeof(float));
}

void CpuMatrix::Free() {
	if (this->m_data != nullptr) {
		Memory::instance()->deallocate(this->m_data, Bridge::CPU);
	}

}

GenericMatrix& CpuMatrix::operator+(const GenericMatrix& rhs) const {

	if (this->m_data == nullptr || rhs.getData() == nullptr || getLength() != rhs.getLength()) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid arguments");
		throw new std::exception("Invalid arguments");
	}
	int length = getLength();
	auto cpuMatrix = new CpuMatrix(rhs);
	for (int i = 0; i < length; ++i) {
		cpuMatrix->m_data[i] = rhs.getData()[i] + m_data[i];
	}
	return *cpuMatrix;
}

GenericMatrix& CpuMatrix::operator+(float val) const {

	if (this->m_data == nullptr) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid arguments");
		throw new std::exception("Invalid arguments");
	}
	int length = getLength();
	CpuMatrix* cpuMatrix = new CpuMatrix(this->m_rows,this->m_cols,this->m_channels);
	for (int i = 0; i < length; ++i) {
		cpuMatrix->m_data[i] =  m_data[i] + val;
	}
	return *cpuMatrix;
}

GenericMatrix& CpuMatrix::operator+(const VectorFloat& rhs) const {
	if (this->m_data == nullptr) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid arguments");
		throw new std::exception("Invalid arguments");
	}
	int n = this->m_cols * this->m_rows;
	CpuMatrix* cpuMatrix = new CpuMatrix(this->m_rows, this->m_cols, this->m_channels);
	for (auto ch = 0; ch < this->m_channels; ch++) {
		for (auto i = 0; i < n; ++i) {
			cpuMatrix->m_data[i + n * ch] = this->m_data[i + n * ch] + rhs.Get(ch);
		}
	}
	return *cpuMatrix;
}

GenericMatrix& CpuMatrix::operator-(const GenericMatrix& rhs) const {

	if (this->m_data == nullptr || rhs.getData() == nullptr || getLength() != rhs.getLength()) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid arguments");
		throw new std::exception("Invalid arguments");
	}
	int length = getLength();
	CpuMatrix* cpuMatrix = new CpuMatrix(rhs);
	for (int i = 0; i < length; ++i) {
		cpuMatrix->m_data[i] = rhs.getData()[i] - m_data[i];
	}
	return *cpuMatrix;
}

GenericMatrix& CpuMatrix::operator-(float val) const {

	if (this->m_data == nullptr) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid arguments");
		throw new std::exception("Invalid arguments");
	}
	int length = getLength();
	CpuMatrix* cpuMatrix = new CpuMatrix(this->m_rows, this->m_cols, this->m_channels);
	for (int i = 0; i < length; ++i) {
		cpuMatrix->m_data[i] = m_data[i] - val;
	}
	return *cpuMatrix;
}

GenericMatrix& CpuMatrix::operator-(const VectorFloat& rhs) const {
	if (this->m_data == nullptr) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid arguments");
		throw new std::exception("Invalid arguments");
	}
	int n = this->m_cols * this->m_rows;
	CpuMatrix* cpuMatrix = new CpuMatrix(this->m_rows, this->m_cols, this->m_channels);
	for (auto ch = 0; ch < this->m_channels; ch++) {
		for (auto i = 0; i < n; ++i) {
			cpuMatrix->m_data[i + n * ch] = this->m_data[i + n * ch] - rhs.Get(ch);
		}
	}
	return *cpuMatrix;
}

GenericMatrix& CpuMatrix::operator*(const GenericMatrix& rhs) const {

	if (this->m_cols != rhs.getRows()) {
		throw new std::exception("Invalid arguments");
	}
	CpuMatrix* cpuMatrix = new CpuMatrix(this->getRows(),rhs.getCols(),1);
	for (int i = 0; i < this->getRows(); i++) {
		for (auto j = 0; j < rhs.getCols(); j++) {
			for (auto k = 0; k < rhs.getRows(); k++) {
				float p = this->Get(i, k, 0) * rhs.Get(k,j,0);
				float newValue = cpuMatrix->Get(i, j, 0) + p;
				cpuMatrix->Set(i, j, 0, newValue);
			}
			cpuMatrix->Set(i, j, 0,cpuMatrix->Get(i, j, 0));
		}
		
	}
	return *cpuMatrix;
}

GenericMatrix& CpuMatrix::operator*(float val) const {

	if (this->m_data == nullptr) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid arguments");
		throw new std::exception("Invalid arguments");
	}
	int length = getLength();
	CpuMatrix* cpuMatrix = new CpuMatrix(this->m_rows, this->m_cols, this->m_channels);
	for (int i = 0; i < length; ++i) {
		cpuMatrix->m_data[i] = m_data[i] * val;
	}
	return *cpuMatrix;
}

GenericMatrix& CpuMatrix::operator*(const VectorFloat& rhs) const {
	if (this->m_data == nullptr) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid arguments");
		throw new std::exception("Invalid arguments");
	}
	int n = this->m_cols * this->m_rows;
	CpuMatrix* cpuMatrix = new CpuMatrix(this->m_rows, this->m_cols, this->m_channels);
	for (auto ch = 0; ch < this->m_channels; ch++) {
		for (auto i = 0; i < n; ++i) {
			cpuMatrix->m_data[i + n * ch] = this->m_data[i + n * ch] * rhs.Get(ch);
		}
	}
	return *cpuMatrix;
}

void	CpuMatrix::SetAll(float val) {
	if (this->m_data == nullptr) {
		Malloc();
	}
	int length = getLength();
	for (auto iterator = 0; iterator < length; iterator++) {
		this->m_data[iterator] = val;
	}
}

void	CpuMatrix::SetAll(const VectorFloat& rhs)
{
	if (this->m_data == nullptr) {
		Malloc();
	}
	int length = getLength();
	for (auto ch = 0; ch < this->m_channels; ch++) {
		for (auto i = 0; i < length; ++i) {
			this->m_data[i + length * ch] = this->m_data[i + length * ch] * rhs.Get(ch);
		}
	}
}

void CpuMatrix::Clone(const GenericMatrix& rhs) {
	this->m_cols = rhs.getCols();
	this->m_rows = rhs.getRows();
	this->m_channels = rhs.getChannels();
	this->Malloc();
	this->Memcpy(const_cast<GenericMatrix&>(rhs));
}

GenericMatrix& CpuMatrix::Transpose() const {
	
	auto rows = this->getRows();
	auto columns = this->getCols();

	CpuMatrix * matrix = new CpuMatrix(columns,rows,1);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
   			matrix->Set(j, i, 0, this->Get(i,j,0));
		}
	}
	return *matrix;
}

CpuMatrix::~CpuMatrix() {
	if(this->m_data != nullptr)
		Memory::instance()->deallocate(this->m_data, Bridge::CPU);
}

mDouble CpuMatrix::getAsMatrix() {
	
	mDouble matrix;
	for (int i = 0; i < this->m_rows; ++i) {
		vDouble v;
		for (int j = 0; j < this->m_cols; ++j) {
			double value = this->Get(i, j, 0);
			v.push_back(value);
		}
		matrix.push_back(v);
	}
	return matrix;
}


// Gpu Matrix Implementation
void GpuMatrix::Malloc()
{
	if (this->m_data == nullptr) {

		try
		{
			// allocates a vector of two integers
			this->m_data = (float*)Memory::instance()->allocate(this->m_cols * this->m_rows * this->m_channels *
				sizeof(float), Bridge::GPU);
		}
		catch (MemoryAllocationException exception) {
			ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
				(exception.what());
		}
		catch (...) {
			ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
				("Unknown Error");
		}
		cudaMemset(this->m_data, 0, m_cols * m_rows * m_channels * sizeof(float));
	}
}

GpuMatrix::GpuMatrix(int height, int width, int channels)
	: GenericMatrix(height, width, channels) {

	this->Malloc();
}

GpuMatrix::GpuMatrix() : GenericMatrix() {

}

void GpuMatrix::Free() {
	Memory::instance()->deallocate(this->m_data, Bridge::GPU);

}

void GpuMatrix::Memcpy(GenericMatrix& rhs) {
	cudaMemcpy(this->m_data, rhs.getData(), getLength(),cudaMemcpyKind::cudaMemcpyDeviceToDevice);
}

void GpuMatrix::Set(int y, int x, int channel, float val) {
	if (this->m_data == nullptr) {
		this->Malloc();
	}
	if (y >= this->m_rows || x >= this->m_cols || channel >= this->m_channels) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid position Gpu");
		throw new std::exception("invalid");
	}
	/*Need to store the float value inside a pointer*/
	float *pData = (float*)malloc(sizeof(float));
	pData = &val;
	cudaError status =  cudaMemcpy(this->m_data + RC2IDX(y, x, this->m_cols) + channel *
		(this->m_rows * this->m_cols), pData, sizeof(float),
		cudaMemcpyKind::cudaMemcpyHostToDevice);

	if (status != cudaError::cudaSuccess) {
		throw new std::exception("Error cudaMemcpy\n");
	}
}

void GpuMatrix::Set(int position, const VectorFloat& rhs) {
	if (this->m_data == nullptr) {
		this->Malloc();
	}
	if (position >= this->m_cols * this->m_rows) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid position Gpu");
		throw new std::exception("invalid");
	}
	float pData;
	for (int i = 0; i < this->m_channels; i++) {
		pData = rhs.Get(i);
		cudaMemcpy(this->m_data + position + i *
			(this->m_rows * this->m_cols), &pData, sizeof(float),
			cudaMemcpyKind::cudaMemcpyHostToDevice);
	}
}

void GpuMatrix::Set(int position,int channel, const VectorFloat& rhs) {
	if (this->m_data == nullptr) {
		this->Malloc();
	}
	if (position >= this->m_cols * this->m_rows) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid position Gpu");
		throw new std::exception("invalid");
	}
	/*Need to store the float value inside a pointer*/
	float *pData = (float*)malloc(sizeof(float));
	cudaMemcpy(this->m_data +  position + channel *
		(this->m_rows * this->m_cols), pData, sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
}

float GpuMatrix::Get(int y, int x, int channel) const {

	if (this->m_data == nullptr || y >= this->m_rows || x >= this->m_cols || channel >= this->m_channels) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid position Gpu");
		throw new std::exception("invalid");
	}
	float host_data = 0;
	auto error = cudaMemcpy(&host_data, this->m_data + 
		RC2IDX(y, x, this->m_cols) + channel * (this->m_rows * this->m_cols),
		sizeof(float), cudaMemcpyDeviceToHost);
	if (error != cudaError::cudaSuccess) {
		throw new std::exception("CudaMemcpy error");
	}
	return host_data;
}

GpuMatrix::GpuMatrix(const GenericMatrix& rhs) : GenericMatrix(rhs) {
	this->m_data = nullptr;
	this->Clone(rhs);
}

void GpuMatrix::Clone(const GenericMatrix& rhs)  {
	this->m_cols = rhs.getCols();
	this->m_rows = rhs.getRows();
	this->m_channels = rhs.getChannels();
	this->Malloc();
	this->Memcpy(const_cast<GenericMatrix&>(rhs));
}


GenericMatrix& GpuMatrix::operator+(const GenericMatrix& rhs) const {

	if (this->m_data == nullptr || rhs.getData() == nullptr || getLength() != rhs.getLength()) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid arguments");
		throw new std::exception("Invalid arguments");
	}
	int length = getLength();
	auto gpuMatrix = new GpuMatrix(rhs);
	for (int i = 0; i < length; ++i) {
		gpuMatrix->m_data[i] = rhs.getData()[i] + m_data[i];
	}
	return *gpuMatrix;
}