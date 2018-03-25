#pragma once
#include "matrix.h"
#include "Memory.h"

using namespace gpuNN;

GenericMatrix::GenericMatrix() {
	m_cols = 0;
	m_rows = 0;
	m_data = nullptr;
}

GenericMatrix::GenericMatrix(const GenericMatrix& rhs) {
	Clone(rhs);
}

GenericMatrix::GenericMatrix(size_t height, size_t width) : m_cols(width),
					m_rows(height),m_data(nullptr){
	this->Malloc();
}

void GenericMatrix::Release() {
	this->Free();
	this->m_cols = 0;
	this->m_rows = 0;
	this->m_data = nullptr;
}

GenericMatrix& GenericMatrix::operator=(const GenericMatrix& rhs)
{
	this->Free();
	this->m_cols = rhs.m_cols;
	this->m_rows = rhs.m_rows;
	this->Malloc();
	this->Memcpy(const_cast<GenericMatrix&>(rhs));
	return *this;
}

void GenericMatrix::SetSize(size_t rows, size_t cols) noexcept {
	this->Free();
	this->m_cols = cols;
	this->m_rows = rows;
	Malloc();
	Zeros();
}

size_t GenericMatrix::getLength() const noexcept {
	return this->m_cols * this->m_rows;
}

void GenericMatrix::Zeros() {
	SetAll(0.0);
}

void GenericMatrix::Ones() {
	SetAll(1.0);
}

float* GenericMatrix::getData() const noexcept {
	return this->m_data;
}

float  GenericMatrix::Get(size_t y, size_t x)const {
	if (this->m_data == nullptr) {
		if (x >= this->m_cols || y >= this->m_rows) {
			ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
				("Invalid position");
			throw new std::exception("Invalid position");
		}
	}
	return this->m_data[RC2IDX(y,x, this->m_cols)];
}

size_t GenericMatrix::getCols() const {
	return this->m_cols;
}

size_t GenericMatrix::getRows() const {
	return this->m_rows;
}

void GenericMatrix::Print(UIInterface * guiInterface) const {
	
	for (auto i = 0; i < this->getRows(); i++) {
		for (auto j = 0; j < this->getCols(); j++) {
			double value = this->Get(i, j);
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

CpuMatrix::CpuMatrix(size_t height, size_t width):GenericMatrix(height,width) {
	this->Malloc();
}

void CpuMatrix::Set(size_t y, size_t x, float val) {
	if (this->m_data == nullptr) {
		Zeros();
	}
	if (y >= this->m_rows || x >= this->m_cols) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid position");
		throw new std::exception("invalid");
	}
	this->m_data[RC2IDX(y,x,this->m_cols)] = val;
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
			this->m_data = (float*)Memory::instance()->allocate(this->m_cols * this->m_rows  *
				sizeof(float), Bridge::CPU);
		}
		catch (MemoryAllocationException* exception) {
			ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
				(exception->what());
			throw exception;
		}
		catch (...) {
			ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
				("Unknown Error");
			return;
		}
		memset(this->m_data, 0 ,m_cols * m_rows  * sizeof(float));
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
	auto length = getLength();
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
	auto length = getLength();
	CpuMatrix* cpuMatrix = new CpuMatrix(this->m_rows,this->m_cols);
	for (int i = 0; i < length; ++i) {
		cpuMatrix->m_data[i] =  m_data[i] + val;
	}
	return *cpuMatrix;
}

GenericMatrix& CpuMatrix::operator-(const GenericMatrix& rhs) const {

	if (this->m_data == nullptr || rhs.getData() == nullptr || getLength() != rhs.getLength()) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid arguments");
		throw new std::exception("Invalid arguments");
	}
	auto length = getLength();
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
	auto length = getLength();
	CpuMatrix* cpuMatrix = new CpuMatrix(this->m_rows, this->m_cols);
	for (int i = 0; i < length; ++i) {
		cpuMatrix->m_data[i] = m_data[i] - val;
	}
	return *cpuMatrix;
}

GenericMatrix& CpuMatrix::operator*(const GenericMatrix& rhs) const {

	if (this->m_cols != rhs.getRows()) {
		throw new std::exception("Invalid arguments");
	}
	CpuMatrix* cpuMatrix = new CpuMatrix(this->getRows(),rhs.getCols());
	for (int i = 0; i < this->getRows(); i++) {
		for (auto j = 0; j < rhs.getCols(); j++) {
			for (auto k = 0; k < rhs.getRows(); k++) {
				float p = this->Get(i, k) * rhs.Get(k,j);
				float newValue = cpuMatrix->Get(i, j) + p;
				cpuMatrix->Set(i, j, newValue);
			}
			cpuMatrix->Set(i, j,cpuMatrix->Get(i, j));
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
	auto length = getLength();
	CpuMatrix* cpuMatrix = new CpuMatrix(this->m_rows, this->m_cols);
	for (int i = 0; i < length; ++i) {
		cpuMatrix->m_data[i] = m_data[i] * val;
	}
	return *cpuMatrix;
}


void	CpuMatrix::SetAll(float val) {
	if (this->m_data == nullptr) {
		Malloc();
	}
	auto  length = getLength();
	for (auto iterator = 0; iterator < length; iterator++) {
		this->m_data[iterator] = val;
	}
}
void CpuMatrix::Clone(const GenericMatrix& rhs) {
	this->m_cols = rhs.getCols();
	this->m_rows = rhs.getRows();
	this->Malloc();
	this->Memcpy(const_cast<GenericMatrix&>(rhs));
}

GenericMatrix& CpuMatrix::Transpose() const {
	
	auto rows = this->getRows();
	auto columns = this->getCols();

	CpuMatrix * matrix = new CpuMatrix(columns,rows);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
   			matrix->Set(j, i, this->Get(i,j));
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
			double value = this->Get(i, j);
			v.push_back(value);
		}
		matrix.push_back(v);
	}
	return matrix;
}

GpuMatrix::~GpuMatrix() {

	if (this->m_data != nullptr)
		Memory::instance()->deallocate(this->m_data, Bridge::GPU);
}


// Gpu Matrix Implementation
void GpuMatrix::Malloc()
{
	if (this->m_data == nullptr) {

		try
		{
			// allocates a vector of two integers
			this->m_data = (float*)Memory::instance()->allocate(this->m_cols * this->m_rows  *
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
		cudaMemset(this->m_data, 0, m_cols * m_rows * sizeof(float));
	}
}

GpuMatrix::GpuMatrix(size_t height,size_t width)
	: GenericMatrix(height, width) {

	this->Malloc();
}

GpuMatrix::GpuMatrix() : GenericMatrix() {

}

void GpuMatrix::Free() {
	Memory::instance()->deallocate(this->m_data, Bridge::GPU);

}

void GpuMatrix::Memcpy(GenericMatrix& rhs) {
	cudaMemcpy(this->m_data, rhs.getData(), rhs.getLength() * sizeof(float),cudaMemcpyKind::cudaMemcpyDeviceToDevice);
}

void GpuMatrix::Set(size_t y, size_t x, float val) {
	if (this->m_data == nullptr) {
		this->Malloc();
	}
	if (y >= this->m_rows || x >= this->m_cols) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid position Gpu");
		throw new std::exception("invalid");
	}
	/*Need to store the float value inside a pointer*/
	float *pData = (float*)malloc(sizeof(float));
	pData = &val;
	cudaError status = cudaMemcpy(this->m_data + RC2IDX(y, x, this->m_cols), pData, sizeof(float),
		cudaMemcpyKind::cudaMemcpyHostToDevice);

	if (status != cudaError::cudaSuccess) {
		throw new std::exception("Error cudaMemcpy\n");
	}
}


float GpuMatrix::Get(size_t y, size_t x) const {

	if (this->m_data == nullptr || y >= this->m_rows || x >= this->m_cols) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid position Gpu");
		throw new std::exception("invalid");
	}
	float host_data = 0;
	auto error = cudaMemcpy(&host_data, this->m_data + 
		RC2IDX(y, x, this->m_cols),sizeof(float), cudaMemcpyDeviceToHost);
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
	this->Malloc();
	this->Memcpy(const_cast<GenericMatrix&>(rhs));
}

void GpuMatrix::SetRandom() {
	if (m_data == nullptr)
		Malloc();
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(generator, rand() % 3456);
	curandGenerateUniform(generator, this->m_data, getLength());
	curandDestroyGenerator(generator);
}

GenericMatrix& GpuMatrix::operator+(const GenericMatrix& rhs) const {

	if (this->m_data == nullptr || rhs.getData() == nullptr || getLength() != rhs.getLength()) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid arguments");
		throw new std::exception("Invalid arguments");
	}
	auto length = getLength();
	GenericMatrix* gpuMatrix = new GpuMatrix(rhs);

	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (length / block_size) + ((length % block_size) ? 1 : 0);

	gpu_add <<< num_blocks, block_size >>> (gpuMatrix->getData(),this->m_data, length);
	return *gpuMatrix;
 }

GenericMatrix& GpuMatrix::operator*(const GenericMatrix& rhs) const {
  	
	GenericMatrix * returnMatrix = new GpuMatrix(this->m_rows, rhs.getCols());
	int TILE_WIDTH = 32;

	dim3 dimGrid;
	dim3 dimBlock(TILE_DIM, TILE_DIM,1);

	dimGrid.x = (rhs.getCols() + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (this->m_rows  + dimBlock.y - 1) / dimBlock.y;

	gpu_multiply <<<dimGrid, dimBlock >>> (this->m_data, rhs.getData(), returnMatrix->getData(),
		this->m_rows,this->m_cols,
		rhs.getRows(),rhs.getCols(),
		returnMatrix->getRows(), returnMatrix->getCols());

	return *returnMatrix;

}

GenericMatrix& GpuMatrix::Transpose() const
{
	auto rows = this->getRows();
	auto columns = this->getCols();

	GpuMatrix * matrix = new GpuMatrix(columns, rows);
	int len = matrix->getRows() * matrix->getCols();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	
	gpu_transpose <<<num_blocks, block_size >>> (this->m_data, matrix->getData(),this->m_cols,matrix->getCols(),len);
	return *matrix;
}

mDouble  GpuMatrix::getAsMatrix()
{
	mDouble matrix;
	for (int i = 0; i < this->m_rows; ++i) {
		vDouble v;
		for (int j = 0; j < this->m_cols; ++j) {
			double value = this->Get(i, j);
			v.push_back(value);
		}
		matrix.push_back(v);
	}
	return matrix;
}