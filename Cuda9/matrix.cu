#pragma once
#include "matrix.h"

using namespace gpuNN;

GenericMatrix::GenericMatrix() {
	m_cols = 0;
	m_rows = 0;
	m_data = nullptr;
	m_channels = 0;
}

GenericMatrix::GenericMatrix(const GenericMatrix& rhs) {
	this->m_cols = rhs.m_cols;
	this->m_rows = rhs.m_rows;
	this->m_channels = rhs.m_channels;

	this->m_data = nullptr;
	this->Malloc();

}

GenericMatrix::GenericMatrix(int height, int width, int channels) {
	this->m_cols = width;
	this->m_rows = height;
	this->m_channels = channels;
	this->m_data = nullptr;
	Malloc();
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

GenericMatrix& GenericMatrix::operator<<=(GenericMatrix& rhs)
{
	this->Free();
	this->m_cols = rhs.m_cols;
	this->m_rows = rhs.m_rows;
	this->m_channels = rhs.m_channels;
	this->Malloc();
	this->Memcpy(const_cast<GenericMatrix&>(rhs));
	rhs.Release();
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

VectorFloat		GenericMatrix::Get(int y, int x)const {
	if (this->m_data == nullptr) {
		if (x >= this->m_cols || y >= this->m_rows || 3 > this->m_channels) {
			ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
				("Invalid position");
			throw new std::exception("Invalid position");
		}
	}
	VectorFloat response;
	for (int i = 0; i < 3; ++i) {
		response.Set(i, this->m_data[RC2IDX(y, x, this->m_cols) + i * (this->m_rows * this->m_cols)]);
	}
	return response;
}

VectorFloat		GenericMatrix::Get(int index)const {
	VectorFloat response;
	return response;
}



// Cpu Matrix Implementation

CpuMatrix::CpuMatrix(int height, int width, int channels)  
	: GenericMatrix(height,width,channels) {

}
void CpuMatrix::Set(int y, int x, int channel, float val) {
	if (this->m_data == nullptr) {
		Zeros();
	}
	if (y >= this->m_cols || x >= this->m_rows || channel >= this->m_channels) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid position");
		return;
	}
	this->m_data[RC2IDX(y,x,this->m_cols) + channel * (this->m_rows * this->m_cols)] = val;
}

void CpuMatrix::Set(int y, int x, const VectorFloat& rhs) {
	if (this->m_data == nullptr) {
		Zeros();
	}
	if (y >= this->m_cols || x >= this->m_rows) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid position");
		return;
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
		memset(this->m_data, m_cols * m_rows * m_channels, sizeof(float));
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
	CpuMatrix* cpuMatrix = new CpuMatrix(rhs);
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

	if (this->m_data == nullptr || rhs.getData() == nullptr || getLength() != rhs.getLength()) {
		ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
			("Invalid arguments");
		throw new std::exception("Invalid arguments");
	}
	int length = getLength();
	CpuMatrix* cpuMatrix = new CpuMatrix(rhs);
	for (int i = 0; i < length; ++i) {
		cpuMatrix->m_data[i] = rhs.getData()[i] * m_data[i];
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
