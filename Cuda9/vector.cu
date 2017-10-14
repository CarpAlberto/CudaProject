#include "vector.h"

using namespace gpuNN;


void VectorInteger::malloc() {

	if (m_data == nullptr)
	{
		try
		{
			// allocates a vector of two integers
			this->m_data = (int*)Memory::instance()->allocate(2 * sizeof(int), Bridge::CPU);
		}
		catch (MemoryAllocationException exception) {
			ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
				(exception.what());
		}
		catch (...) {
			ApplicationContext::instance()->getLog().get()->print<SeverityType::ERROR>
				("Unknown Error");
		}
		memset(this->m_data, 0, 2 * sizeof(int));
	}
}

void VectorInteger::Set(int position, int value)
{
	m_data[position] = value;
}

VectorInteger::VectorInteger(int a , int b)
{
	m_data = nullptr;
	this->malloc();
	Set(0, a);
	Set(1, b);
}
VectorInteger::VectorInteger()
{
	this->m_data = nullptr;
}

VectorInteger::VectorInteger(const VectorInteger& integer)
{
	this->m_data = nullptr;
	this->malloc();
	memcpy(this->m_data, integer.m_data, 2 * sizeof(int));
}

VectorInteger::~VectorInteger() {
	if (this->m_data != nullptr)
		Memory::instance()->deallocate(m_data, Bridge::CPU);
	this->m_data = nullptr;
}

void VectorInteger::Release() {
	if (this->m_data != nullptr)
		Memory::instance()->deallocate(m_data, Bridge::CPU);
	this->m_data = nullptr;
}

VectorInteger& VectorInteger::operator=(const VectorInteger& rhs)
{
	if (this->m_data != nullptr) {
		Memory::instance()->deallocate(m_data, Bridge::CPU);
		this->m_data = nullptr;
	}
	this->malloc();
	memcpy(this->m_data, rhs.m_data, 2 * sizeof(int));
	return *this;
}

void VectorInteger::Zeros()
{
	for (auto i = 0; i < 2; ++i) {
		Set(i, 0);
	}
}
void VectorInteger::Ones()
{
	for (auto i = 0; i < 2; ++i) {
		Set(i,1);
	}
}

void VectorInteger::SetAll(int val) {
	for (int i = 0; i < 2; ++i) {
		Set(i, val);
	}
}

void VectorInteger::CopyTo(VectorInteger& rhs) const
{
	if (rhs.m_data == nullptr) {
		Memory::instance()->deallocate(rhs.m_data, Bridge::CPU);
		rhs.m_data = nullptr;
	}
	rhs.malloc();
	memcpy(rhs.m_data, m_data, 2 * sizeof(int));
}

VectorInteger VectorInteger::operator+(const VectorInteger& rhs) const
{
	VectorInteger tmp;
	this->CopyTo(tmp);
	for (auto i = 0; i < 2; i++) {
		tmp.Set(i, tmp.Get(i) + rhs.Get(i));
	}
	return tmp;
}

VectorInteger VectorInteger::operator-(const VectorInteger& rhs) const
{
	VectorInteger tmp;
	this->CopyTo(tmp);
	for (auto i = 0; i < 2; i++) {
		tmp.Set(i, tmp.Get(i) - rhs.Get(i));
	}
	return tmp;
}

VectorInteger VectorInteger::operator*(const VectorInteger& rhs) const
{
	VectorInteger tmp;
	this->CopyTo(tmp);
	for (auto i = 0; i < 2; i++) {
		tmp.Set(i, tmp.Get(i) * rhs.Get(i));
	}
	return tmp;
}

VectorInteger VectorInteger::operator+(int rhs) const
{
	VectorInteger tmp;
	this->CopyTo(tmp);
	for (auto i = 0; i < 2; i++) {
		tmp.Set(i, tmp.Get(i) + rhs);
	}
	return tmp;
}

VectorInteger VectorInteger::operator-(int rhs) const
{
	VectorInteger tmp;
	this->CopyTo(tmp);
	for (auto i = 0; i < 2; i++) {
		tmp.Set(i, tmp.Get(i) - rhs);
	}
	return tmp;
}

VectorInteger VectorInteger::operator*(int rhs) const
{
	VectorInteger tmp;
	this->CopyTo(tmp);
	for (auto i = 0; i < 2; i++) {
		tmp.Set(i, tmp.Get(i) * rhs);
	}
	return tmp;
}

VectorInteger& VectorInteger::operator+= (const VectorInteger& rhs)
{
	for (auto i = 0; i < 2; i++) {
		this->Set(i, Get(i) + rhs.Get(i));
	}
	return *this;
}

VectorInteger& VectorInteger::operator -= (const VectorInteger& rhs)
{
	for (auto i = 0; i < 2; i++) {
		this->Set(i, Get(i) - rhs.Get(i));
	}
	return *this;
}

VectorInteger& VectorInteger::operator *= (const VectorInteger& rhs)
{
	for (auto i = 0; i < 2; i++) {
		this->Set(i, Get(i) * rhs.Get(i));
	}
	return *this;
}

VectorInteger& VectorInteger::operator+=(int rhs)
{
	for (auto i = 0; i < 2; i++) {
		this->Set(i, Get(i) + rhs);
	}
	return *this;
}

VectorInteger& VectorInteger::operator -= (int rhs)
{
	for (auto i = 0; i < 2; i++) {
		this->Set(i, Get(i) - rhs);
	}
	return *this;
}

VectorInteger& VectorInteger::operator *= (int rhs)
{
	for (auto i = 0; i < 2; i++) {
		this->Set(i, Get(i) * rhs);
	}
	return *this;
}


VectorInteger VectorInteger::Mul(const VectorInteger& rhs) const
{
	VectorInteger tmp;
	CopyTo(tmp);
	for (int i = 0; i < 2; ++i) {
		tmp.Set(i, tmp.Get(i) * rhs.Get(i));
	}
	return tmp;
}

VectorInteger VectorInteger::Mul(const int rhs) const {
	VectorInteger tmp;
	CopyTo(tmp);
	for (int i = 0; i < 2; ++i) {
		tmp.Set(i, tmp.Get(i) * rhs);
	}
	return tmp;
}

void VectorInteger::Print(const std::string& rhs) const {
	std::cout << rhs << std::endl;
	std::cout << "VectorInteger: [ ";
	for (int i = 0; i < 2; ++i) {
		std::cout << Get(i) << " ";
	}
	std::cout << "]" << std::endl;
}

VectorFloat::VectorFloat() {
	this->m_data = nullptr;
	this->malloc();
}

VectorFloat::VectorFloat(float a, float b, float c)
{
	this->m_data = nullptr;
	this->malloc();
	Set(0, a);
	Set(1, b);
	Set(2, c);
}

VectorFloat::VectorFloat(const VectorFloat& rhs) {
	this->m_data = nullptr;
	this->malloc();
	memcpy(this->m_data, rhs.m_data, 3 * sizeof(float));
}

VectorFloat::~VectorFloat() {
	if (this->m_data != nullptr) {
		Memory::instance()->deallocate(this->m_data,Bridge::CPU);
	}
	this->m_data = nullptr;
}

VectorFloat& VectorFloat::operator=(const VectorFloat& rhs) {

	if (this->m_data != nullptr) {
		Memory::instance()->deallocate(this->m_data, Bridge::CPU);
		this->m_data = nullptr;
	}
	malloc();
	memcpy(this->m_data,rhs.m_data, 3 * sizeof(float));
	return *this;
}