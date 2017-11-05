#include "MemoryAllocationException.h"



MemoryAllocationException::MemoryAllocationException(const std::string& message) 
	:exception(message.c_str())
{
}


MemoryAllocationException::~MemoryAllocationException()
{
}
