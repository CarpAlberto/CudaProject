#pragma once
#include "includes.h"

class MemoryAllocationException :
	public std::exception
{
public:
	MemoryAllocationException(const std::string&);
	~MemoryAllocationException();
};

