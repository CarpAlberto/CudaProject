#pragma once
#include "includes.h"

class NonCopyableObject
{
public:
	NonCopyableObject() = default;
	~NonCopyableObject() = default;
public:
	NonCopyableObject(const NonCopyableObject&) = delete;
	NonCopyableObject& operator=(const NonCopyableObject&) = delete;
};