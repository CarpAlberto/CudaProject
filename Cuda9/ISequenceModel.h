#pragma once
#include "includes.h"

class ISequenceModel {
public:
	virtual vDouble toVector() = 0;
};