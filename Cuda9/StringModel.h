#pragma once
#include "ISequenceModel.h"
class StringModel : public ISequenceModel
{
	vStrings arrayStringsBool;
	vDouble vDoble;
public:
	virtual vDouble toVector();
	StringModel(const vStrings& rhs);
	StringModel(const vDouble& rhs);
	StringModel() = default;
	~StringModel();
};

