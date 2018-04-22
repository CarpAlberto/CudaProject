#include "StringModel.h"



StringModel::StringModel(const vStrings& rhs)
{
	this->arrayStringsBool = rhs;
}

StringModel::StringModel(const vDouble& rhs)
{
	this->vDoble = rhs;
}

StringModel::~StringModel()
{
}

vDouble StringModel::toVector()
{
	return this->vDoble;
}