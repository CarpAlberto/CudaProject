#pragma once

class ITest {

public:
	virtual void Reset() = 0;
	virtual void Classify(int,int) = 0;
	virtual double Accuracy()=0;

	virtual void Show()=0;

};