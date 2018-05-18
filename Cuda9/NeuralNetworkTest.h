#pragma once
#include "ITest.h"
#include "template_data.h"

namespace gpuNN
{
	class NeuralNetworkTest :
		public ITest
	{
	protected:
		HostMatrix<int> results;
		int classes;
	public:
		NeuralNetworkTest(int classes);
		~NeuralNetworkTest();
		virtual void Reset();
		virtual void Classify(int correct, int predicted);
		virtual double Accuracy();
		virtual void Show();
		double Precision();
	protected:
		int FP(int _class);
		int TP(int _class);
		int Positives(int _class);
		int FN(int _class);
		double Precision(int _class);
	};
}
