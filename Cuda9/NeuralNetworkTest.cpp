#include "NeuralNetworkTest.h"
namespace gpuNN
{


	NeuralNetworkTest::NeuralNetworkTest(int classes)
	{
		this->classes = classes;
		this->results.ResizeWithoutPreservingData(classes, classes);
		this->Reset();
	}


	NeuralNetworkTest::~NeuralNetworkTest()
	{
	}

	void NeuralNetworkTest::Reset()
	{
		for (int c = 0; c < classes; c++) {
			for (int p = 0; p < classes; p++) 
				results(c, p) = 0;
		}
	}
	void NeuralNetworkTest::Classify(int correct, int predicted)
	{
		results(correct, predicted)++;
	}

	double NeuralNetworkTest::Accuracy()
	{
		double correct = 0;
		double total = 0;

		for (int c = 0; c < classes; c++) {
			for (int p = 0; p < classes; p++) {
				int classified = results(c, p);

				if (c == p) correct += classified;
				total += classified;
			}
		}
		return correct / total;
	}

	double NeuralNetworkTest::Precision()
	{
		int count = 0;
		double sum = 0.0;
		for (int c = 0; c < classes; c++) {
			if (Positives(c) > 0) {
				sum += Precision(c);
				count++;
			}
		}
		return sum / count;
	}
	int NeuralNetworkTest::FP(int _class)
	{
		int fp = 0;
		for (int c = 0; c < classes; c++) 
			if (c != _class) 
				fp += results(c, _class);

		return fp;
	}
	
	int NeuralNetworkTest::TP(int _class)
	{
		return results(_class, _class);
	}
	int NeuralNetworkTest::Positives(int _class)
	{
		return TP(_class) + FN(_class);
	}
	int NeuralNetworkTest::FN(int _class)
	{
		int fn = 0;
		for (int c = 0; c < classes; c++) 
			if (c != _class) 
				fn += results(_class, c);
		return fn;
	}

	double NeuralNetworkTest::Precision(int _class)
	{
		int tp = TP(_class);
		if (tp == 0) {
			return (Positives(_class) == 0) ? 1.0 : 0.0;
		}
		else {
			return (double)tp / (tp + FP(_class));
		}
	}

	void NeuralNetworkTest::Show(){
		std::cout << std::endl << "\t\tPredicted" << std::endl;
		std::cout << "actual\t";

		for (int p = 0; p < classes; p++) 
			std::cout << '\t' << p;
		std::cout << std::endl << std::endl;

		for (int c = 0; c < classes; c++) {
			std::cout << c << '\t';
			for (int p = 0; p < classes; p++) 
				std::cout << '\t' << results(c, p);
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}