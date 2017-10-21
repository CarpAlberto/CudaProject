#pragma once
#include "BaseTest.h"
#include <vector.h>
#include <iostream>
#include <NeuralNetwork.h>
using namespace TestProject;

class TestNeuralNetwork :
	public BaseTest {
	
public:
	void TestConstructor() {

		std::vector<size_t> topology = {3,2,1};
		std::vector<double> data = { 2,3,2 };

		gpuNN::NeuralNetwork network(topology);
		network.SetCurrentInput(data);
		network.feedForward();
	}
};