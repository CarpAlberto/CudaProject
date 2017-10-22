#pragma once
#include "BaseTest.h"
#include <vector.h>
#include <iostream>
#include <NeuralNetwork.h>
using namespace TestProject;

class TestNeuralNetwork :
	public BaseTest {
	
public:
	static void TestConstructor() {

		std::vector<size_t> topology = {3,2,1};
		std::vector<double> data = { 1,0,1 };

		gpuNN::NeuralNetwork network(topology);
		network.SetCurrentInput(data);
		network.feedForward();
		network.Print();
	}
};