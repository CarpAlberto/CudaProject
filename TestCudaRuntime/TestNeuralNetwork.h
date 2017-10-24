#pragma once
#include "BaseTest.h"
#include <vector.h>
#include <iostream>
#include <NeuralNetwork.h>
using namespace TestProject;

class TestNeuralNetwork :
	public BaseTest {
	
public:
	static void TestNetwork() {

		std::vector<size_t> topology = {3,2,3};
		std::vector<double> data = { 1,0,1 };

		gpuNN::NeuralNetwork network(topology);
		network.SetCurrentInput(data);
		network.SetCurrentTarget(data);
		network.feedForward();
		network.setErrors();
		network.BackPropagation();
		std::cout << "Total Error: "  << network.getTotalError() << std::endl;
		network.feedForward();
		network.setErrors();
		network.BackPropagation();
	}
	static void TestIterativeNetwork() {

		std::vector<size_t> topology = { 3,2,3 };
		std::vector<double> data = { 1,0,1 };

		gpuNN::NeuralNetwork network(topology);
		network.SetCurrentInput(data);
		network.SetCurrentTarget(data);
		for (int i = 0; i < 5; i++) {
			std::cout << "Epock:" << i << std::endl;
			network.feedForward();
			network.setErrors();
			std::cout << "Total Error: " << network.getTotalError() << std::endl;
			network.BackPropagation();
		
		}
	}
};