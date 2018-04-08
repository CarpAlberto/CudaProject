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

		gpuNN::NeuralNetwork network(topology,1,0.05,1);
		network.SetCurrentInput(data);
		network.SetCurrentTarget(data);
		network.FeedForward();
		network.setErrors();
		network.BackPropagation();
		std::cout << "Total Error: "  << network.getTotalError() << std::endl;
		network.FeedForward();
		network.setErrors();
		network.BackPropagation();
	}

	static void TestIterativeNetwork() {

		std::vector<size_t> topology = { 3,2,3 };
		std::vector<double> data = { 1,0,1 };

		gpuNN::NeuralNetwork network(topology, 1, 0.1, 1);
		network.SetCurrentInput(data);
		network.SetCurrentTarget(data);
		for (int i = 0; i < 280; i++) {
			std::cout << "Epock:" << i << std::endl;
			network.FeedForward();
			network.setErrors();
			std::cout << "Total Error: " << network.getTotalError() << std::endl;
			network.BackPropagation();

			std::cout << "===========" << std::endl;
			std::cout << "Output" << std::endl;
			network.PrintOutput();

			std::cout << "===========" << std::endl;
			std::cout << "Target" << std::endl;
			network.PrintTarget();

			std::cout << "===============" << std::endl;
		}
		Memory::instance()->PrintMemoryUsage();
		//Memory::instance()->PrintLayoutMemory();
	}

	static void TestTrain() 
	{
		std::vector<size_t> topology = { 3,2,3 };
		std::vector<double> data = { 2,3,4 };

		gpuNN::NeuralNetwork network(topology, 1, 0.1, 1);
		network.SetCurrentInput(data);
		network.SetCurrentTarget(data);

		network.Train(400);

		network.PrintOutput();
		network.PrintTarget();

		Memory::instance()->PrintMemoryUsage();
		ApplicationContext::instance()->destroy();
	}

};