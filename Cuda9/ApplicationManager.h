#pragma once
#include "MalwareAnalyzer.h"
#include "NeuralNetwork.h"

namespace gpuNN{
	class ApplicationManager
	{
	protected:
		IAnalyzer * analyzer;
		INeuralNetwork * neuralNetwork;
	public:
		ApplicationManager();
		~ApplicationManager();
	};
}