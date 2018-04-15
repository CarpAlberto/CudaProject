#pragma once
#include "MalwareAnalyzer.h"
#include "NeuralNetwork.h"
#include "ApplicationContext.h"

namespace gpuNN{
	class ApplicationManager
	{
	protected:
		IAnalyzer * analyzer;
		INeuralNetwork * neuralNetwork;
		ApplicationContext* instance;
	public:
		ApplicationManager();
		~ApplicationManager();
	};
}