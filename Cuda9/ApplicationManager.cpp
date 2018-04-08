#include "ApplicationManager.h"

using namespace gpuNN;

ApplicationManager::ApplicationManager()
{
	this->analyzer = new MalwareAnalyzer();

	this->analyzer->BuildFeatures("TestProject.exe", this->neuralNetwork);

	this->analyzer->Analyze(this->neuralNetwork);

}


ApplicationManager::~ApplicationManager()
{
}
