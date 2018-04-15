#include "ApplicationManager.h"

using namespace gpuNN;

ApplicationManager::ApplicationManager()
{
	this->analyzer = new MalwareAnalyzer();

	this->analyzer->BuildFeatures("xmrig-amd.exe", this->neuralNetwork);

	this->analyzer->Analyze(this->neuralNetwork);

	this->neuralNetwork->Save("trained.txt", IOStrategy::ASCII);



}


ApplicationManager::~ApplicationManager()
{
}
