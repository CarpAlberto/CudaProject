#pragma once
#include "includes.h"
#include "INeuralNetwork.h"

class IAnalyzer
{
public:
	/// <summary>
	/// Perform the analyze of the given filename with the neural network
	/// </summary>
	/// <param name="filename">The filename</param>
	/// <param name="network">The used neural network</param>
	virtual void Analyze(INeuralNetwork* network) = 0;

	/// <summary>
	/// Fetch the features from the filename provide by the <param name="filename">filename</param> and
	/// vonstruct the topology inside the network
	/// </summary>
	/// <param name="filename">The file to be analyzed</param>
	/// <param name="network">The ANN</param>
	virtual void BuildFeatures(const std::string& filename, INeuralNetwork*& network) = 0;
};