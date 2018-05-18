#pragma once
#include "includes.h"
#include "INeuralNetwork.h"

class IAnalyzer{

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
	virtual void BuildFeatures(const std::string& filename,
		INeuralNetwork*& network,bool) = 0;

	/// <summary>
	/// Fetch the features from the filename and store them in the file given the out
	/// </summary>
	/// <param name="filename">The given filename</param>
	/// <param name="out">The out filename`</param>
	virtual void BuildFeatures(const std::string& filename, 
		const std::string& out)=0;

	/// <summary>
	/// Builds the features from the database.
	/// </summary>
	/// <param name="directory"></param>
	/// <param name="database"></param>
	virtual void BuildFeaturesFromDirectory(const std::string& directory, 
		const std::string& database) = 0;

	virtual void BuildDataFromDirectory(const std::string& directory,
		const std::string& file_out)=0;

	virtual void TrainNeuralNetworkFromBothDirectories(const std::string& directory,
		const std::string& benigns, const std::string& malware,
		const std::string& neuralNetworkOut) = 0;

	virtual void TestNeuralNetwork(const std::string& file,
		const std::string& database) =0;

	virtual void TrainNeuralNetwork(const std::string& directory, const std::string& rhs, bool save)=0;

	virtual float TrainNeuralNetwork(const std::string& directory, const std::string& rhs,
		float rms,float hiddenLayer,float treeshold)=0;
};