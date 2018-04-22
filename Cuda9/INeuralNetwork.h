#pragma once

class INeuralNetwork {

public:
	/// <summary>
	/// Perform the training operation with given epocks 
	/// </summary>
	/// <param name="noEpocks">The number of epocks</param>
	virtual void Train(int noEpocks) = 0;

	/// <summary>
	/// Sets the curent input
	/// </summary>
	virtual void SetCurrentInput(const vDouble& input) = 0;

	/// <summary>
	/// Sets the target of the ANN
	/// </summary>
	/// <param name="target">The target ANN</param>
	virtual void SetCurrentTarget(const vDouble& target) = 0;

	/// <summary>
	/// Save the configuration of the network in the filename
	/// </summary>
	/// <param name="filename"></param>
	virtual void Save(const std::string& filename, IOStrategy strategy)=0;

	/// <summary>
	/// Loads the neural network from the database
	/// </summary>
	/// <param name="filename"></param>
	/// <param name="strategy"></param>
	virtual void Load(const std::string& filename, IOStrategy strategy)=0;

};