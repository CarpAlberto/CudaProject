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



};