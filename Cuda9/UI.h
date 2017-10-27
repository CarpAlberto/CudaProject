#pragma once
#include <string>
#include <iostream>

/// <summary>
/// A generic interface for any GUI interface
/// </summary>
class UIInterface {

public:
	/// <summary>
	/// Display an message to the screen
	/// </summary>
	/// <param name="message">The message to be displayed
	/// </param>
	virtual void showMessage(const std::string& message)=0;
	/// <summary>
	/// Display an error message to the screen
	/// </summary>
	/// <param name="errorMessage">The error message to be displayed</param>
	virtual void showErrorMessage(const std::string& errorMessage)=0;
	/// <summary>
	/// Shows a double to the screen
	/// </summary>
	virtual void Show(double value)=0;
	/// <summary>
	/// Shows an Integer to the 
	/// </summary>
	/// <param name="value">The value to be printed</param>
	virtual void Show(int value) = 0;
};