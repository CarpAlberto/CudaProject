#pragma once
#include "MalwareAnalyzer.h"
#include "NeuralNetwork.h"
#include "ApplicationContext.h"

namespace gpuNN{

	/// <summary>
	/// The manager of application used
	/// </summary>
	class ApplicationManager
	{
	protected:
		/// <summary>
		/// The analyzer of the network
		/// </summary>
		IAnalyzer * analyzer;
		/// <summary>
		/// The instance of neural network
		/// </summary>
		INeuralNetwork * neuralNetwork;
		/// <summary>
		/// Pointer to the instance of the application context
		/// </summary>
		ApplicationContext* instance;
	protected:
		void ParseInternal(char** argv, int argc);
		void ParseConfig();
		void ParseTrain(char** argv, int argc);
		void ParseGenerateFeaturesItem(char** argv, int argc,int argStart);
		void ParseGenerateFeaturesFilename(char** argv, int argc, int argStart);
		void ParseDirectory(const std::string& directory);
	public:
		ApplicationManager();
		~ApplicationManager();
		/// <summary>
		/// Returns the application context
		/// </summary>
		/// <returns></returns>
		ApplicationContext* getContext();
	public:
		/// <summary>
		/// Parse the given arguments and executes the commands.
		/// --generate-feature -d directory - -Directory To generate the array of feature based on the given directory
		/// --generate-feature -f filename  -Generate the feature based on the data
		/// </summary>
		/// <param name="argv">The name of the arguments</param>
		/// <param name="argc">The cound of arguments</param>
		void Parse(char** argv, int argc);

		void TrainFromDirectory();
	};
}