#pragma once
#include "ApplicationContext.h"
#include <string>

namespace gpuNN {

	std::string ApplicationConfiguration::getMode() const {
		std::string type = baseConfiguration.Value("GeneralSettings", "MODE");
		return type;
	}

	size_t ApplicationConfiguration::getThreadBlockSize() const {
		char *endPtr;
		std::string value = baseConfiguration.Value("GeneralSettings", "THREAD_BLOCK_SIZE");
		return (size_t)strtol(value.c_str(), &endPtr, 10);
	}

	ApplicationConfiguration::ApplicationConfiguration(const std::string& fileName) {
		this->baseConfiguration.Construct(fileName);
	}

	bool ApplicationConfiguration::isDirectoryModeEnabled() {
		std::string type = baseConfiguration.Value("GeneralSettings",
			"DIRECTORY_GENERATE_FEATURES");
		if (type == "FALSE")
			return false;
		if (type == "TRUE")
			return true;
	}

	bool ApplicationConfiguration::isFilenameModeEnabled(){

		std::string type = baseConfiguration.Value("GeneralSettings",
			"FILENAME_GENERATE_FEATURES");
		if (type == "FALSE")
			return false;
		if (type == "TRUE")
			return true;
	}
	std::string ApplicationConfiguration::getDirectoryBenigns(){
		return baseConfiguration.Value("Directory",
			"DIRECTORY_BENIGN");
	}
	std::string ApplicationConfiguration::getDirectoryMalware() {
		return baseConfiguration.Value("Directory",
			"DIRECTORY_VIRUSES");
	}

	std::string ApplicationConfiguration::getFilenameIn()
	{
		return baseConfiguration.Value("Filename",
			"FILENAME_BENIGN");
	}

	std::string ApplicationConfiguration::getFilenameOut()
	{
		return baseConfiguration.Value("Filename",
			"FILENAME_VIRUSES");
	}

	std::string ApplicationConfiguration::getDirectoryBase()
	{
		return baseConfiguration.Value("Directory",
			"DIRECTORY_BASE");
	}
	std::string ApplicationConfiguration::getDatabaseOut()
	{
		return baseConfiguration.Value("Directory",
			"DATABASE_OUT");
	}

	bool ApplicationConfiguration::isStringLengthEncoding()
	{
		auto bValue =  baseConfiguration.Value("GeneralSettings",
			"ENCODING_ALGORITHMS");
		if (bValue == "FILE_LENGTH")
			return true;
		else
			return false;
	}

	std::string ApplicationConfiguration::getTrainBenignsDirectory()
	{
		return baseConfiguration.Value("Directory",
			"DIRECTORY_TRAIN_BENIGNS");
	}

	bool ApplicationConfiguration::isTrainingModeEnabled(){
		auto bValue = baseConfiguration.Value("GeneralSettings",
			"TRAIN_MODE");
		if (bValue == "TRUE")
			return true;
		else
			return false;
	}
	bool ApplicationConfiguration::isGenerateDataMode(){
		auto bValue = baseConfiguration.Value("GeneralSettings",
			"GENERATE_DATA_MODE");
		if (bValue == "TRUE")
			return true;
		else
			return false;
	}
	bool ApplicationConfiguration::isTestMode()
	{
		auto bValue = baseConfiguration.Value("GeneralSettings",
			"TEST_MODE");
		if (bValue == "TRUE")
			return true;
		else
			return false;
	}
	float ApplicationConfiguration::getRootMeanSquareMin(){
		auto bValue = baseConfiguration.Value("NeuralNetworkParameters",
			"RootMeanSquareMin");
		return stof(bValue);
	}

	int ApplicationConfiguration::getEpocksLimit()
	{
		auto bValue = baseConfiguration.Value("NeuralNetworkParameters",
			"EpocksLimis");
		return stoi(bValue);
	}

	std::string ApplicationConfiguration::getTrainDirectory(){
		return baseConfiguration.Value("Directory",
			"DIRECTORY_TRAIN");
	}

	std::string ApplicationConfiguration::getTestDirectory(){
		return baseConfiguration.Value("Directory",
			"DIRECTORY_TEST");
	}

	std::string ApplicationConfiguration::getDatabaseInstruction()
	{
		return baseConfiguration.Value("Directory",
			"INSTR_DATABASE_OUT");
	}
}