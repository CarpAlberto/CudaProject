#pragma once
#include "ApplicationContext.h"

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

	bool ApplicationConfiguration::isTrainingModeEnabled()
	{
		auto bValue = baseConfiguration.Value("GeneralSettings",
			"ENABLE_TRAINING");
		if (bValue == "TRUE")
			return true;
		else
			return false;
	}
}