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
}