#pragma once
#include "includes.h"

namespace gpuNN 
{
	template<typename Object>
	class Configuration
	{
		std::map<std::string, Object> m_content;
	public:
		Configuration(const std::string& fileName);
		Object Value(const std::string& section, const std::string& entry);
		~Configuration();
	};


	template<typename Object>
	Configuration<Object>
		::Configuration(const std::string&  fileName) {

		std::ifstream file(fileName.c_str());
		std::string line;
		std::string name;
		std::string value;
		std::string inSection;
		int posEqual;

		while (std::getline(file, line)) {

			if (!line.length()) continue;
			if (line[0] == '#') continue;
			if (line[0] == ';') continue;
			if (line[0] == '[') {
				inSection = Utils::Trim(line.substr(1, line.find(']') - 1));
				continue;
			}
			posEqual = line.find('=');
			name = Utils::Trim(line.substr(0, posEqual));
			value = Utils::Trim(line.substr(posEqual + 1));
			this->m_content[inSection + '/' + name] = Object(value);
		}

	}

	template<typename Object>
	Object Configuration<Object>::Value(const std::string& section, const std::string& entry) {

		auto iterator = this->m_content.find(section + '/' + entry);
		if (iterator == this->m_content.end()) {
			throw new std::exception("configuration entry not found");
		}
		return iterator->second;
	}
}