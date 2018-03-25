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
		Configuration() = default;
		Object Value(const std::string& section, const std::string& entry) const;
		void Construct(const std::string& filename);
		~Configuration();
	};

	/// <summary>
	/// The Application Configuration
	/// </summary>
	class ApplicationConfiguration {

	private:
		/// <summary>
		/// The Base configuration.This should change to inherit
		/// </summary>
		Configuration<std::string> baseConfiguration;
	public:
		ApplicationConfiguration(const std::string& fileName);
		/// <summary>
		/// Returns the Mode of the operation
		/// </summary>
		/// <returns></returns>
		std::string getMode() const;
		/// <summary>
		/// Returns the thread block size
		/// </summary>
		/// <returns>The size of the bloack size</returns>
		size_t getThreadBlockSize() const;
	};

	template<typename Object>
	Configuration<Object>::Configuration(const std::string&  fileName) {

		Construct(filename);

	}

	template<typename Object>
	void Configuration<Object>::Construct(const std::string&  fileName) {
		std::ifstream file(fileName.c_str());
		std::string line;
		std::string name;
		std::string value;
		std::string inSection;
		size_t posEqual;

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
	Object Configuration<Object>::Value(const std::string& section, const std::string& entry) const {

		auto iterator = this->m_content.find(section + '/' + entry);
		if (iterator == this->m_content.end()) {
			throw new std::exception("configuration entry not found");
		}
		return iterator->second;
	}

	template<typename Object>
	Configuration<Object>::~Configuration() {

	}

}