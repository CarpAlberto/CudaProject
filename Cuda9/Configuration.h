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
		/// <summary>
		/// Returns trus if the directory mode is enabled
		/// </summary>
		/// <returns></returns>
		bool isDirectoryModeEnabled();
		/// <summary>
		/// Enables the generation features based on sources
		/// </summary>
		/// <returns></returns>
		bool isFilenameModeEnabled();
		/// <summary>
		/// Returns the direcotry benings folder
		/// </summary>
		/// <returns></returns>
		std::string getDirectoryBenigns();
		/// <summary>
		/// Returns the malware directory
		/// </summary>
		/// <returns></returns>
		std::string getDirectoryMalware();
		/// <summary>
		/// Returns the filename in
		/// </summary>
		/// <returns></returns>
		std::string getFilenameIn();
		/// <summary>
		/// Returns the filename out
		/// </summary>
		/// <returns></returns>
		std::string getFilenameOut();

		bool isStringLengthEncoding();

		std::string getDirectoryBase();

		std::string getDatabaseOut();

		std::string getTrainBenignsDirectory();

		/// <summary>
		/// True if the mode is enabled
		/// </summary>
		/// <returns></returns>
		bool isTrainingModeEnabled();

		bool isGenerateDataMode();
		/// <summary>
		/// Returns the root mean square min.The ANN
		/// stops if the RMS fails
		/// </summary>
		/// <returns></returns>
		float getRootMeanSquareMin();
		/// <summary>
		/// Returns the max numbers of epocks
		/// </summary>
		/// <returns></returns>
		int getEpocksLimit();

		std::string getTrainDirectory();

		std::string getTestDirectory();

		bool isTestMode();

		std::string getDatabaseInstruction(); //INSTR_DATABASE_OUT

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