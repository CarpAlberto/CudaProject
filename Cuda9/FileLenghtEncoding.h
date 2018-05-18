#pragma once
#include "IWordProcessing.h"
#include "StringModel.h"
namespace gpuNN {

	class FileLenghtEncoding : public IWordProcessing
	{
	protected:
		std::vector<std::pair<std::string,int>> arrayFrequency;

		std::string directoryBase = ApplicationContext::instance()->
			getConfiguration()->getDirectoryBase();

		std::string database = ApplicationContext::instance()->
			getConfiguration()->getDatabaseOut();
	public:
		// Unused
		virtual ISequenceModel& BuildModel(const std::string& filename) { 
			return *new StringModel(); }
		// Unused
		virtual ISequenceModel& BuildModel(vStrings& words);
		// used for FSE
		virtual void CreateAndAppendModel(vStrings& words);
		/// <summary>
		/// Srtores the data into the given database
		/// </summary>
		/// <param name="database">The given database</param>
		virtual void OnFinish(const std::string& database);

		vDouble MatchAgainst(const vStrings& words);

		FileLenghtEncoding();
		~FileLenghtEncoding();
	protected:
		void Load(const std::string& filename);
	};
}

