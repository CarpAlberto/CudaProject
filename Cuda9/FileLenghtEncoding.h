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

		std::string databaseInstruction = ApplicationContext::instance()->
			getConfiguration()->getDatabaseInstruction();
	public:
		// Unused
		virtual ISequenceModel& BuildModel(const std::string& filename) { 
			return *new StringModel(); }

		virtual ISequenceModel& BuildModel(vStrings& words);
		
		// used for FSE
		virtual void CreateAndAppendModel(vStrings& words);
		/// <summary>
		/// Srtores the data into the given database
		/// </summary>
		/// <param name="database">The given database</param>
		virtual void OnFinish(const std::string& database,int size);

		vDouble MatchAgainst(const vStrings& words);

		FileLenghtEncoding();
		
		~FileLenghtEncoding();
		
		void Load(bool instructionMode);

	private:
		bool m_Initialized = false;
	};
}

