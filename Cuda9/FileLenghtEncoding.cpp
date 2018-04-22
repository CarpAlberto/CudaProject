#include "FileLenghtEncoding.h"
#include <map>
#include <algorithm>
using namespace gpuNN;

namespace gpuNN
{
	FileLenghtEncoding::FileLenghtEncoding()
	{
		//load
		Load(this->directoryBase + "/" + this->database);
	}

	FileLenghtEncoding::~FileLenghtEncoding()
	{
	}

	void FileLenghtEncoding::CreateAndAppendModel(vStrings& words)
	{
		for (auto& word : words) {
			
			auto& foundWord = std::find_if(this->arrayFrequency.begin(),
				this->arrayFrequency.end(),
				[word](const std::pair<std::string,int>& rhs) {
				return word == rhs.first;
			});
			if (foundWord != this->arrayFrequency.end()) {
				foundWord->second++;
			}
			else {
				this->arrayFrequency.emplace_back(std::make_pair(word,1));
			}
		}
	}

	void FileLenghtEncoding::Load(const std::string& filename)
	{
		std::ifstream is(filename);
		std::string line;
		while (std::getline(is, line))
		{
			std::istringstream iss(line);
			std::string a;
			int b;
			if (!(iss >> a >> b)) 
			{
				break; 
			}
			this->arrayFrequency.emplace_back(std::make_pair(a, b));
		}
	}

	void FileLenghtEncoding::OnFinish(const std::string& database)
	{	

		std::sort(this->arrayFrequency.begin(), this->arrayFrequency.end(), 
			[=](std::pair<std::string, int>& a, std::pair<std::string,int>& b) {
			return a.second > b.second;
		});

		std::ofstream os(database);
		for (auto & word : this->arrayFrequency) {
			os << word.first << " " << word.second << std::endl;
		}
		os.close();
	}

	ISequenceModel& FileLenghtEncoding::BuildModel(vStrings& words)
	{
		vDouble wordsBooleans(words.size());
		int i = 0;
		for (auto & iterator : this->arrayFrequency) {
			auto& word = iterator.first;
			if (std::find_if(words.begin(), words.end(), [&word](const std::string& rhs) {
				return word == rhs;
			}) == words.end()) {
				wordsBooleans.push_back(0);
			}
			else{
				wordsBooleans.push_back(1);
			}
			i++;
		}
		ISequenceModel& model = *new StringModel(wordsBooleans);
		return model;
	}
}