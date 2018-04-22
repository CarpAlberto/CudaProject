#include "ApplicationManager.h"

using namespace gpuNN;

ApplicationManager::ApplicationManager()
{
	this->analyzer = new MalwareAnalyzer();
	this->neuralNetwork = nullptr;
	//this->analyzer->BuildFeatures("xmrig-amd.exe", this->neuralNetwork);

   //this->analyzer->Analyze(this->neuralNetwork);

	//this->neuralNetwork->Save("trained.txt", IOStrategy::ASCII);

}


ApplicationManager::~ApplicationManager()
{
}

void ApplicationManager::ParseTrain(char** argv, int argc)
{
}
void ApplicationManager::Parse(char** argv, int argc)
{
	ParseInternal(argv, argc);
}
void ApplicationManager::ParseGenerateFeaturesItem(char** argv, int argc, int argStart){
	for (auto i = argStart; i < argc; i++) {
		char* item = argv[i];
		if (strcmp(item, "-f") == 0) {
			ParseGenerateFeaturesFilename(argv, argc, i+1);
		}
	}
}
void ApplicationManager::ParseGenerateFeaturesFilename(char** argv, int argc, int argStart)
{
	if (argc < argStart)
		throw new std::exception("Invalid arguments");
	char* filename = argv[argStart];
	char* out;
	argStart++;
	for (auto i = argStart; i < argc; i++) {
		char* item = argv[i];
		// Need to be out
		if (strcmp(item, "-out") == 0) {
			i++;
			out = argv[i];
			this->analyzer->BuildFeatures(filename, out);
		}
	}
}

void ApplicationManager::ParseConfig(){
	
	auto config = this->getContext()->getConfiguration();
	bool isDirectoryMode = config->isDirectoryModeEnabled();
	bool isFilenameMode = config->isFilenameModeEnabled();
	bool isStringLengthEncodedEnabled = config->isStringLengthEncoding();
	std::vector<std::string> vArray;
	
	if (isStringLengthEncodedEnabled){
		if (isDirectoryMode){
			auto directory = config->getDirectoryBase();
			auto directoryBenings = directory +  config->getDirectoryBenigns();
			auto database = directory + config->getDatabaseOut();
			this->analyzer->BuildFeaturesFromDirectory(directoryBenings, database);
		}
	}
	else
	{
		// Huffman
		if (isDirectoryMode)
		{
			auto directoryBenings = config->getDirectoryBenigns();
			auto directoryBeningsOut = directoryBenings + "_" + "benings";
			if (CreateDirectory(directoryBeningsOut.c_str(), NULL) ||
				ERROR_ALREADY_EXISTS == GetLastError()) {
				Utils::ReadDirectory(directoryBenings, vArray);
				for (auto filename : vArray) {
					if (filename == "." || filename == "..")continue;
					auto fileOut = directoryBeningsOut + "/" + filename + "-" + "features.bin";
					filename = directoryBenings + "/" + filename;
					this->analyzer->BuildFeatures(filename, fileOut);
				}
			}
		}
		if (isFilenameMode)
		{
			auto in = config->getFilenameIn();
			auto out = config->getFilenameOut();
			this->analyzer->BuildFeatures(in, out);
		}
	}

}

void ApplicationManager::ParseInternal(char** argv, int argc)
{
	for (auto i = 0; i < argc; i++) {
		char* item = argv[i];
		if (strcmp(item, "--generate-feature") == 0) {
			i++;
			ParseGenerateFeaturesItem(argv, argc, i);
		}
		if (strcmp(item, "--config") == 0) {
			i++;
			ParseConfig();
		}
		if (strcmp(item, "--train") == 0) {
			i++;
			ParseTrain(argv,argc);
		}
	}
}

ApplicationContext* ApplicationManager::getContext()
{
	return this->instance->instance();
}

void ApplicationManager::ParseDirectory(const std::string& directory)
{
	std::vector<std::string> vArray;
	Utils::ReadDirectory(directory,vArray);


}

void ApplicationManager::TrainFromDirectory()
{
	auto config = this->getContext()->getConfiguration();
	bool isTrainingMode = config->isTrainingModeEnabled();
	std::vector<std::string> vArray;
	auto directory = config->getDirectoryBase();
	auto trainingBenignDirectory = directory + config->getTrainBenignsDirectory();

	Utils::ReadDirectory(trainingBenignDirectory, vArray);
	for (auto filename : vArray) {
		if (filename == "." || filename == "..")continue;
		
		auto file = trainingBenignDirectory + "/" + filename;
		this->analyzer->BuildFeatures(file, this->neuralNetwork);
		if(this->neuralNetwork != nullptr)
			this->analyzer->Analyze(this->neuralNetwork);
	}
	this->neuralNetwork->Save("database.txt", IOStrategy::ASCII);
}