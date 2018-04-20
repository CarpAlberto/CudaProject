#include "ApplicationManager.h"

using namespace gpuNN;

ApplicationManager::ApplicationManager()
{
	this->analyzer = new MalwareAnalyzer();

	//this->analyzer->BuildFeatures("xmrig-amd.exe", this->neuralNetwork);

   //this->analyzer->Analyze(this->neuralNetwork);

	//this->neuralNetwork->Save("trained.txt", IOStrategy::ASCII);

}


ApplicationManager::~ApplicationManager()
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
	
	auto config = this->instance->getConfiguration();
	bool isDirectoryMode = config->isDirectoryModeEnabled();
	bool isFilenameMode = config->isFilenameModeEnabled();

	if (isDirectoryMode) {

	}
	if (isFilenameMode) {
		auto in = config->getFilenameIn();
		auto out = config->getFilenameOut();
		this->analyzer->BuildFeatures(in, out);
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
		if (strcmp(item, "-config") == 0) {
			i++;
			ParseConfig();
		}
	}
}