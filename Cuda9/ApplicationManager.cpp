#include "ApplicationManager.h"
using namespace gpuNN;

ApplicationManager::ApplicationManager(){

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
	bool isTrainingMode = config->isTrainingModeEnabled();
	bool isGenerateDataMode = config->isGenerateDataMode();
	bool isFilenameMode = config->isFilenameModeEnabled();
	bool isStringLengthEncodedEnabled = config->isStringLengthEncoding();
	bool isTestMode = config->isTestMode();
	auto directory = config->getDirectoryBase();
	auto trainDirectory = directory + config->getTrainDirectory();
	auto directoryTrainBenings = trainDirectory + "\\" + config->getDirectoryBenigns();
	auto directoryTrainMalware = trainDirectory + "\\" + config->getDirectoryMalware();
	auto database = directory + config->getDatabaseOut();
	auto directoryTest = directory + config->getTestDirectory();
	auto directoryTestBenigns = directoryTest + "\\" + config->getDirectoryBenigns();
	auto directoryTestMalware = directoryTest + "\\" + config->getDirectoryMalware();
	std::vector<std::string> vArray;
	
	
	if (isStringLengthEncodedEnabled){
		// Directory Mode
		if (isDirectoryMode){
			this->analyzer->BuildFeaturesFromDirectory(directoryTrainBenings, database);
			this->analyzer->BuildFeaturesFromDirectory(directoryTrainMalware, database);
		}
		// Training Mode
		if (isTrainingMode) {
			this->analyzer->TrainNeuralNetwork(trainDirectory + "\\benigns.txt","network_benigns.txt",true);
			//this->analyzer->TrainNeuralNetwork(trainDirectory + "\\malware.txt", "network_malware.txt", true);
		}

		if (isGenerateDataMode){
			// Build the data 
			this->analyzer->BuildDataFromDirectory(directoryTrainBenings,
				trainDirectory + "\\benigns.txt");
			this->analyzer->BuildDataFromDirectory(directoryTrainMalware,
				trainDirectory + "\\malware.txt");

			// Build the test data
			this->analyzer->BuildDataFromDirectory(directoryTestBenigns,
				directoryTest + "\\benigns.txt");
			this->analyzer->BuildDataFromDirectory(directoryTestMalware,
				directoryTest + "\\malware.txt");
		}

		if (isTestMode) {
			this->analyzer->TestNeuralNetwork(directoryTest + "\\benigns.txt",
				"network_benigns.txt");
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

void ApplicationManager::TrainFromDirectory(bool viruses)
{
	auto config = this->getContext()->getConfiguration();
	bool isTrainingMode = config->isTrainingModeEnabled();
	std::vector<std::string> vArray;
	auto directory = config->getDirectoryBase();
	auto trainingBenignDirectory = directory + config->getTrainBenignsDirectory();
	auto trainingVirusesDirectory = directory + config->getDirectoryMalware();
	
	if(viruses)
		Utils::ReadDirectory(trainingVirusesDirectory, vArray);
	else
		Utils::ReadDirectory(trainingBenignDirectory, vArray);

	for (auto filename : vArray) 
	{
		if (filename == "." || filename == "..") continue;
		auto file = trainingBenignDirectory + "/" + filename;
		this->analyzer->BuildFeatures(file, this->neuralNetwork,true);
		if(this->neuralNetwork != nullptr)
			this->analyzer->Analyze(this->neuralNetwork);
	}
	this->neuralNetwork->Save("database.txt", IOStrategy::ASCII);
}

void ApplicationManager::LoadFromFile(const std::string& filename)
{
	std::ifstream is(filename);
	std::vector<int> layers;
	std::string line;
	std::getline(is,line);
	auto Topology = Utils::Split(line, '-');
	CpuArray<int> vTopology;
	vTopology.Resize(Topology.size());
	int i = 0;
	for (auto& item :  Topology) {
		vTopology[i] = (stoi(item));
		i++;
	}
	RealHostMatrix input(1, stoi(Topology[0]));

	for (auto i = 0; i < Topology.size(); i++) {
		input(0, i) = 0;
	}

	RealHostMatrix outputs(1, 1);
	outputs(0, 0) = 1.0f;

	auto neuralNetwork = new OptimizedNeuralNetwork(vTopology, input, outputs, 0.7,0.001);

	float elm;
	for (auto i = 0; i < vTopology.Size() - 1; i++) {
		int weights = (vTopology[i] + 1) * vTopology[i + 1];
		RealCpuArray array;
		array.Resize(weights);
		for (int j = 0; j < weights; j++) {
			std::getline(is, line);
			elm = stof(line);
			array[j] = elm;
		}
		neuralNetwork->SetWeights(array,i);
	}
}