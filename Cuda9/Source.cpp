#include "OptimizedNeuralNetwork.h"
#include "ApplicationManager.h"
#include "GeneticAlgorithm.h"


char* dummy_args[] = { "--generate-feature","-f","xmrig-amd.exe","-out","test.bin", NULL };
char* dummy_args2[] = { "--config","config.cfg", NULL };


int main(int argc,char* argv[])
{
	argv = dummy_args2;
	argc = sizeof(dummy_args2) / sizeof(dummy_args2[0]) - 1;
	ApplicationManager manager;
	//manager.TrainFromDirectory();
	//manager.LoadFromFile("database.txt");
	//manager.Parse(argv, argc);

	
	Population Pop;

	Objective obj;
	obj.parametersInput = { 50 ,0.15f,0.002f }; // NeuronsLayer  RMS Threeshold
	obj.parametersOutput = { 0.9f }; // Accuracy
	srand(static_cast <unsigned> (time(0)));


	for (int i = 0; i < Pop.members.size(); i++) {
		Pop.members.at(i).parameters.resize(obj.parametersInput.size());

		Pop.members.at(i).parameters.at(0) = (int)Utils::randBetween(45,65);
		Pop.members.at(i).parameters.at(1) = Utils::randBetween(0.001, 0.3);
		Pop.members.at(i).parameters.at(2) = Utils::randBetween(0.0014,0.005);
		Pop.members.at(i).Fitness = 0;
	}

	MalwareAnalyzer malware;

	GeneticAlgorithm alg(Pop,obj,malware);
	alg.run();

}