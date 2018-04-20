#include "OptimizedNeuralNetwork.h"
#include "ApplicationManager.h"


char* dummy_args[] = { "--generate-feature","-f","xmrig-amd.exe","-out","test.bin", NULL };
char* dummy_args2[] = { "--config","config.cfg", NULL };


int main(int argc,char* argv[])
{

	/*
	RealHostMatrix input(1, 3);
	
	RealHostMatrix outputs(1, 3);

	input(0, 0) = 1;
	input(0, 1) = 0;
	input(0, 2) = 1;

	outputs(0, 0) = 0;
	outputs(0, 1) = 1;
	outputs(0, 2) = 1;

	TopologyOptimized topology(4);
	topology[0] = 3;
	topology[1] = 30;
	topology[2] = 10;
	topology[3] = 1;

	OptimizedNeuralNetwork n(topology, input, outputs, 0.7);
	n.Train(2);

	*/

	argv = dummy_args;
	argc = sizeof(dummy_args) / sizeof(dummy_args[0]) - 1;


	ApplicationManager manager;
	manager.Parse(argv, argc);
}