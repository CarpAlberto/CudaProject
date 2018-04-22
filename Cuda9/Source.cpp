#include "OptimizedNeuralNetwork.h"
#include "ApplicationManager.h"


char* dummy_args[] = { "--generate-feature","-f","xmrig-amd.exe","-out","test.bin", NULL };
char* dummy_args2[] = { "--config","config.cfg", NULL };


int main(int argc,char* argv[])
{


	argv = dummy_args2;
	argc = sizeof(dummy_args2) / sizeof(dummy_args2[0]) - 1;


	ApplicationManager manager;
	manager.TrainFromDirectory();
	//manager.Parse(argv, argc);
}