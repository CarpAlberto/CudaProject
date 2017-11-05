#include "base.h"
#include "TestMatrix.h"
#include "TestNeuralNetwork.h"
#include "TestConfiguration.h"
#include "settings.h"

char* run_constructor_cpu() {

	TestContainer t;

	t.add("Test Constructor Default", &TestMatrix::TestConstructor_Default);
	t.add("Test Constructor Int", &TestMatrix::TestConstructor_Int);
	t.add("Test Constructor Set", &TestMatrix::TestConstuctor_Set);
	t.add("Test Constructor Copy Cpu", &TestMatrix::TestCopyConstructor);
	std::string result = t.runStream(LaunchWithBenchmark::WithClock);
	char* returnType = strdup(result.c_str());
	return returnType;
}

char* run_constructor_gpu() {

	TestContainer t;

	t.add("Test Constructor Default", &TestMatrix::TestConstructor_Default_Gpu);
	t.add("Test Constructor Int", &TestMatrix::TestConstructor_Int_Gpu);
	t.add("Test Constructor Set", &TestMatrix::TestConstuctor_Set_Gpus);
	t.add("Test Constructor Copy Cpu", &TestMatrix::TestCopyConstructor_GPU);
	std::string result = t.runStream(LaunchWithBenchmark::WithClock);
	char* returnType = strdup(result.c_str());
	return returnType;
}