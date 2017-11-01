#include "TestMatrix.h"
#include "TestNeuralNetwork.h"
#include "TestConfiguration.h"

int main() 
{
		
	TestContainer t;
	
	/*Add the tests*/

	t.add("Test Constructor Default",&TestMatrix::TestConstructor_Default);
	t.add("Test Constructor Default Gpu ",&TestMatrix::TestConstructor_Default_Gpu);
	t.add("Test Constructor Int",&TestMatrix::TestConstructor_Int);
	t.add("Test Constructor Int Gpu", &TestMatrix::TestConstructor_Int_Gpu);
	t.add("Test Constructor Set",&TestMatrix::TestConstuctor_Set);
	t.add("Test Constructor Set Gpu", &TestMatrix::TestConstuctor_Set_Gpus);
	t.add("Test Constructor Copy Cpu",&TestMatrix::TestCopyConstructor);
	t.add("Test Constructor Copy Cpu", &TestMatrix::TestCopyConstructor_GPU);
	t.add("Test Training Network",&TestNeuralNetwork::TestTrain);
    t.add("Test Load Config", &TestConfiguration::TestLoadConfig);


	t.run(LaunchWithBenchmark::WithClock);

	system("pause");

}