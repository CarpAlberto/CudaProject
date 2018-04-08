#include "TestMatrix.h"
#include "TestNeuralNetwork.h"
#include "TestConfiguration.h"
#include "TestPeHeader.h"
#include "TestWordEncoding.h"
#include "settings.h"
#include "base.h"


int main() 
{
		
	TestContainer t;
	
	/*Add the tests*/
	
	//t.add("Test Constructor Default",&TestMatrix::TestConstructor_Default);
	//t.add("Test Constructor Default Gpu ",&TestMatrix::TestConstructor_Default_Gpu);
	//t.add("Test Constructor Int",&TestMatrix::TestConstructor_Int);
	//t.add("Test Constructor Int Gpu", &TestMatrix::TestConstructor_Int_Gpu);
	//t.add("Test Constructor Set",&TestMatrix::TestConstuctor_Set);
	//t.add("Test Constructor Set Gpu", &TestMatrix::TestConstuctor_Set_Gpus);
	//t.add("Test Constructor Copy Cpu",&TestMatrix::TestCopyConstructor);
	//t.add("Test Constructor Copy Gpu", &TestMatrix::TestCopyConstructor_GPU);
	//t.add("Test matrix transpose Cpu", &TestMatrix::TestTransposeMatrixCpu);
	//t.add("Test matrix transpose Gpu", &TestMatrix::TestTransposeMatrixGpu);
	//t.add("Test Training Network",&TestNeuralNetwork::TestTrain);
	//t.add("Test Training Network", &TestNeuralNetwork::TestIterativeNetwork);
    // t.add("Test Load Config", &TestConfiguration::TestLoadConfig);
	//t.add("Test sum gpu", &TestMatrix::TestSumMatrixGpu);
	//t.add("Test product gpu", &TestMatrix::TestProductMatrixGpu);
	//t.add("Test product cpu", &TestMatrix::TestProductMatrixCpu);
	//t.add("Test load pe", &TestPE::TestLoadExecutable);
	//t.add("Test Train words", &TestWordEncoding::TestTrainWords);
	t.add("Test Application", &TestWordEncoding::TestApplication);
	t.run(LaunchWithBenchmark::WithClock);

	system("pause");

}