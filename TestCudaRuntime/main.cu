#include "TestMatrix.h"
#include "TestNeuralNetwork.h"

int main() {
		
	TestContainer t;
	
	/*Add the tests*/
	t.add(&TestMatrix::TestConstructor_Default);
	t.add(&TestMatrix::TestConstructor_Int);
	t.add(&TestMatrix::TestConstuctor_Set);
	t.add(&TestMatrix::TestCopyConstructor);
	t.add(&TestNeuralNetwork::TestConstructor);

	t.run();

	system("pause");

}