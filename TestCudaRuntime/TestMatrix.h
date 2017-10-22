#pragma once
#pragma once
#include "BaseTest.h"
#include <vector.h>
#include <iostream>
#include <matrix.h>
using namespace TestProject;

class TestMatrix :
	public BaseTest {

public:

	static void TestConstructor_Default() {

		GenericMatrix* matrix = new CpuMatrix();
		AssertEqual(matrix->getCols(), 0);
		AssertEqual(matrix->getRows(), 0);
		AssertEqual(matrix->getData(), nullptr);

		delete matrix;
	}
	
	static void TestConstructor_Int() {

		GenericMatrix* matrix = new CpuMatrix(2,2,1);
		AssertEqual(matrix->getCols(), 2);
		AssertEqual(matrix->getRows(), 2);
		AssertEqual(matrix->getChannels(), 1);
		delete matrix;
	}
	
	static void TestConstuctor_Set() {

		GenericMatrix* matrix = new CpuMatrix(3, 3, 1);
		AssertEqual(matrix->getCols(), 3);
		AssertEqual(matrix->getRows(), 3);
		AssertEqual(matrix->getChannels(), 1);

		matrix->Set(0,0,0,2);
		matrix->Set(0,1,0,1);
		matrix->Set(0,2,0,3);
		
		AssertEqual(matrix->Get(0,1,0), 1);
		AssertEqual(matrix->Get(0,0,0), 2);
		AssertEqual(matrix->Get(0,2,0), 3);

		delete matrix;
	}

	static void TestCopyConstructor() {


		GenericMatrix* matrix = new CpuMatrix(3, 3, 1);
		AssertEqual(matrix->getCols(), 3);
		AssertEqual(matrix->getRows(), 3);
		AssertEqual(matrix->getChannels(), 1);

		matrix->Set(0, 0, 0, 2);
		matrix->Set(0, 1, 0, 1);
		matrix->Set(0, 2, 0, 3);

		GenericMatrix* copy = new CpuMatrix(*matrix);

		AssertEqual(copy->Get(0, 1, 0), 1);
		AssertEqual(copy->Get(0, 0, 0), 2);
		AssertEqual(copy->Get(0, 2, 0), 3);

		delete matrix;
		delete copy;

	}
};