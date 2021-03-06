#pragma once
#include "BaseTest.h"
#include <vector.h>
#include <MatrixFactory.h>
#include <iostream>
#include <matrix.h>
using namespace TestProject;

class TestMatrix : public BaseTest {

public:

	static void TestConstructor_Default() 
	{

		GenericMatrix* matrix = new CpuMatrix();
		AssertEqual(matrix->getCols(), 0);
		AssertEqual(matrix->getRows(), 0);
		AssertEqual(matrix->getData(), nullptr);

		delete matrix;
	}
	
	static void TestConstructor_Default_Gpu() {

		GenericMatrix* matrix = new GpuMatrix();
		AssertEqual(matrix->getCols(), 0);
		AssertEqual(matrix->getRows(), 0);
		AssertEqual(matrix->getData(), nullptr);

		delete matrix;
	}

	static void TestConstructor_Int() {

		GenericMatrix* matrix = new CpuMatrix(10,10);
		AssertEqual(matrix->getCols(), 10);
		AssertEqual(matrix->getRows(), 10);
		delete matrix;
	}

	static void TestConstructor_Int_Gpu() {

		GenericMatrix* matrix = new GpuMatrix(10, 10);
		AssertEqual(matrix->getCols(), 10);
		AssertEqual(matrix->getRows(), 10);
		delete matrix;
	}
	
	static void TestConstuctor_Set() {

		GenericMatrix* matrix = new CpuMatrix(3, 3);
		AssertEqual(matrix->getCols(), 3);
		AssertEqual(matrix->getRows(), 3);

		matrix->Set(0,0,2);
		matrix->Set(0,1,1);
		matrix->Set(0,2,3);
		
		AssertEqual(matrix->Get(0,1), 1);
		AssertEqual(matrix->Get(0,0), 2);
		AssertEqual(matrix->Get(0,2), 3);

		delete matrix;
	}

	static void TestConstuctor_Set_Gpus() {

		GenericMatrix* matrix = new GpuMatrix(3, 3);
		AssertEqual(matrix->getCols(), 3);
		AssertEqual(matrix->getRows(), 3);

		matrix->Set(0, 0, 2);
		matrix->Set(0, 1, 1);
		matrix->Set(0, 2, 3);

		AssertEqual(matrix->Get(0, 1), 1);
		AssertEqual(matrix->Get(0, 0), 2);
		AssertEqual(matrix->Get(0, 2), 3);

		delete matrix;
	}
	
	static void TestCopyConstructor() {


		GenericMatrix* matrix = new CpuMatrix(3, 3);
		AssertEqual(matrix->getCols(), 3);
		AssertEqual(matrix->getRows(), 3);

		matrix->Set(0, 0, 2);
		matrix->Set(0, 1, 1);
		matrix->Set(0, 2, 3);

		GenericMatrix* copy = new CpuMatrix(*matrix);

		AssertEqual(copy->Get(0, 1), 1);
		AssertEqual(copy->Get(0, 0), 2);
		AssertEqual(copy->Get(0, 2), 3);
		delete matrix;
		delete copy;
	}

	static void TestCopyConstructor_GPU() {


    	GenericMatrix* matrix = new GpuMatrix(3, 3);

		AssertEqual(matrix->getCols(), 3);
		AssertEqual(matrix->getRows(), 3);


		matrix->Set(0, 0, 2);
		matrix->Set(0, 1, 1);
		matrix->Set(0, 2, 3);

		GenericMatrix* copy = new GpuMatrix(*matrix);

		AssertEqual(copy->Get(0, 1), 1);
		AssertEqual(copy->Get(0, 0), 2);
		AssertEqual(copy->Get(0, 2), 3);
		delete matrix;
		delete copy;
	}

	static void TestSumMatrixGpu() {

		GenericMatrix*  matrix = MatrixFactory::getMatrix(2, 2);
		GenericMatrix*  matrix2 = MatrixFactory::getMatrix(2, 2);

		matrix->Set(0, 0, 1);
		matrix->Set(0, 1, 1);
		matrix->Set(1, 0, 1);
		matrix->Set(1, 1, 1);

		matrix2->Set(0, 0, 3);
		matrix2->Set(0, 1, 3);
		matrix2->Set(1, 0, 3);
		matrix2->Set(1, 1, 3);

		GenericMatrix& sum = (*matrix) + (*matrix2);

		AssertEqual(sum.Get(0, 0), 4);
		AssertEqual(sum.Get(0, 1), 4);
		AssertEqual(sum.Get(1, 0), 4);
		AssertEqual(sum.Get(1, 1), 4);


		delete matrix;
		delete matrix2;
		delete &sum;
	}

	static void TestProductMatrixCpu() {

		GenericMatrix*  matrix = MatrixFactory::getMatrix(4, 3);
		GenericMatrix*  matrix2 = MatrixFactory::getMatrix(3, 2);

		matrix->Set(0, 0, 1);
		matrix->Set(0, 1, 1);
		matrix->Set(0, 2, 1);
		
		matrix->Set(1, 0, 1);
		matrix->Set(1, 1, 1);
		matrix->Set(1, 2, 1);

		matrix->Set(2, 0, 2);
		matrix->Set(2, 1, 2);
		matrix->Set(2, 2, 2);

		matrix->Set(3, 0, 2);
		matrix->Set(3, 1, 2);
		matrix->Set(3, 2, 2);

		matrix2->Set(0, 0, 2);
		matrix2->Set(0, 1, 2);
		matrix2->Set(1, 0, 2);
		matrix2->Set(1, 1, 2);
		matrix2->Set(2, 0, 2);
		matrix2->Set(2, 1, 2);


		GenericMatrix& sum = (*matrix) * (*matrix2);

		sum.Print(ApplicationContext::instance()->getGUI().get());

		AssertEqual(sum.Get(0, 0), 6);
		AssertEqual(sum.Get(0, 1), 6);
		AssertEqual(sum.Get(1, 0), 6);
		AssertEqual(sum.Get(1, 1), 6);
		AssertEqual(sum.Get(2, 0), 12);
		AssertEqual(sum.Get(2, 1), 12);
		AssertEqual(sum.Get(3, 0), 12);
		AssertEqual(sum.Get(3, 1), 12);
	}

	static void TestProductMatrixGpu() {
		
		GenericMatrix*  matrix = MatrixFactory::getMatrix(2, 2);
		GenericMatrix*  matrix2 = MatrixFactory::getMatrix(2, 2);

		matrix->Set(0, 0, 1);
		matrix->Set(0, 1, 1);
		matrix->Set(1, 0, 1);
		matrix->Set(1, 1, 1);

		matrix2->Set(0, 0, 2);
		matrix2->Set(0, 1, 2);
		matrix2->Set(1, 0, 2);
		matrix2->Set(1, 1, 2);

		GenericMatrix& sum = (*matrix2) * (*matrix);

		AssertEqual(sum.Get(0, 1), 4);
		AssertEqual(sum.Get(0, 0), 4);
		AssertEqual(sum.Get(1, 0), 4);
		AssertEqual(sum.Get(1, 1), 4);
	}

	static void TestTransposeMatrixCpu()
	{
		GenericMatrix*  matrix = new CpuMatrix(2, 2);

		matrix->Set(0, 0, 2);
		matrix->Set(0, 1, 2);
		matrix->Set(1, 0, 1);
		matrix->Set(1, 1, 1);

		auto& transpose = matrix->Transpose();

		AssertEqual(transpose.Get(0, 0), 2);
		AssertEqual(transpose.Get(0, 1), 1);
		AssertEqual(transpose.Get(1, 0), 2);
		AssertEqual(transpose.Get(1, 1), 1);
	}

	static void TestTransposeMatrixGpu()
	{
		GenericMatrix*  matrix = new GpuMatrix(2, 2);

		matrix->Set(0, 0, 2);
		matrix->Set(0, 1, 2);
		matrix->Set(1, 0, 1);
		matrix->Set(1, 1, 1);

		auto& transpose = matrix->Transpose();

		AssertEqual(transpose.Get(0, 0), 2);
		AssertEqual(transpose.Get(0, 1), 1);
		AssertEqual(transpose.Get(1, 0), 2);
		AssertEqual(transpose.Get(1, 1), 1);
	}

};