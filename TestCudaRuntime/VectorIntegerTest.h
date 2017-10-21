#pragma once
#include "BaseTest.h"
#include <vector.h>
#include <iostream>
using namespace TestProject;

class VectorIntegerTest : public BaseTest 
{

public:
	void TestConstructor_Int_Int() 
	{
		try 
		{
			gpuNN::VectorInteger vInteger(2, 3);
			AssertEqual(vInteger.getData()[0], 2);
			AssertEqual(vInteger.getData()[1], 3);
			std::cout << "Test Succeeded" << std::endl;
		}
		catch (TestException* exc) {
			    std::cout << exc->what();
		}
	}
};