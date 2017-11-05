#pragma once
#include "BaseTest.h"
#include <vector.h>
#include <iostream>
#include <matrix.h>
using namespace TestProject;


class TestConfiguration : BaseTest {

public:
	static void TestLoadConfig() 
	{
		auto context = ApplicationContext::instance();
		auto config = context->getConfiguration();
		auto mode = config->Value("GeneralSettings", "MODE");
		AssertEqual(mode, "CPU");

	}
};