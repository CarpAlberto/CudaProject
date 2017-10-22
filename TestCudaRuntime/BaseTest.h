#pragma once
#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>
#include <functional>
#include <vector>



namespace TestProject {

	class TestException : public std::exception {
	public:
		TestException(const std::string& rhs)  :
			std::exception(rhs.c_str())
		{
			
		}
	};
	
	class BaseTest {

	protected:
		template<typename Left,typename Right>
			static void AssertEqual(Left left,Right right) {
				if (left == right) {
					// do nothin
				}
				else {
					throw new TestException("Test Failed");
				}
		}
		template<typename Left, typename Right>
			static void AssertNotEqual(Left left,Right right) {
				return left != right;
			}
	};

	class TestContainer {
		
	protected:
		std::vector<std::function<void()>> functions;

	public:
		template<typename Function>
		void add(Function && function) {
			functions.push_back(std::forward<Function>(function));
		}
		virtual void run() {
			try
			{
				for (auto && fn : this->functions) {
					fn();
					std::cout << "Test succeeded" << std::endl;
				}
			}
			catch (TestException* exc) {
				std::cout << "Test Failed" << std::endl;
			}
		}
	};
}