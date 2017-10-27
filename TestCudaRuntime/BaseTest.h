#pragma once
#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>
#include <functional>
#include <vector>
#include <chrono>

using namespace std::chrono;



namespace TestProject {

	class TestException : public std::exception {
	public:
		TestException(const std::string& rhs)  :
			std::exception(rhs.c_str())
		{
			
		}
	};
	
	enum class LaunchWithBenchmark {
		WithClock,
		NoClock
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
		virtual void run(LaunchWithBenchmark clock = LaunchWithBenchmark::NoClock) {
			try
			{
				for (auto && fn : this->functions) {
					if (clock == LaunchWithBenchmark::WithClock) {
						auto start = std::chrono::high_resolution_clock::now();
						fn();
						auto end = std::chrono::high_resolution_clock::now();

						duration<double> time_span = duration_cast<duration<double>>(end- start);


						std::cout << "\nTime Elapsed : " << (time_span).count() << "\n";
					}
					else {
						fn();
					}
					std::cout << "Test succeeded" << std::endl;
				}
			}
			catch (TestException* exc) {
				std::cout << "Test Failed" << std::endl;
			}
		}
	};
}