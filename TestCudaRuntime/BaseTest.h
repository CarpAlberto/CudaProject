#pragma once
#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>
#include <functional>
#include <vector>
#include <chrono>
#include <string>
#include <map>

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
		std::map<std::string,std::function<void()>> functions;
	public:
		template<typename Function>
		void add(Function && function) {
			functions.insert(std::make_pair<>("",std::forward<Function>(function)));
		}
		template<typename Function>
		void add(std::string description, Function  && function) {
			functions.insert(std::make_pair<>(description, std::forward<Function>(function)));
		}
		virtual void run(LaunchWithBenchmark clock = LaunchWithBenchmark::NoClock) {
			try
			{
				for (auto && fnMap : this->functions) {
					std::string description = fnMap.first;
					auto fn = fnMap.second;
					if (clock == LaunchWithBenchmark::WithClock) {
						std::cout << "Running Test " << description << std::endl;
						auto start = std::chrono::high_resolution_clock::now();
						fn();
						auto end = std::chrono::high_resolution_clock::now();
						duration<double> time_span = duration_cast<duration<double>>(end- start);
						std::cout << "Time Elapsed : " << (time_span).count() << "\n";
					}
					else {
						fn();
					}
					std::cout << "Test succeeded\n" << std::endl;
				}
			}
			catch (TestException* exc) {
				std::cout << "Test Failed" << std::endl;
			}
		}
	};
}