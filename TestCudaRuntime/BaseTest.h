
#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>



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
}