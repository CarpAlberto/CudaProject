#pragma once
#include "includes.h"
#include <sstream>
namespace gpuNN {
	class Utils
	{
	public:
		/// <summary>
		/// Calculates the necessary padding for baseAddress
		/// </summary>
		/// <param name="baseAddress">The base address in the memory</param>
		/// <param name="alignment">The alignament in bytes</param>
		/// <returns>The necessary padding</returns>
		static std::size_t CalculatePadding(const std::size_t baseAddress, const std::size_t alignment);
		/// <summary>
		/// Calculates the necessary padding 
		/// </summary>
		/// <param name="baseAddress">The base address </param>
		/// <param name="alignment"></param>
		/// <param name="headerSize">The size of the header </param>
		/// <returns></returns>
		static std::size_t CalculatePaddingWithHeader(const std::size_t baseAddress, const std::size_t alignment, const std::size_t headerSize);

		/// <summary>
		/// Given a string this method trims the data and returns the trimed 
		/// string
		/// </summary>
		/// <param name="source">The given string</param>
		/// <param name="delims">The given delims</param>
		/// <returns>The returned string</returns>
		static std::string Trim(std::string const& source, char const* delims = " \t\r\n");

		/// <summary>
		/// Generate random number
		/// </summary>
		/// <returns></returns>
		static double generateRandom() {
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<> dis(-0.3 , 1);
			return dis(gen);
		}
		/// <summary>
		/// Returns if the character is ASCII false otherwise
		/// </summary>
		/// <param name="ch">The character</param>
		/// <returns></returns>
		static bool isAscii(char ch);
		/// <summary>
		/// Splits the string and returns the vector of the tokens
		/// </summary>
		/// <param name="rhs">The strings to be splitted</param>
		/// <param name="delim">The delimiter</param>
		/// <returns></returns>
		static std::vector<std::string> Split(std::string rhs, char delim);

		/// <summary>
		/// Reads the directory content and put it in the array
		/// </summary>
		/// <param name="directory">The directory where the data should be fetched</param>
		/// <param name="array">The array where the data will be stored</param>
		static void ReadDirectory(const std::string directory, std::vector<std::string>& array);

		/// <summary>
		/// Returns the number of lines of a given stream 
		/// </summary>
		/// <returns></returns>
		static size_t getNumberLines(std::ifstream& stream);

		static double randBetween(double M, double N)
		{
			return M + (rand() / (RAND_MAX / (N - M)));
		}
	};
}

