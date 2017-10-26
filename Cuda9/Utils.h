#pragma once
#include "includes.h"
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
		/// Generate random number
		/// </summary>
		/// <returns></returns>
		static double generateRandom() {
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<> dis(-0.3 , 1);
			return dis(gen);
		}
	};
}

