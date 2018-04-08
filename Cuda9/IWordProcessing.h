#pragma once
#include "SequenceModel.h"
#include "includes.h"

namespace gpuNN {
	/// <summary>
	/// Based class for each word processor 
	/// </summary>
	class IWordProcessing {

	public:
		/// <summary>
		/// Build the model based on the filename with sequence 
		/// </summary>
		/// <returns>The model of sequences</returns>
		virtual SequenceModel BuildModel(const std::string& filename) = 0;
		/// <summary>
		/// Build the models based on a given 
		/// </summary>
		/// <param name="words"></param>
		/// <returns></returns>
		virtual SequenceModel BuildModel(vStrings& words) = 0;
	};
}