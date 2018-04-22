#pragma once
#include "ISequenceModel.h"
#include "includes.h"
#include <string>

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
		virtual ISequenceModel& BuildModel(const std::string& filename) = 0;
		/// <summary>
		/// Build the models based on a given 
		/// </summary>
		/// <param name="words"></param>
		/// <returns></returns>
		virtual ISequenceModel& BuildModel(vStrings& words) = 0;

		virtual void CreateAndAppendModel(vStrings& words) = 0;

		virtual void OnFinish(const std::string& filename) = 0;
	};
}