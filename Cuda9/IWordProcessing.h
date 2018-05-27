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

		/// <summary>
		/// Creates the model and append the words to itself
		/// </summary>
		/// <param name="words"></param>
		virtual void CreateAndAppendModel(vStrings& words) = 0;

		/// <summary>
		/// Handles the Finish event
		/// </summary>
		/// <param name="filename">The filename where the database will be stored</param>
		virtual void OnFinish(const std::string& filename,int size) = 0;

		/// <summary>
		/// Match a given list against a list of words
		/// </summary>
		/// <param name="words">The list of words</param>
		/// <returns>The array Doubles</returns>
		virtual vDouble MatchAgainst(const vStrings& words)=0;
	};
}