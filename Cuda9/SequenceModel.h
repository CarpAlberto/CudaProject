#pragma once
#include "includes.h"
#include "vector.h"
#include "ISequenceModel.h"
namespace gpuNN {

	class SequenceModel : ISequenceModel
	{
	protected:
		const int chunkArray = 200;
		const int nWorkers = 4;
		using Sentence = Word2Vec<std::string>::Sentence;
		using SentenceP = Word2Vec<std::string>::SentenceP;
		Word2Vec<std::string> internalModel;
	public:
		/// <summary>
		/// Build the sequence model based on the input filename
		/// </summary>
		/// <param name="filename">The input filename</param>
		SequenceModel(const std::string& filename);
		/// <summary>
		/// Returns an array of doubles based on the codes 
		/// </summary>
		/// <returns></returns>
		vDouble toVector();
		/// <summary>
		/// Construct the object based on the array of object
		/// </summary>
		/// <param name="vArray"></param>
		SequenceModel(vStrings& vArray);

		SequenceModel() = default;
		~SequenceModel();
	};
}
