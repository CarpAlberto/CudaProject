#pragma once
#include "IWordProcessing.h"

namespace gpuNN {

	class HuffmanWordProcessor :
		public IWordProcessing
	{
	public:
		virtual SequenceModel BuildModel(const std::string& filename);
		virtual SequenceModel BuildModel(vStrings& words);
		HuffmanWordProcessor();
		~HuffmanWordProcessor();
	};
}

