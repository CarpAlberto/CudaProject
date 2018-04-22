#ifdef HUFFMAN
#pragma once
#include "IWordProcessing.h"
#include "SequenceModel.h"

namespace gpuNN {

	class HuffmanWordProcessor :
		public IWordProcessing
	{
	public:
		virtual ISequenceModel& BuildModel(const std::string& filename);
		virtual ISequenceModel& BuildModel(vStrings& words);
		HuffmanWordProcessor();
		~HuffmanWordProcessor();
	};
}


#endif
