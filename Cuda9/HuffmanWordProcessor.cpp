#ifdef HUFFMAN
#include "HuffmanWordProcessor.h"

using namespace gpuNN;

HuffmanWordProcessor::HuffmanWordProcessor()
{
}

HuffmanWordProcessor::~HuffmanWordProcessor()
{
}

ISequenceModel& HuffmanWordProcessor::BuildModel(const std::string& filename)
{
	ISequenceModel m(filename);
	return std::move(m);
}

ISequenceModel& HuffmanWordProcessor::BuildModel(vStrings& words)
{
	SequenceModel m(words);
	return std::move(m);
}

#endif