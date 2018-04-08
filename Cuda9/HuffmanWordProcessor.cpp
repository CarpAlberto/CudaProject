#include "HuffmanWordProcessor.h"

using namespace gpuNN;

HuffmanWordProcessor::HuffmanWordProcessor()
{
}

HuffmanWordProcessor::~HuffmanWordProcessor()
{
}

SequenceModel HuffmanWordProcessor::BuildModel(const std::string& filename)
{
	SequenceModel m(filename);
	return std::move(m);
}

SequenceModel HuffmanWordProcessor::BuildModel(vStrings& words)
{
	SequenceModel m(words);
	return std::move(m);
}