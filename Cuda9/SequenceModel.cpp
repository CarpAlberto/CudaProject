#include "SequenceModel.h"

using namespace gpuNN;

SequenceModel::SequenceModel(const std::string& filename)
{
	std::vector<SentenceP> sentences;
	size_t count = 0;
	const size_t max_sentence_len = this->chunkArray;
	SentenceP sentence(new Sentence);
	std::ifstream in(filename);
	while (true) {
		std::string s;
		in >> s;
		if (s.empty()) 
			break;
		++count;
		sentence->tokens_.push_back(std::move(s));
		if (count == this->chunkArray) {
			count = 0;
			sentence->words_.reserve(sentence->tokens_.size());
			sentences.push_back(std::move(sentence));
			sentence.reset(new Sentence);
		}
	}
	if (!sentence->tokens_.empty())
		sentences.push_back(std::move(sentence));
	this->internalModel.build_vocab(sentences);
}


SequenceModel::~SequenceModel()
{
}

vDouble SequenceModel::toVector()
{
	auto internalSequence = this->internalModel;
	auto words = internalSequence.words_;
	vDouble bArray;

	for (auto& word : words) {
		for (auto& code : word->codes_) {
			bArray.emplace_back(code);
		}
	}
	return std::move(bArray);
}

SequenceModel::SequenceModel(vStrings& vArray)
{
	size_t count = 0;
	std::vector<SentenceP> sentences;
	SentenceP sentence(new Sentence);
	for (auto& s : vArray) {
		if (s.empty())
			break;
		++count;
		sentence->tokens_.push_back(std::move(s));
		if (count == this->chunkArray) {
			count = 0;
			sentence->words_.reserve(sentence->tokens_.size());
			sentences.push_back(std::move(sentence));
			sentence.reset(new Sentence);
		}
	}
	if (!sentence->tokens_.empty())
		sentences.push_back(std::move(sentence));
	this->internalModel.build_vocab(sentences);
}