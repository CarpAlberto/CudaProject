#pragma once
#include "BaseTest.h"
#include <vector.h>
#include <MatrixFactory.h>
#include <iostream>
#include <matrix.h>
#include <include/word-to-vec/word2vec.h>
#include <iostream>
#include <initializer_list>
#include <ApplicationManager.h>
using namespace TestProject;

class TestWordEncoding : public BaseTest {
	
public:
	static void TestTrainWords()
	{
		Word2Vec<std::string> model(200);
		using Sentence = Word2Vec<std::string>::Sentence;
		using SentenceP = Word2Vec<std::string>::SentenceP;
		int n_workers = 4;

		std::vector<SentenceP> sentences;
		size_t count = 0;
		const size_t max_sentence_len = 200;

		SentenceP sentence(new Sentence);
		std::ifstream in("text8");		
		while (true) {
			std::string s;
			in >> s;
			if (s.empty()) break;

			++count;
			sentence->tokens_.push_back(std::move(s));
			if (count == max_sentence_len) {
				count = 0;
				sentence->words_.reserve(sentence->tokens_.size());
				sentences.push_back(std::move(sentence));
				sentence.reset(new Sentence);
			}
		}

		if (!sentence->tokens_.empty())
			sentences.push_back(std::move(sentence));

		auto cstart = std::chrono::high_resolution_clock::now();
		model.build_vocab(sentences);
		auto cend = std::chrono::high_resolution_clock::now();
		printf("load vocab: %.4f seconds\n", std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000000.0);

		cstart = cend;
		model.train(sentences, n_workers);
		cend = std::chrono::high_resolution_clock::now();
		printf("train: %.4f seconds\n", std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000000.0);

		cstart = cend;
		model.save("vectors.bin");
		model.save_text("vectors.txt");
		cend = std::chrono::high_resolution_clock::now();
		printf("save model: %.4f seconds\n", std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000000.0);


	}

	static void TestApplication()
	{
		ApplicationManager m;
	}
};