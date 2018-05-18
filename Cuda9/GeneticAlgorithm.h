#pragma once
#include "includes.h"

#include "MalwareAnalyzer.h"

typedef float Parameter;

struct Objective {

	std::vector<Parameter> parametersInput;
	std::vector<Parameter> parametersOutput;
};

struct ADN {
	float Fitness;
	std::vector<Parameter> parameters;


	std::string toString()
	{
		std::stringstream ss;
		ss << "[";
		for (auto item : parameters) {
			ss << item; 
			ss << ",";
		}
		ss << "]";
		return ss.str();
	}
};

struct Population {
	std::vector<ADN> members = std::vector<ADN>(50);
};

class GeneticAlgorithm
{
	Population Pop;

	Objective objective;

	MalwareAnalyzer malwareAnalyzer;


public:
	GeneticAlgorithm(const Population& p,const Objective& obj,const MalwareAnalyzer& rhs);
	~GeneticAlgorithm();
	void run();

protected:
	void CalculateFitness(ADN& adn);
};

