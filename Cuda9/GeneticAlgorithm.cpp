#include "GeneticAlgorithm.h"



GeneticAlgorithm::GeneticAlgorithm(const Population& p, const Objective& obj, const MalwareAnalyzer& rhs)
{
	this->Pop = p;
	this->objective = obj;
	this->malwareAnalyzer = rhs;

	srand(static_cast <unsigned> (time(0)));
}


GeneticAlgorithm::~GeneticAlgorithm()
{
}

void GeneticAlgorithm::run()
{
	bool Found = false;
	int Generation = 0;

	while (!Found) {

		Generation++;

		for (int i = 0; i < Pop.members.size(); i++) {
			Pop.members.at(i).Fitness = 0;
			CalculateFitness(Pop.members.at(i));

			if ( abs(Pop.members.at(i).Fitness - this->objective.parametersOutput[0]) < 0.001) 
				Found = true;
		}

		std::sort(Pop.members.begin(), Pop.members.end(), [](ADN const &a, ADN &b) 
		{return a.Fitness > b.Fitness; });

		std::vector<ADN> Parents{ Pop.members.at(0), Pop.members.at(1) };

		for (int i = 0; i < Pop.members.size(); i++) 
		{

			for (int j = 0; j < Pop.members.at(i).parameters.size(); j++) {

				int TempSelection = rand() % Parents.size();
				Pop.members.at(i).parameters.at(j) = Parents.at(TempSelection).parameters.at(j);

				//dont forget to apply random mutation based on our value from above
				if (rand() % 1000 < 25){
					std::cout << "Mutation on " << j << std::endl;
					if(j == 0)
						Pop.members.at(i).parameters.at(j) = (int)Utils::randBetween(40, 60);
					if (j == 1)
						Pop.members.at(i).parameters.at(j) = Utils::randBetween(0.006, 0.15);
					if (j == 2)
						Pop.members.at(i).parameters.at(j) = Utils::randBetween(0.0016, 0.0022);
				}
			}
			std::cout << "Iteration : " << i << std::endl;
		}
		std::cout << "Generation : " << Generation << " Highest Fitness : " << Parents.at(0).Fitness
			<< " With Sequence : " << Parents.at(0).toString() << std::endl;
	}
	std::cout << "Generation " << Generation << " Evolved to the full sequence" << std::endl;
}

void GeneticAlgorithm::CalculateFitness(ADN& adn)
{
	auto config = ApplicationContext::instance()->getConfiguration();
	auto directoryBase = config->getDirectoryBase();
	auto trainDirectory = directoryBase + config->getTrainDirectory();


	float accuracy = this->malwareAnalyzer.
		TrainNeuralNetwork(trainDirectory + "\\benigns.txt", "network_benigns.txt",
			adn.parameters[0],adn.parameters[1],adn.parameters[2]);

	adn.Fitness = accuracy;

}