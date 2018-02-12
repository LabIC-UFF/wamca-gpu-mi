#include <iostream>

#include <stdio.h>
#include <unistd.h>
#include <map>

#include "WamcaExperiment.hpp"

using namespace std;

void envInit();

extern "C" MLProblem * getProblem(char * file, unsigned int hostCode = 0) {
	static std::map<int, MLProblem *> problems;
	if (problems.find(hostCode) != problems.end()) {
		return problems[hostCode];
	}
	bool costTour = true;
	bool distRound = false;
	bool coordShift = false;
	MLProblem * temp = new MLProblem(costTour, distRound, coordShift);
	temp->load(file);
	problems[hostCode] = temp;
	return temp;
}

extern "C" WAMCAExperiment * getExperiment(MLProblem * problem, unsigned int hostCode = 0, int seed = 500) {
	static std::map<int, WAMCAExperiment *> experiments;
	if (experiments.find(hostCode) != experiments.end()) {
		return experiments[hostCode];
	}
	return experiments[hostCode] = new WAMCAExperiment(*problem, seed);
}

extern "C" unsigned int bestNeighbor(char * file, int *solution, unsigned int solutionSize, int neighborhood,
		bool justCalc = false, unsigned int hostCode = 0) {
	if (!justCalc) {
		envInit();
	}

//	MLProblem *problem = getProblem(file, hostCode);
	static MLProblem *problem = NULL;
	if (!problem) {
		bool costTour = true;
		bool distRound = false;
		bool coordShift = false;
		problem = new MLProblem(costTour, distRound, coordShift);
		problem->load(file);
	}

	if (justCalc) {
//		printf("%u;%d;%p\n", hostCode, neighborhood, problem);
		MLSolution* solDevice = new MLSolution(*problem);
		solDevice->clientCount = solutionSize;
		for (int si = 0; si < solutionSize; si++) {
			solDevice->clients[si] = solution[si];
		}
		solDevice->update();
		unsigned int value = solDevice->costCalc();
		delete solDevice;
		return value;
	}

	int seed = 500; // 0: random
//	WAMCAExperiment *exper = getExperiment(problem, hostCode, seed);
	static WAMCAExperiment *exper = NULL;
	if (!exper) {
		exper = new WAMCAExperiment(*problem, seed);
	}
//	printf("%u;%d;%p;%p\n", hostCode, neighborhood, problem, exper);
	unsigned int resp = exper->runWAMCA2016(1, neighborhood, neighborhood + 1, solution, solutionSize);

	return resp;
}

extern "C" void removeProblem(MLProblem * problem) {
	delete problem;
}

extern "C" void removeExperiment(WAMCAExperiment * exper) {
	delete exper;
}
