#include <iostream>

#include <stdio.h>
#include <unistd.h>

#include "WamcaExperiment.hpp"

using namespace std;

void envInit();

extern "C" unsigned int bestNeighbor(char * file, int *solution, unsigned int solutionSize, int neighborhood,
		bool justCalc = false) {
	if (!justCalc) {
		envInit();
	}

	bool costTour = true;
	bool distRound = false;
	bool coordShift = false;

	static MLProblem *problem = NULL;
	if (problem == NULL) {
		problem = new MLProblem(costTour, distRound, coordShift);
	}
	problem->load(file);

	if (justCalc) {
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
	static WAMCAExperiment *exper = NULL;
	if (exper == NULL) {
		exper = new WAMCAExperiment(*problem, seed);
	}
	unsigned int resp = exper->runWAMCA2016(1, neighborhood, neighborhood + 1, solution, solutionSize);

	l4printf(">>>> BUILT AT %s %s\n", __DATE__, __TIME__);

	lprintf("finished successfully\n");
	return resp;
}

extern "C" unsigned int calculateValue(char * file, int *solution, unsigned int solutionSize) {
	return bestNeighbor(file, solution, solutionSize, 0);
}
