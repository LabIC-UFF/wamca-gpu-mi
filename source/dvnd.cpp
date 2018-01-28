#include <iostream>

#include <stdio.h>
#include <unistd.h>

#include "WamcaExperiment.hpp"

using namespace std;

void envInit();

extern "C" unsigned int bestNeighbor(char * file, int *solution, unsigned int solutionSize, int neighborhood) {
	envInit();

	bool costTour = true;
	bool distRound = false;
	bool coordShift = false;

	static MLProblem *problem = NULL;
	if (problem == NULL)
	{
		problem = new MLProblem(costTour, distRound, coordShift);
	}
	problem->load(file);

	int seed = 500; // 0: random
	static WAMCAExperiment *exper = NULL;
	if (exper == NULL)
	{
		exper = new WAMCAExperiment(*problem, seed);
	}
	unsigned int resp = exper->runWAMCA2016(1, neighborhood, neighborhood + 1, solution, solutionSize);

	l4printf(">>>> BUILT AT %s %s\n", __DATE__, __TIME__);

	lprintf("finished successfully\n");
	return resp;
}
