#include <iostream>

#include <stdio.h>
#include <unistd.h>
#include <map>
#include <vector>

#include "WamcaExperiment.hpp"
#include "dvnd.cuh"

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

extern "C" unsigned int bestNeighbor(char * file, int *solution, unsigned int solutionSize, int neighborhood, bool justCalc = false, unsigned int hostCode = 0,
		unsigned int useMoves = 0, unsigned short *ids = NULL, unsigned int *is = NULL, unsigned int *js = NULL, int *costs = NULL) {
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
	std::vector<MLMove> *moves = NULL;
	if (useMoves) {
		moves = new std::vector<MLMove>();
	}
	unsigned int resp = exper->runWAMCA2016(1, neighborhood, neighborhood + 1, solution, solutionSize, moves);
	if (useMoves) {
		unsigned int size = moves->size();
//		printf("size: %hu, useMoves: %hu\n", size, useMoves);
		size = size < useMoves ? size : useMoves;
		for (unsigned int i = 0; i < size; i++) {
			MLMove move = (*moves)[i];
			ids[i] = move.id;
			is[i] = move.i;
			js[i] = move.j;
			costs[i] = move.cost;
//			printf("id:%hu;i:%u;j:%u;c:%d\n", move.id, move.i, move.j, move.cost);
		}

		delete moves;
	}

	return resp;
}

extern "C" void removeProblem(MLProblem * problem) {
	delete problem;
}

extern "C" void removeExperiment(WAMCAExperiment * exper) {
	delete exper;
}

extern "C" int getNoConflictMoves(unsigned int useMoves = 0, unsigned short *ids = NULL, unsigned int *is = NULL, unsigned int *js = NULL, int *costs = NULL, int *selectedMoves = NULL) {
	MLMove64 *moves = new MLMove64[useMoves];
	for (int i = 0; i < useMoves; i++) {
		moves[i].id = ids[i];
		moves[i].i = is[i];
		moves[i].j = js[i];
		moves[i].cost = costs[i];
		printf("id:%hu;i:%u;j:%u;c:%d\n", moves[i].id, moves[i].i, moves[i].j, moves[i].cost);
	}
	int cont =  betterNoConflict(moves, useMoves, selectedMoves);
	delete []moves;
	return cont;
}
