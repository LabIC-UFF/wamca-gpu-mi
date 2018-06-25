#include <iostream>

#include <stdio.h>
#include <unistd.h>
#include <map>
#include <vector>

#include "WamcaExperiment.hpp"
#include "dvnd.cuh"

using namespace std;

void envInit();

MLProblem * getProblem(char * file, unsigned int hostCode = 0) {
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

MLSolution* getSolution(MLProblem * problem, int *solution, unsigned int solutionSize) {
	MLSolution* solDevice = new MLSolution(*problem);
	solDevice->clientCount = solutionSize;
	for (int si = 0; si < solutionSize; si++) {
		solDevice->clients[si] = solution[si];
	}
	solDevice->update(); // ldsUpdate
	solDevice->ldsUpdate();
	return solDevice;
}

WAMCAExperiment * getExperiment(MLProblem * problem, unsigned int hostCode = 0, int seed = 500) {
	static std::map<int, WAMCAExperiment *> experiments;
	if (experiments.find(hostCode) != experiments.end()) {
		return experiments[hostCode];
	}
	return experiments[hostCode] = new WAMCAExperiment(*problem, seed);
}

MLMove64 * vectorsToMove64(unsigned int useMoves = 0, unsigned short *ids = NULL, unsigned int *is = NULL, unsigned int *js = NULL, int *costs = NULL) {
	MLMove64 *moves = new MLMove64[useMoves];
	for (int i = 0; i < useMoves; i++) {
		moves[i].id = ids[i];
		moves[i].i = is[i];
		moves[i].j = js[i];
		moves[i].cost = costs[i];
//		PRINT_MOVE(i, moves[i]);
	}
	return moves;
}

MLMove * vectorsToMove(unsigned int useMoves = 0, unsigned short *ids = NULL, unsigned int *is = NULL, unsigned int *js = NULL, int *costs = NULL) {
	MLMove *moves = new MLMove[useMoves];
	for (int i = 0; i < useMoves; i++) {
		moves[i].id = MLMoveId(ids[i]);
		moves[i].i = is[i];
		moves[i].j = js[i];
		moves[i].cost = costs[i];
//		PRINT_MOVE(i, moves[i]);
	}
	return moves;
}

void move64ToVectors(MLMove64 *moves, unsigned short *ids = NULL, unsigned int *is = NULL, unsigned int *js = NULL, int *costs = NULL,
		unsigned int size = 0) {
	for (unsigned int i = 0; i < size; i++) {
		ids[i] = moves[i].id;
		is[i] = moves[i].i;
		js[i] = moves[i].j;
		costs[i] = moves[i].cost;
//		printf("%d;id:%hu;i:%u;j:%u;c:%d\n", i, move.id, move.i, move.j, move.cost);
	}
}

extern "C" unsigned int bestNeighbor(char * file, int *solution, unsigned int solutionSize, int neighborhood, bool justCalc = false, unsigned int hostCode = 0,
		unsigned int *useMoves = 0, unsigned short *ids = NULL, unsigned int *is = NULL, unsigned int *js = NULL, int *costs = NULL) {
	if (!justCalc) {
		envInit();
	}

//	MLProblem *problem = getProblem(file, hostCode);
	static MLProblem *problem = getProblem(file, hostCode);
	/*
	if (!problem) {
		bool costTour = true;
		bool distRound = false;
		bool coordShift = false;
		problem = new MLProblem(costTour, distRound, coordShift);
		problem->load(file);
	}
	*/

	if (justCalc) {
//		printf("%u;%d;%p\n", hostCode, neighborhood, problem);
		MLSolution* solDevice = getSolution(problem, solution, solutionSize);
		unsigned int value = solDevice->cost;
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
	if (*useMoves) {
		moves = new std::vector<MLMove>();
	}
	unsigned int resp = exper->runWAMCA2016(1, neighborhood, neighborhood + 1, solution, solutionSize, moves);
	if (*useMoves) {
		unsigned int size = moves->size();
//		printf("size: %hu, useMoves: %hu\n", size, useMoves);
		size = size < *useMoves ? size : *useMoves;
		for (unsigned int i = 0; i < size; i++) {
			MLMove move = (*moves)[i];
			ids[i] = move.id;
			is[i] = move.i;
			js[i] = move.j;
			costs[i] = move.cost;
		}

		delete moves;
		*useMoves = size;
	}

	return resp;
}

void removeProblem(MLProblem * problem) {
	delete problem;
}

void removeExperiment(WAMCAExperiment * exper) {
	delete exper;
}

extern "C" int getNoConflictMoves(unsigned int useMoves = 0, unsigned short *ids = NULL, unsigned int *is = NULL, unsigned int *js = NULL, int *costs = NULL,
		int *selectedMoves = NULL, int *impValue = NULL, bool maximize = false) {
	MLMove64 *moves = vectorsToMove64(useMoves, ids, is, js, costs);
	int cont = betterNoConflict(moves, useMoves, selectedMoves, impValue[0], maximize);
	delete[] moves;
	return cont;
}

extern "C" unsigned int applyMoves(char * file, int *solution, unsigned int solutionSize, unsigned int useMoves = 0, unsigned short *ids = NULL,
		unsigned int *is = NULL, unsigned int *js = NULL, int *costs = NULL) {
	static MLKernel **kernels = NULL;
	MLProblem * problem = getProblem(file);
	if (!kernels) {
		kernels = new MLKernel*[5];

		kernels[0] = new MLKernelSwap(*problem);
		kernels[0]->init(true);

		kernels[1] = new MLKernel2Opt(*problem);
		kernels[1]->init(true);

		kernels[2] = new MLKernelOrOpt(*problem, 1);
		kernels[2]->init(true);

		kernels[3] = new MLKernelOrOpt(*problem, 2);
		kernels[3]->init(true);

		kernels[4] = new MLKernelOrOpt(*problem, 3);
		kernels[4]->init(true);
	}

	MLMove *moves = vectorsToMove(useMoves, ids, is, js, costs);
	MLSolution* solDevice = getSolution(problem, solution, solutionSize);
	/*
	printf("\nuseMoves: %d\n", useMoves);
	for (int i = 0; i < useMoves; i++) {
		printf("%d-id:%d, i: %3d, j: %3d, cost: %9d\n", i, moves[i].id, moves[i].i, moves[i].j, moves[i].cost);
	}
	*/
	for (int i = 0; i < useMoves; i++) {
		if (i == 0 || ids[i - 1] != ids[i]) {
			if (i > 0) {
				kernels[ids[i - 1]]->getSolution(solDevice);
				solDevice->update();
				solDevice->ldsUpdate();
			}

			kernels[ids[i]]->setSolution(solDevice);
//			kernels[ids[i]]->sendSolution();
//			kernels[ids[i]]->defineKernelGrid();
		}
//		printf("%d-id:%d, i: %d, j: %d, cost: %d\n", i, moves[i].id, moves[i].i, moves[i].j, moves[i].cost);
//		kernels[ids[i]]->recvResult();
//		kernels[ids[i]]->sync();
		kernels[ids[i]]->applyMove(moves[i]);
	}
	unsigned int value = 0;
	if (useMoves) {
		kernels[ids[useMoves - 1]]->getSolution(solDevice);
		solDevice->update();
		solDevice->ldsUpdate();

		#pragma omp parallel for
		for (int si = 0; si < solutionSize; si++) {
			solution[si] = solDevice->clients[si];
		}
		solDevice->update();
		solDevice->ldsUpdate();
		unsigned int value = solDevice->cost;
	}

	delete solDevice;
	delete[] moves;

	return value;
}
