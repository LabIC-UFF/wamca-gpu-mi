#include <iostream>
#include <ctime>
#include <map>
#include <vector>

#include <stdio.h>
#include <unistd.h>

#include "WamcaExperiment.hpp"
#include "dvnd.cuh"
#include "dvnd.h"

using namespace std;

extern "C" unsigned int applyMoves(char * file, int *solution, unsigned int solutionSize, unsigned int useMoves, unsigned short *ids,
		unsigned int *is, unsigned int *js, int *costs) {
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

		for (int si = 0; si < solutionSize; si++) {
			solution[si] = solDevice->clients[si];
		}
		solDevice->update();
		solDevice->ldsUpdate();
//		value = solDevice->costCalc();
		value = solDevice->cost;
	}

	delete solDevice;
	delete[] moves;

	return value;
}

WAMCAExperiment * getExperiment(MLProblem * problem, unsigned int hostCode, int seed) {
	static std::map<int, WAMCAExperiment *> experiments;
	if (experiments.find(hostCode) != experiments.end()) {
		return experiments[hostCode];
	}
	return experiments[hostCode] = new WAMCAExperiment(*problem, seed);
}

extern "C" int getNoConflictMoves(unsigned int useMoves, unsigned short *ids, unsigned int *is, unsigned int *js, int *costs,
		int *selectedMoves, int *impValue, bool maximize) {
	MLMove64 *moves = vectorsToMove64(useMoves, ids, is, js, costs);
	int cont = betterNoConflict(moves, useMoves, selectedMoves, impValue[0], maximize);
	delete[] moves;
	return cont;
}

MLProblem * getProblem(char * file, unsigned int hostCode) {
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

void move64ToVectors(MLMove64 *moves, unsigned short *ids, unsigned int *is, unsigned int *js, int *costs, unsigned int size) {
	for (unsigned int i = 0; i < size; i++) {
		ids[i] = moves[i].id;
		is[i] = moves[i].i;
		js[i] = moves[i].j;
		costs[i] = moves[i].cost;
//		printf("%d;id:%hu;i:%u;j:%u;c:%d\n", i, move.id, move.i, move.j, move.cost);
	}
}

void removeProblem(MLProblem * problem) {
	delete problem;
}

void removeExperiment(WAMCAExperiment * exper) {
	delete exper;
}

MLMove64 * vectorsToMove64(unsigned int useMoves, unsigned short *ids, unsigned int *is, unsigned int *js, int *costs) {
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

MLMove * vectorsToMove(unsigned int useMoves, unsigned short *ids, unsigned int *is, unsigned int *js, int *costs) {
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
