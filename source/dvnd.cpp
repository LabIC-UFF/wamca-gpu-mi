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

extern "C" unsigned int applyMoves(char * file, int *solutionInt, unsigned int solutionSize, unsigned int numberOfMoves, unsigned short *ids, unsigned int *is,
		unsigned int *js, int *costs) {
	MLProblem * problem = getProblem(file);
	MLMove *moves = vectorsToMove(numberOfMoves, ids, is, js, costs);
	MLSolution* solution = getSolution(problem, solutionInt, solutionSize);


	for (int cont_i = 0; cont_i < numberOfMoves; cont_i++) {
		MLMove move = moves[cont_i];

		if (ids[cont_i] == 0) {
			//kernels[0] = new MLKernelSwap(*problem);
			int i, j;
			ushort t;

			i = move.i;
			j = move.j;

			//solution->showCostCalc("BASE    : ");

			solution->cost += move.cost;
//			solution->time = sysTimer();
//			solution->move = move;

			t = solution->clients[i];
			solution->clients[i] = solution->clients[j];
			solution->clients[j] = t;

			solution->weights[i] = solution->dist(i - 1, i);
			solution->weights[i + 1] = solution->dist(i, i + 1);

			solution->weights[j] = solution->dist(j - 1, j);
			if (j + 1 < solution->clientCount) {
				solution->weights[j + 1] = solution->dist(j, j + 1);
			}
		} else if (ids[cont_i] == 1) {
			//kernels[1] = new MLKernel2Opt(*problem);
			int i, j;
			ushort t;

			i = move.i;
			j = move.j;

			//solution->showCostCalc("BASE    : ");

			solution->cost += move.cost;
//			solution->time = sysTimer();
//			solution->move = move;

			t = solution->clients[i];
			solution->clients[i] = solution->clients[j];
			solution->clients[j] = t;

			solution->weights[i] = solution->dist(i - 1, i);
			solution->weights[i + 1] = solution->dist(i, i + 1);

			solution->weights[j] = solution->dist(j - 1, j);
			if (j + 1 < solution->clientCount) {
				solution->weights[j + 1] = solution->dist(j, j + 1);
			}
		} else if (ids[cont_i] > 1 && ids[cont_i] < MLP_MAX_NEIGHBOR) {
			// kernels[2] = new MLKernelOrOpt(*problem, 1);
			// kernels[3] = new MLKernelOrOpt(*problem, 2);
			// kernels[4] = new MLKernelOrOpt(*problem, 3);
			uint tag = ids[cont_i] - 1;

			uint i, j, l;
			ushort temp[tag];

			//solution->showCostCalc("BASE    : ");

			solution->cost += move.cost;
//			solution->time = sysTimer();
//			solution->move = move;

			i = move.i;
			j = move.j;

			for (l = 0; l < tag; l++) {
				temp[l] = solution->clients[l + i];
			}

			if (i < j) {
				for (l = i + tag; l < j + tag; l++) {
					solution->clients[i++] = solution->clients[l];
				}

				for (l = 0; l < tag; l++) {
					solution->clients[i++] = temp[l];
				}

				i = move.i;
				j = move.j + tag;
				if (j >= solution->clientCount) {
					j--;
				}
			} else if (i > j) {
				for (l = i - 1; l >= j; l--) {
					solution->clients[l + tag] = solution->clients[l];
				}

				for (l = 0; l < tag; l++) {
					solution->clients[l + j] = temp[l];
				}

				i = move.j;
				j = move.i + tag;
				if (j >= solution->clientCount) {
					j--;
				}
			}
			for (; i <= j; i++) {
				solution->weights[i] = problem->clients[solution->clients[i - 1]].weight[solution->clients[i]];
			}
		}
	}

	for (int si = 0; si < solutionSize; si++) {
		solutionInt[si] = solution->clients[si];
	}

	solution->update();
	unsigned int value = solution->costCalc();

	delete solution;
	delete[] moves;

//	return solution->cost;;
	return value;
}

WAMCAExperiment * getExperiment(MLProblem * problem, unsigned int hostCode, int seed) {
	static std::map<int, WAMCAExperiment *> experiments;
	if (experiments.find(hostCode) != experiments.end()) {
		return experiments[hostCode];
	}
	return experiments[hostCode] = new WAMCAExperiment(*problem, seed);
}

extern "C" int getNoConflictMoves(unsigned int numberOfMoves, unsigned short *ids, unsigned int *is, unsigned int *js, int *costs,
		int *selectedMoves, int *impValue, bool maximize, bool melhorParaPior) {
	MLMove64 *moves = vectorsToMove64(numberOfMoves, ids, is, js, costs);
	int cont = betterNoConflict(moves, numberOfMoves, selectedMoves, impValue[0], maximize, melhorParaPior);
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

MLMove64 * vectorsToMove64(unsigned int numberOfMoves, unsigned short *ids, unsigned int *is, unsigned int *js, int *costs) {
	MLMove64 *moves = new MLMove64[numberOfMoves];
	for (int i = 0; i < numberOfMoves; i++) {
		moves[i].id = ids[i];
		moves[i].i = is[i];
		moves[i].j = js[i];
		moves[i].cost = costs[i];
//		PRINT_MOVE(i, moves[i]);
	}
	return moves;
}

MLMove * vectorsToMove(unsigned int numberOfMoves, unsigned short *ids, unsigned int *is, unsigned int *js, int *costs) {
	MLMove *moves = new MLMove[numberOfMoves];
	for (int i = 0; i < numberOfMoves; i++) {
		moves[i].id = MLMoveId(ids[i]);
		moves[i].i = is[i];
		moves[i].j = js[i];
		moves[i].cost = costs[i];
//		PRINT_MOVE(i, moves[i]);
	}
	return moves;
}
