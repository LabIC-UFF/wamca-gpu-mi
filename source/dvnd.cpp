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

extern "C" unsigned int applyMovesOld(char * file, int *solution, unsigned int solutionSize, unsigned int useMoves, unsigned short *ids,
		unsigned int *is, unsigned int *js, int *costs) {
	clock_t begin = clock();
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
//	printf("comecou: %d\n", useMoves);
	int contTrocas = 0;
	double elapsed_secs_begin = 0;
	clock_t beginCopy = 0;
	for (int i = 0; i < useMoves; i++) {
		if (i == 0 || ids[i - 1] != ids[i]) {
//			printf("trocou: %d\n", contTrocas++);
			if (i > 0) {
				beginCopy = -clock();
				kernels[ids[i - 1]]->getSolution(solDevice);
				solDevice->update();
				solDevice->ldsUpdate();
				beginCopy += clock();
				elapsed_secs_begin = beginCopy;
			}

			beginCopy = -clock();
			kernels[ids[i]]->setSolution(solDevice);
			beginCopy += clock();
			elapsed_secs_begin += beginCopy;
//			kernels[ids[i]]->sendSolution();
//			kernels[ids[i]]->defineKernelGrid();
		}
//		printf("%d-id:%d, i: %d, j: %d, cost: %d\n", i, moves[i].id, moves[i].i, moves[i].j, moves[i].cost);
//		kernels[ids[i]]->recvResult();
//		kernels[ids[i]]->sync();
		kernels[ids[i]]->applyMove(moves[i]);
	}
//	puts("terminou");
	unsigned int value = 0;
	if (useMoves) {
		beginCopy = -clock();
		kernels[ids[useMoves - 1]]->getSolution(solDevice);
		solDevice->update();
		solDevice->ldsUpdate();

		#pragma omp parallel for
		for (int si = 0; si < solutionSize; si++) {
			solution[si] = solDevice->clients[si];
		}
		solDevice->update();
		solDevice->ldsUpdate();
		beginCopy += clock();
		elapsed_secs_begin += beginCopy;
//		value = solDevice->costCalc();
		value = solDevice->cost;
	}

	delete solDevice;
	delete[] moves;

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "tot:" << elapsed_secs << " copy:" << (elapsed_secs_begin / CLOCKS_PER_SEC) << endl;

	return value;
}

extern "C" unsigned int applyMoves(char * file, int *solutionInt, unsigned int solutionSize, unsigned int useMoves, unsigned short *ids, unsigned int *is,
		unsigned int *js, int *costs) {
	MLProblem * problem = getProblem(file);
	MLMove *moves = vectorsToMove(useMoves, ids, is, js, costs);
	MLSolution* solution = getSolution(problem, solutionInt, solutionSize);

	for (int cont_i = 0; cont_i < useMoves; cont_i++) {
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

			// DEFINIR -DMLP_COST_CHECK
//#ifdef MLP_COST_CHECK
//			if(problem.params.checkCost) {
//				uint ccost = solution->costCalc();
//				l4printf("CHECK COST: %s,\tcost=%u, check=%u\n",name,solution->cost,ccost);
//				if(solution->cost != ccost) {
//					lprintf("%s(%u,%u): wrong=%u, right=%u\n",name ,move.i ,move.j ,solution->cost ,ccost );
//					solution->showCostCalc("SOLUTION: ");
//					EXCEPTION("INVALID COST: %s",problem.name);
//				}
//			}
//#endif
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

			// DEFINIR -DMLP_COST_CHECK

//#ifdef MLP_COST_CHECK
//			if(problem.params.checkCost) {
//				uint ccost = solution->costCalc();
//				l4printf("CHECK COST: %s,\tcost=%u, check=%u\n",name,solution->cost,ccost);
//				if(solution->cost != ccost) {
//					lprintf("%s(%u,%u): wrong=%u, right=%u\n",name ,move.i ,move.j ,solution->cost ,ccost );
//					solution->showCostCalc("SOLUTION: ");
//					EXCEPTION("INVALID COST: %s",problem.name);
//				}
//			}
//#endif
		} else if (ids[cont_i] >= 2 && ids[cont_i] <= 4) {
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

			for (l = 0; l < tag; l++)
				temp[l] = solution->clients[l + i];

			if (i < j) {
				for (l = i + tag; l < j + tag; l++)
					solution->clients[i++] = solution->clients[l];

				for (l = 0; l < tag; l++)
					solution->clients[i++] = temp[l];

				i = move.i;
				j = move.j + tag;
				if (j >= solution->clientCount)
					j--;
			} else if (i > j) {
				for (l = i - 1; l >= j; l--)
					solution->clients[l + tag] = solution->clients[l];

				for (l = 0; l < tag; l++)
					solution->clients[l + j] = temp[l];

				i = move.j;
				j = move.i + tag;
				if (j >= solution->clientCount)
					j--;
			}
			for (; i <= j; i++)
				solution->weights[i] = problem->clients[solution->clients[i - 1]].weight[solution->clients[i]];

//#ifdef MLP_COST_CHECK
//			if(problem.params.checkCost) {
//				uint ccost = solution->costCalc();
//				l4printf("CHECK COST: %s,\tcost=%u, check=%u\n",name,solution->cost,ccost);
//				if(solution->cost != ccost) {
//					lprintf("%s(%u,%u): wrong=%u, right=%u\n",name,move.i,move.j,solution->cost,ccost);
//					solution->showCostCalc("SOLUTION: ");
//					EXCEPTION("INVALID COST: %s",problem.name);
//				}
//			}
//#endif
		}
	}

	for (int si = 0; si < solutionSize; si++) {
		solutionInt[si] = solution->clients[si];
	}

	delete solution;
	delete[] moves;

	return solution->cost;;
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
