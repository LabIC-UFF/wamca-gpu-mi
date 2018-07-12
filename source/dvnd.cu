#include <algorithm>
#include <omp.h>

#include "WamcaExperiment.hpp"
#include "dvnd.cuh"
#include "dvnd.h"

extern "C" unsigned int bestNeighborSimple(char * file, int *solution, unsigned int solutionSize, int neighborhood) {
	return bestNeighbor(file, solution, solutionSize, neighborhood);
}

extern "C" unsigned int bestNeighbor(char * file, int *solution, unsigned int solutionSize, int neighborhood, bool justCalc, unsigned int hostCode,
		unsigned int *useMoves, unsigned short *ids, unsigned int *is, unsigned int *js, int *costs, bool useMultipleGpu, unsigned int deviceCount, int *solutionResp) {
	unsigned int selectedDevice = 0;
	if (useMultipleGpu) {
//		checkCudaErrors(cudaGetDeviceCount(&device_count));
		selectedDevice = hostCode % deviceCount;
//		puts("---");
//		printf("number of devices: %d, mpi process code: %u, selected device: %u\n", deviceCount, hostCode, selectedDevice);
//		puts("---");
//		cudaSetDevice(selectedDevice);
	}

	if (!justCalc) {
//		envInit();
		envInit(selectedDevice);
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

//	int seed = 500; // 0: random
	int seed = 0; // 0: random
//	WAMCAExperiment *exper = getExperiment(problem, hostCode, seed);
	static WAMCAExperiment *exper = NULL;
	if (!exper) {
		exper = new WAMCAExperiment(*problem, seed);
	}
	exper->gpuId = selectedDevice;
//	printf("%u;%d;%p;%p\n", hostCode, neighborhood, problem, exper);
	std::vector<MLMove> *moves = NULL;
	if (*useMoves) {
		moves = new std::vector<MLMove>();
	}
	cudaSetDevice(selectedDevice);
	unsigned int resp = exper->runWAMCA2016(1, neighborhood, neighborhood + 1, solution, solutionSize, moves, solutionResp);
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

extern "C" bool noConflict(unsigned short id1, unsigned int i1, unsigned int j1, unsigned short id2, unsigned int i2, unsigned int j2) {
	MLMove64 move1, move2;
	move1.id = id1;
	move1.i = i1;
	move1.j = j1;
	move2.id = id2;
	move2.i = i2;
	move2.j = j2;
	return noConflict(move1, move2);
}

int betterNoConflict(MLMove64 *moves, unsigned int nMoves, int *selectedMoves, int &impValue, bool maximize) {
	MoveIndex movesIndex[nMoves];
	for (int i = 0; i < nMoves; i++) {
		movesIndex[i].move = moves + i;
		movesIndex[i].index = i;
	}

	if (!maximize) {
		std::sort(movesIndex, movesIndex + nMoves, ordenaDecrescente);
	} else {
		std::sort(movesIndex, movesIndex + nMoves, ordenaCrescente);
	}

//	for (int i = 0; i < nMoves; i++) {
//		 unsigned int    id : 4;
//		    unsigned int    i : 14;
//		    unsigned int    j : 14;
//		    int     cost;    // REMOVED :32 !! Goddamn Eyder...
//		printf("%u{id:%u, i:%u, j:%u, cost:%d} ", movesIndex[i].index,
//			movesIndex[i].move->id, movesIndex[i].move->i, movesIndex[i].move->j, movesIndex[i].move->cost);
//	}
//	putchar('\n');
	for (int i = 0; i < nMoves; i++) {
		for (int j = i + 1; j < nMoves; j++) {
			if (!noConflict(movesIndex[i].move->id, movesIndex[i].move->i, movesIndex[i].move->j, movesIndex[j].move->id, movesIndex[j].move->i, movesIndex[j].move->j)) {
//				printf("conflict %d-%d(%u-%u)\n", i, j, movesIndex[i].index, movesIndex[j].index);
				movesIndex[i].index = -1;
				break;
			}
		}
	}

	int selectedMovesLen = impValue = 0;
	for (int i = 0; i < nMoves; i++) {
		if (movesIndex[i].index != -1) {
			selectedMoves[selectedMovesLen++] = movesIndex[i].index;
			impValue += movesIndex[i].move->cost;
		}
	}

	return selectedMovesLen;
}

inline bool noConflict(const MLMove64 &move1, const MLMove64 &move2) {
//	putchar('\n');
	if (move1.id == MLMI_SWAP) {
//		printf("swap x ");
		if (move2.id == MLMI_SWAP) {
//			printf("swap\n");
			return (ABS(move1.i - move2.i) > 1) && (ABS(move1.i - move2.j) > 1) && (ABS(move1.j - move2.i) > 1) && (ABS(move1.j - move2.j) > 1);
		} else if (move2.id == MLMI_2OPT) {
//			printf("2-opt\n");
			return ((move1.i < move2.i - 1) || (move1.i > move2.j - 1)) && ((move1.j < move2.i - 1) || (move1.j > move2.j + 1));
		} else {
			const unsigned int k2 = move2.id == MLMI_OROPT1 ? 1 : (move2.id == MLMI_OROPT2 ? 2 : 3);
//			printf("oropt-%u\n", k2);
			return (move1.j < MIN(move2.i, move2.j) - 1) || (move1.i > MAX(move2.i, move2.j) + k2)
					|| ((move1.i < MIN(move2.i, move2.j) - 1) && (move1.j > MAX(move2.i, move2.j) + k2));
		}
	} else if (move1.id == MLMI_2OPT) {
//		printf("2-opt x ");
		if (move2.id == MLMI_SWAP) {
			return ((move2.i < move1.i - 1) || (move2.i > move1.j + 1)) && ((move2.j < move1.i - 1) || (move2.j > move1.j + 1));
		} else if (move2.id == MLMI_2OPT) {
//			printf("2-opt\n");
			return (move1.j < move2.i - 1) || (move1.i > move2.j + 1) || (move2.j > move1.i - 1) || (move2.i > move1.j + 1);
		} else {
			const unsigned int k2 = move2.id == MLMI_OROPT1 ? 1 : (move2.id == MLMI_OROPT2 ? 2 : 3);
//			printf("oropt-%u\n", k2);
			return (move1.i > MAX(move2.i, move2.j) + k2) || (move1.j < MIN(move2.i, move2.j) - 1);
		}
	} else {
		const unsigned int k1 = move1.id == MLMI_OROPT1 ? 1 : (move1.id == MLMI_OROPT2 ? 2 : 3);
//		printf("oropt-%u x ", k1);
		if (move2.id == MLMI_SWAP) {
			return (move2.j < MIN(move1.i, move2.i) - 1) || (move2.i > MAX(move1.i, move2.i) + k1)
					|| ((move2.i < MIN(move1.i, move2.i) - 1) && (move2.j > MAX(move1.i, move2.i) + k1));
		} else if (move2.id == MLMI_2OPT) {
//			printf("2-opt\n");
			return (move2.j < MIN(move1.i, move1.j) - 1) || (move2.i > MAX(move1.i, move1.j) + k1);
		} else {
			const unsigned int k2 = move2.id == MLMI_OROPT1 ? 1 : (move2.id == MLMI_OROPT2 ? 2 : 3);
//			printf("oropt-%u\n", k2);
			return (MAX(move1.i, move1.j) + k1 < MIN(move2.i, move2.j)) || (MIN(move1.i, move1.j) > MAX(move2.i, move2.j) + k2);
		}
	}
}

bool ordenaCrescente(const MoveIndex &x, const MoveIndex &y) { return (x.move->cost < y.move->cost); }
bool ordenaDecrescente(const MoveIndex &x, const MoveIndex &y) { return (x.move->cost > y.move->cost); }
