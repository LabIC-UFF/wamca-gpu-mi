#include <omp.h>

#include "dvnd.cuh"

int betterNoConflict(MLMove64 *moves, unsigned int nMoves, int *selectedMoves, int &impValue) {
	int *noConflicts = new int[nMoves * nMoves];
//	printf("nMoves: %u\n", nMoves);
//	#pragma omp parallel for
	for (int i = 0; i < nMoves; i++) {
		noConflicts[i * nMoves + i] = moves[i].cost;
		for (int j = i + 1; j < nMoves; j++) {
			if (i != j) {
				noConflicts[i * nMoves + j] = noConflicts[j * nMoves + i] = !noConflict(moves[i], moves[j]) * moves[i].cost;
			}
		}
	}
	PRINT_CONFLICT(noConflicts, nMoves);
	int nThreads = 1;
//	#pragma omp parallel
	nThreads = omp_get_num_threads();

	int *tempMoves = new int[nMoves * nThreads];
	int *valueTempMoves = new int[nThreads];
//	printf("OMP threads: %d\n", nThreads );
	int nTempMoves, nImp;
	nImp = 0;
	impValue = 1;
//	#pragma omp parallel for
	for (int i = 1; i < nMoves; i++) {
		const unsigned int tid = omp_get_thread_num();
		tempMoves[nMoves * tid] = i;
		nTempMoves = 1;
		for (int j = 0; j < i; j++) {
			if (noConflicts[i * nMoves + j]) {
				tempMoves[nMoves * tid + nTempMoves++] = j;
			}
		}

		/*
		printf("listing\n");
		for (int j = 0; j < nTempMoves; j++) {
			printf("%d ", tempMoves[j]);
		}
		putchar('\n');
		*/

		for (int j = 0; j < nTempMoves; j++) {
			valueTempMoves[tid] = 0;
			for (int k = j + 1; k < nTempMoves; ) {
				if (!noConflicts[tempMoves[nMoves * tid + j] * nMoves + tempMoves[nMoves * tid + k]]) {
//					printf("%d conflict %d\n", tempMoves[j], tempMoves[k]);
					tempMoves[nMoves * tid + k] = tempMoves[nMoves * tid + --nTempMoves];
				} else {
//					printf("%d no conflict %d\n", tempMoves[j], tempMoves[k]);
					k++;
				}
			}
			for (int k = 0; k < nTempMoves; k++) {
//				PRINT_MOVE(tempMoves[k], moves[tempMoves[k]]);
				valueTempMoves[tid] += moves[tempMoves[nMoves * tid + k]].cost;
			}
//			#pragma omp critical
			if (valueTempMoves[tid] < impValue) {
				nImp = nTempMoves;
				for (int k = 0; k < nTempMoves; k++) {
					selectedMoves[k] = tempMoves[nMoves * tid + k];
				}
				impValue = valueTempMoves[tid];
			}
//			printf("value: %d\n", valueTempMoves);
		}
	}
//	printf("%d moves, impvalue: %d\n", nImp, impValue);
	// TODO debug
	puts("selectedMoves");
	for (int i = 0; i < nImp; i++) {
		printf("%d ", selectedMoves[i]);
	}
	putchar('\n');

	delete[] valueTempMoves;
	delete[] tempMoves;
	delete[] noConflicts;

	return nImp;
}

inline bool noConflict(const MLMove64 &move1, const MLMove64 &move2) {
	if (move1.id == MLMI_SWAP) {
		if (move2.id == MLMI_SWAP) {
			return (ABS(move1.i - move2.i) > 1) && (ABS(move1.i - move2.j) > 1) && (ABS(move1.j - move2.i) > 1) && (ABS(move1.j - move2.j) > 1);
		} else if (move2.id == MLMI_2OPT) {
			return ((move1.i < move2.i - 1) || (move1.i > move2.j - 1)) && ((move1.j < move2.i - 1) || (move1.j > move2.j + 1));
		} else {
			const unsigned int k2 = move2.id == MLMI_OROPT1 ? 1 : (move2.id == MLMI_OROPT2 ? 2 : 3);
			return (move1.j < MIN(move2.i, move2.j) - 1) || (move1.i > MAX(move2.i, move2.j) + k2)
					|| ((move1.i < MIN(move2.i, move2.j) - 1) && (move1.j > MAX(move2.i, move2.j) + k2));
		}
	} else if (move1.id == MLMI_2OPT) {
		if (move2.id == MLMI_SWAP) {
			return ((move2.i < move1.i - 1) || (move2.i > move1.j + 1)) && ((move2.j < move1.i - 1) || (move2.j > move1.j + 1));
		} else if (move2.id == MLMI_2OPT) {
			return (move1.j < move2.i - 1) || (move1.i > move2.j + 1) || (move2.j > move1.i - 1) || (move2.i > move1.j + 1);
		} else {
			const unsigned int k2 = move2.id == MLMI_OROPT1 ? 1 : (move2.id == MLMI_OROPT2 ? 2 : 3);
			return (move1.i > MAX(move2.i, move2.j) + k2) || (move1.j < MIN(move2.i, move2.j) - 1);
		}
	} else {
		const unsigned int k1 = move1.id == MLMI_OROPT1 ? 1 : (move1.id == MLMI_OROPT2 ? 2 : 3);
		if (move2.id == MLMI_SWAP) {
			return (move2.j < MIN(move1.i, move2.i) - 1) || (move2.i > MAX(move1.i, move2.i) + k1)
					|| ((move2.i < MIN(move1.i, move2.i) - 1) && (move2.j > MAX(move1.i, move2.i) + k1));
		} else if (move2.id == MLMI_2OPT) {
			return (move2.j < MIN(move1.i, move1.j) - 1) || (move2.i > MAX(move1.i, move1.j) + k1);
		} else {
			const unsigned int k2 = move2.id == MLMI_OROPT1 ? 1 : (move2.id == MLMI_OROPT2 ? 2 : 3);
			return (MAX(move1.i, move1.j) + k1 < MIN(move2.i, move2.j)) || (MIN(move1.i, move1.j) > MAX(move2.i, move2.j) + k2);
		}
	}
}
