#include "dvnd.cuh"

#define ABS(X) ((X) >= 0 ? (X) : -(X))
#define MIN(X, Y) (((X)<(Y)) ? (X) : (Y))

int betterNoConflict(MLMove64 *moves, unsigned int nMoves, int *selectedMoves) {
	int *conflicts = new int[nMoves * nMoves];
	printf("nMoves: %u\n", nMoves);
	for (int i = 0; i < nMoves; i++) {
		conflicts[i * nMoves + i] = moves[i].cost;
//		conflicts[i * nMoves + i] = 0;
		for (int j = 0; j < nMoves; j++) {
			if (i != j) {
				conflicts[i * nMoves + j] = !noConflict(moves[i], moves[j]) * moves[j].cost;
//				conflicts[i * nMoves + j] = -1;
			}
		}
	}
	/*
	for (int i = 0; i < nMoves; i++) {
		for (int j = i + 1; j < nMoves; j++) {
			if (i == j) {
				conflicts[i * nMoves + i] = 0;
			} else {
				conflicts[j * nMoves + i] = conflicts[i * nMoves + j] = !noConflict(moves[i], moves[j]) * moves[j].cost;
			}
		}
	}
	*/
	for (int i = 0; i < nMoves; i++) {
		for (int j = 0; j < nMoves; j++) {
			printf("%6d ", conflicts[i * nMoves + j]);
		}
		putchar('\n');
	}
	delete []conflicts;
	return 0;
}

bool noConflict(const MLMove64 &move1, const MLMove64 &move2) {
	if (move1.id == MLMI_SWAP) {
		if (move2.id == MLMI_SWAP) {
			return (ABS(move1.i - move2.i) > 1) &&
					(ABS(move1.i - move2.j) > 1) &&
					(ABS(move1.j - move2.i) > 1) &&
					(ABS(move1.j - move2.j) > 1);
		} else if (move2.id == MLMI_2OPT) {
			return ((move1.i < move2.i - 1) ||
					(move1.i > move2.j - 1)) &&
					((move1.j < move2.i -1) ||
					(move1.j > move2.j + 1));
		} else {
			const unsigned int k2 = move2.id == MLMI_OROPT1 ? 1 : (move2.id == MLMI_OROPT2 ? 2 : 3);
			return (move1.j < MIN(move2.i, move2.j) - 1) ||
					(move1.i > MAX(move2.i, move2.j) + k2) ||
					((move1.i < MIN(move2.i, move2.j) - 1) &&
					(move1.j > MAX(move2.i, move2.j) + k2));
		}
	} else if (move1.id == MLMI_2OPT) {
		if (move2.id == MLMI_SWAP) {
			return ((move2.i < move1.i -1) ||
					(move2.i > move1.j + 1)) &&
					((move2.j < move1.i - 1) ||
					(move2.j > move1.j + 1));
		} else if (move2.id == MLMI_2OPT) {
			return (move1.j < move2.i - 1) ||
					(move1.i > move2.j + 1) ||
					(move2.j > move1.i -1 ) ||
					(move2. i > move1.j + 1);
		} else {
			const unsigned int k2 = move2.id == MLMI_OROPT1 ? 1 : (move2.id == MLMI_OROPT2 ? 2 : 3);
			return (move1.i > MAX(move2.i, move2.j) + k2) ||
					(move1.j < MIN(move2.i, move2.j) - 1);
		}
	} else {
		const unsigned int k1 = move1.id == MLMI_OROPT1 ? 1 : (move1.id == MLMI_OROPT2 ? 2 : 3);
		if (move2.id == MLMI_SWAP) {
			return (move2.j < MIN(move1.i, move2.i) - 1) ||
					(move2.i > MAX(move1.i, move2.i) + k1) ||
					((move2.i < MIN(move1.i, move2.i) - 1) &&
					(move2.j > MAX(move1.i, move2.i) + k1));
		} else if (move2.id == MLMI_2OPT) {
			return (move2.j < MIN(move1.i, move1.j) - 1) ||
					(move2.i > MAX(move1.i, move1.j) + k1);
		} else {
			const unsigned int k2 = move2.id == MLMI_OROPT1 ? 1 : (move2.id == MLMI_OROPT2 ? 2 : 3);
			return (MAX(move1.i, move1.j) + k1 < MIN(move2.i, move2.j)) ||
					(MIN(move1.i, move1.j) > MAX(move2.i, move2.j) + k2);
		}
	}
}
