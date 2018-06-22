#include <algorithm>
#include <omp.h>

#include "dvnd.cuh"

struct MoveIndex {
	MLMove64 *move;
	unsigned int index;
};

bool ordenaCrescente(const MoveIndex &x, const MoveIndex &y) { return (x.move->cost < y.move->cost); }
bool ordenaDecrescente(const MoveIndex &x, const MoveIndex &y) { return (x.move->cost > y.move->cost); }

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

	// Criar as threads não necessariamente melhora a performance e adiciona aleatoriedade
//	#pragma omp parallel for
	for (int i = 0; i < nMoves; i++) {
		for (int j = i + 1; j < nMoves; j++) {
			if (!noConflict(movesIndex[i].move->id, movesIndex[i].move->i, movesIndex[i].move->j, movesIndex[j].move->id, movesIndex[j].move->i, movesIndex[j].move->j)) {
//				printf("conflict %d-%d\n", i, j);
				movesIndex[i].index = -1;
				break;
			}
		}
	}

	int selectedMovesLen = impValue = 0;
	#pragma omp parallel for
	for (int i = 0; i < nMoves; i++) {
		if (movesIndex[i].index != -1) {
			#pragma omp critical
			{
				selectedMoves[selectedMovesLen++] = movesIndex[i].index;
				impValue += movesIndex[i].move->cost;
			}
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
