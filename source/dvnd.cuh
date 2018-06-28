#ifndef __wamca_dvnd_cuh
#define __wamca_dvnd_cuh

#include "mlads.h"

#define ABS(X) ((X) >= 0 ? (X) : -(X))
#define MIN(X, Y) (((X)<(Y)) ? (X) : (Y))
#define PRINT_CONFLICT(X, N) { \
	printf("        "); \
	for (int __i_ = 0; __i_ < N; __i_++) { \
		printf("%7d ", __i_); \
	} \
	putchar('\n'); \
	for (int __i_ = 0; __i_ < N; __i_++) { \
		printf("%7d ", __i_); \
		for (int __j_ = 0; __j_ < N; __j_++) { \
			if (__j_ >= __i_) { \
				printf("%7d ", (X)[__i_ * (N) + __j_]); \
			} else { \
				printf("        "); \
			} \
		} \
		printf("%7d ", __i_); \
		putchar('\n'); \
	} }
#define PRINT_MOVE(I, M) printf("%d;id:%hu;i:%u;j:%u;c:%d\n", I, (M).id, (M).i, (M).j, (M).cost)

struct MoveIndex {
	MLMove64 *move;
	unsigned int index;
};

extern "C" unsigned int bestNeighborSimple(char * file, int *solution, unsigned int solutionSize, int neighborhood);
extern "C" unsigned int bestNeighbor(char * file, int *solution, unsigned int solutionSize, int neighborhood, bool justCalc = false, unsigned int hostCode = 0,
		unsigned int *useMoves = NULL, unsigned short *ids = NULL, unsigned int *is = NULL, unsigned int *js = NULL, int *costs = NULL, bool useMultipleGpu = false, unsigned int deviceCount = 1);
int betterNoConflict(MLMove64 *moves, unsigned int nMoves, int *selectedMoves, int &impValue, bool maximize);
void envInit(int deviceNumber = 0);
bool noConflict(const MLMove64 &move1, const MLMove64 &move2);
extern "C" bool noConflict(unsigned short id1, unsigned int i1, unsigned int j1, unsigned short id2, unsigned int i2, unsigned int j2);
bool ordenaCrescente(const MoveIndex &x, const MoveIndex &y);
bool ordenaDecrescente(const MoveIndex &x, const MoveIndex &y);

#endif
