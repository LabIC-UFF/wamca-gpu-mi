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

int betterNoConflict(MLMove64 *moves, unsigned int nMoves, int *selectedMoves, int &impValue);
bool noConflict(const MLMove64 &move1, const MLMove64 &move2);
