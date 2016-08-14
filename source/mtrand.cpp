/**
 * @file   mtrand.cpp
 *
 * @brief Mersenne Twister(MT) is a pseudorandom number generating algorithm developped
 *        by Makoto Matsumoto and Takuji Nishimura (alphabetical order) in 1996/1997.
 *
 *        An improvement on initialization was given on 2002 Jan.
 *        MT has the following merits:
 *        - It is designed with consideration on the flaws of various existing generators.
 *        - The algorithm is coded into a C-source downloadable below.
 *        - Far longer period and far higher order of equidistribution than any other
 *          implemented generators. (It is proved that the period is 2^19937-1, and
 *          623-dimensional equidistribution property is assured.)
 *        - Fast generation. (Although it depends on the system, it is reported that MT
 *          is sometimes faster than the standard ANSI-C library in a system with pipeline
 *          and cache memory.) (Note added in 2004/3: on 1998, usually MT was much faster
 *          than rand(), but the algorithm for rand() has been substituted, and now there
 *          are no much difference in speed.)
 *        - Efficient use of the memory. (The implemented C-code mt19937.c consumes
 *          only 624 words of working area.)
 *
 * @author Eyder Rios
 * @date   2014-09-13
 */

#if LOG_LEVEL > 0
#include <iostream>
#endif

#include "mtrand.h"


#define MATRIX_A 	0x9908b0dfUL   	// constant vector a
#define UPPER_MASK 	0x80000000UL 	// most significant w-r bits
#define LOWER_MASK 	0x7fffffffUL 	// least significant r bits


/******************************************************************************************
 *
 *                                 Class MTRandom
 *
 ******************************************************************************************/

/*
 * Instance id
 */
uint
MTRandom::idGen = 0;


void
MTRandom::seed(uint s)
{
	rseed = s;

	state[0] = s & 0xffffffffU;
	for (p=1; p < MTW_N; ++p) {
		state[p] = 1812433253U * (state[p - 1] ^ (state[p - 1] >> 30)) + p;
		state[p] &= 0xffffffffU;
	}
}

void
MTRandom::seed(const uint *array, int size)
{
	int i,j,k;

	seed(19650218U);

	i = 1;
	j = 0;
	k = (size < MTW_N) ? MTW_N : size;

	for (; k; k--) {
		state[i] = (state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)) * 1664525U))
				+ array[j] + j;
		state[i] &= 0xffffffffU;
		i++; j++;
		if(i >= MTW_N) {
			state[0] = state[MTW_N - 1];
			i = 1;
		}
        if (j >= size)
        	j=0;
    }
    for (k=MTW_N-1; k; k--) {
        state[i] = (state[i] ^ ((state[i-1] ^ (state[i-1] >> 30)) * 1566083941UL))- i;
        state[i] &= 0xffffffffUL;
        i++;
        if (i >= MTW_N) {
        	state[0] = state[MTW_N - 1];
        	i = 1;
        }
    }
    // MSB is 1; assuring non-zero initial array
    state[0] = 0x80000000UL;
}

void
MTRandom::genState()
{
	uint mag[2] = { 0x0U, MATRIX_A };
	uint y;
	int  i;

    if (p == MTW_N + 1)
        seed(5489UL);

    for (i=0;i < MTW_N - MTW_M;i++) {
        y = (state[i] & UPPER_MASK) | (state[i + 1] & LOWER_MASK);
        state[i] = state[i + MTW_M] ^ (y >> 1) ^ mag[y & 0x1U];
    }
    for (;i < MTW_N - 1;i++) {
        y = (state[i] & UPPER_MASK) | (state[i + 1] & LOWER_MASK);
        state[i] = state[i + (MTW_M - MTW_N)] ^ (y >> 1) ^ mag[y & 0x1U];
    }
    y = (state[MTW_N - 1] & UPPER_MASK) | (state[0] & LOWER_MASK);
    state[MTW_N - 1] = state[MTW_M - 1] ^ (y >> 1) ^ mag[y & 0x1U];

    p = 0;
}

uint
MTRandom::set(uint max, uint *buffer, uint size)
{
	uint  i,j;

	if(size > max + 1)
		size = max + 1;

	for(i=0;i < size;) {
		buffer[i] = rand(max);
		for(j=0;j < i;j++) {
			if(buffer[j] == buffer[i])
				break;
		}
		if(j == i)
			i++;
	}
	return size;
}

uint
MTRandom::set(uint min, uint max, uint *buffer, uint size)
{
	uint  i,j;

	if(min > max)
		return 0;

	if(size > max - min + 1)
		size = max - min + 1;

	for(i=0;i < size;) {
		buffer[i] = rand(min,max);
		for(j=0;j < i;j++) {
			if(buffer[j] == buffer[i])
				break;
		}
		if(j == i)
			i++;
	}
	return size;
}

