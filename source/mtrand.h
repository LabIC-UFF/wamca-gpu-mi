/**
 * @file   mtrand.h
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

#ifndef __mtrand_h
#define __mtrand_h

#include <ctime>
#include "types.h"

//#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Warray-bounds"


#define MTW_N			624				///< State machine size
#define MTW_M			397
#define MTW_RAND_MAX	0xffffffffU

/******************************************************************************************
 *
 *                                  Class MTRandom
 *
 ******************************************************************************************/

/*!
 * Mersenne Twister(MT) is a pseudorandom number generating algorithm developped by
 * Makoto Matsumoto and Takuji Nishimura (alphabetical order) in 1996/1997. An improvement
 * on initialization was given on 2002 Jan.
 *
 * MT has the following merits:
 * - It is designed with consideration on the flaws of various existing generators.
 * - The algorithm is coded into a C-source downloadable below.
 * - Far longer period and far higher order of equidistribution than any other
 *   implemented generators. (It is proved that the period is 2^19937-1, and
 *   623-dimensional equidistribution property is assured.)
 * - Fast generation. (Although it depends on the system, it is reported that MT is
 *   sometimes faster than the standard ANSI-C library in a system with pipeline and
 *   cache memory.) (Note added in 2004/3: on 1998, usually MT was much faster than rand(),
 *   but the algorithm for rand() has been substituted, and now there are no much
 *   difference in speed.)
 * - Efficient use of the memory. (The implemented C-code mt19937.c consumes only 624
 *   words of working area.)
 */
class MTRandom
{
private:
	static
	uint    idGen;					///< Instance id

	uint    id;						///< Instance id
	ulong 	state[MTW_N]; 			///< State vector array
	int 	p; 						///< Position in state array
	uint 	rseed;					///< Last random seed used

private:
	void
	genState();

public:
	/*!
	 * Creates and initializes an instance of Mersenne Twister (MT) random
	 * number generator.
	 *
	 * Doesn't matter how instances of MTRadom is created, only one state engine
	 * is maintained in memory.
	 *
	 * @see MtRandom()
	 * @see seed
	 */
	MTRandom() {
		id = idGen++;
		p  = MTW_N + 1;
		seed((uint) time(NULL) + id);
	}
	/*!
	 * Creates and initializes an instance of Mersenne Twister (MT) random
	 * number generator.
	 *
	 * @param	s	Random seed that used to initialize the random engine.
	 *
	 * @see MtRandom(uint)
	 * @see seed
	 */
	MTRandom(uint s) {
		id = idGen++;
		p  = MTW_N + 1;
		seed(s);
	}
	/*!
	 * Creates and initializes an instance of Mersenne Twister (MT) random
	 * number generator.
	 *
	 * @param	array	Random seed that used to initialize the random engine.
	 * @param	size	Number of seeds in array
	 *
	 * @see MtRandom(uint)
	 * @see seed
	 */
	MTRandom(const uint *array, int size) {
		seed(array, size);
	}
	/*!
	 * Get the last random seed used to initialize the engine.
	 *
	 * @return Returns the last random seed.
	 *
	 * @see seed()
	 */
	uint
	getSeed() {
		return rseed;
	}
	/*!
	 * Initialize a MTRandom engine with a 32-bit seed obtained from POSIX
	 * \a time() function.
	 *
	 * @see seed()
	 */
	void seed() {
		seed((uint) time(NULL));
	}
	/*!
	 * Initialize a MTRandom engine with a 32-bit seed.
	 *
	 * @see seed()
	 */
	void seed(uint s);
	/*!
	 * Initialize a MTRandom engine with an array of seeds.
	 *
	 * @param	array	Random seed that used to initialize the random engine.
	 * @param	size	Number of seeds in array
	 *
	 * @see seed()
	 */
	void seed(const uint *array, int size);
	/*
	 * Generates a 32-bit integer random number at interval of [0,MTW_RAND_MAX], where
	 * MTW_RAND_MAX = 2^32 - 1.
	 *
	 * @return	Returns the generated random number.
	 *
	 * @see rand(uint)
	 * @see rand(uint,uint)
	 * @see range()
	 */
	uint
	rand() {
		uint x;

		// New state vector needed
		if (p >= MTW_N)
			genState();

		x = state[p++];
		x ^= (x >> 11);
		x ^= (x << 7)  & 0x9d2c5680U;
		x ^= (x << 15) & 0xefc60000U;

		return x ^ (x >> 18);
	}
	/*
	 * Generates a 32-bit integer random number at interval [0,max]
	 *
	 * @param	max		The upper limit for generated random number

	 * @return	Returns the generated random number.
	 *
	 * @see rand()
	 */
	uint
	rand(uint max) {
	    return (uint) (((max + 1.0) * rand()) / (MTW_RAND_MAX + 1.0));
	}
	/*
	 * Generates a 32-bit integer random number at interval [min,max]
	 *
	 * @param	min		The lower limit for generated random number
	 * @param	max		The upper limit for generated random number

	 * @return	Returns the generated random number.
	 *
	 * @see rand()
	 */
	uint
	rand(uint min, uint max) {
	    return min + rand(max - min);
	}
	/*
	 * Generates a pool of \a size random numbers at interval [0,max] and store then
	 * at \a buffer. There is no guarantee that numbers are unique.
	 *
	 * @param	max		The upper limit for generated random number
	 * @param	buffer	An array that receives the set of random numbers
	 * @param	size	The array size

	 * @see set()
	 */
	void
	pool(uint max, uint *buffer, uint size) {
		while(size-- > 0)
			*buffer++ = rand(max);
	}
	/*
	 * Generates a pool of \a size random numbers at interval [min,max] and store
	 * then at \a buffer. There is no guarantee that numbers are unique.
	 *
	 * @param	min		The lower limit for generated random number
	 * @param	max		The upper limit for generated random number
	 * @param	buffer	An array that receives the set of random numbers
	 * @param	size	The array size
	 *
	 * @see set()
	 */
	void
	pool(uint min, uint max, uint *buffer, uint size) {
		while(size-- > 0)
			*buffer++ = rand(min,max);
	}
	/*
	 * Generates a set of \a size unique random numbers at interval [0,max] and
	 * stored then at \a buffer.
	 *
	 * @param	max		The upper limit for generated random number
	 * @param	buffer	An array that receives the set of random numbers
	 * @param	size	The array size
	 *
	 * @return	Returns the quantity of numbers effectively stored in \a buffer.
	 *
	 * @see pool()
	 */
	uint
	set(uint max, uint *buffer, uint size);
	/*
	 * Generates a set of \a size unique random numbers at interval [min,max] and
	 * stored then at \a buffer.
	 *
	 * @param	min		The lower limit for generated random number
	 * @param	max		The upper limit for generated random number
	 * @param	buffer	An array that receives the set of random numbers
	 * @param	size	The array size
	 *
	 * @return	Returns the quantity of numbers effectively stored in \a buffer.
	 */
	uint
	set(uint min, uint max, uint *buffer, uint size);
	/*
	 * Generates a real number in interval [0,1].
	 *
	 * @see realCO
	 * @see realOO
	 */
	double
	realCC() {
	    return ((double) rand()) * (1.0 / 4294967295.0);
	}
	/*
	 * Generates a real number in interval [0,1).
	 *
	 * @see mtrRealCC
	 * @see mtrRealOO
	 */
	double
	realCO() {
	    return ((double) rand()) * (1.0 / 4294967296.0);
	}
	/*
	 * Generates a real number in interval (0,1).
	 *
	 * @see mtrRealCC
	 * @see mtrRealCO
	 */
	double
	realOO() {
	    return (((double) rand()) + 0.5) * (1.0 / 4294967296.0);
	}
	/*!
	 * Shuffle elementos of an array.
	 */
	template<typename T>
	void
	shuffle(T *data, uint n) {
	    uint i,j;
	    T    t;
	    for(i=0;i < n - 1;i++) {
	        j = rand(n - i - 1);
	        t = data[i];
	        data[i] = data[j];
	        data[j] = t;
	    }
	}
};

//#pragma GCC diagnostic pop

#endif
