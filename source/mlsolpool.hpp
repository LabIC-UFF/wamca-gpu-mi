/**
 * @file	mlsolpool.hpp
 *
 * @brief	Implements a pool of objects.
 *
 * @author	Eyder Rios
 * @date    2015-05-28
 */

#ifndef __mlsolpool_hpp
#define __mlsolpool_hpp

#include <iostream>
#include <pthread.h>
#include "except.h"
#include "log.h"
#include "mtrand.h"
#include "mlsolution.h"


// ################################################################################ //
// ##                                                                            ## //
// ##                             CONSTANTS & MACROS                             ## //
// ##                                                                            ## //
// ################################################################################ //

// ################################################################################ //
// ##                                                                            ## //
// ##                                 DATA TYPES                                 ## //
// ##                                                                            ## //
// ################################################################################ //

// ################################################################################ //
// ##                                                                            ## //
// ##                            CLASS MLSolutionPool                            ## //
// ##                                                                            ## //
// ################################################################################ //

template<int sizPool, bool flag>
class MLSolutionPool
{
public:
    bool             needLock;
    int              count;
    int              size;
    MLSolution      *buffer[sizPool];
    pthread_mutex_t  mtxPool;
    MTRandom        &rng;

public:
    /*!
     * Create a MLSolutionPool instance.
     */
    MLSolutionPool(MLProblem &prob, MTRandom &rand, bool flock) : rng(rand) {
        count = 0;
        size  = sizPool;

        for(int i=0;i < size;i++) {
            buffer[i] = new MLSolution(prob);
            if(flag)
                buffer[i]->adsAlloc();
        }
        needLock = flock;
        pthread_mutex_init(&mtxPool,NULL);
    }
    /*!
     * Destroy a MLSolutionPool instance.
     */
    ~MLSolutionPool() {
        for(int i=0;i < size;i++)
            delete buffer[i];
        pthread_mutex_destroy(&mtxPool);
    }
    /*!
     * Define mutex lock necessity.
     */
    void
    setNeedLock(bool flock) {
        needLock = flock;
        lprintf("MLSolutionPool.needLock = %d\n",needLock);
    }
    /*!
     * Define mutex lock necessity.
     */
    bool
    getNeedLock() {
        return needLock;
    }
    /*!
     * Lock pool for further access.
     */
    inline
    void
    lock() {
        if(needLock) {
            l4printf("Locking pool.\n");
            pthread_mutex_lock(&mtxPool);
        }
    }
    /*!
     * Unlock pool.
     */
    inline
    void
    unlock() {
        if(needLock) {
            l4printf("Unlocking pool.\n");
            pthread_mutex_unlock(&mtxPool);
        }
    }
    /*!
     * Is pool locked?
     */
    inline
    bool
    trylock() {
        return pthread_mutex_trylock(&mtxPool);
    }
    /*!
     * Check if pool is empty.
     */
    inline
    bool
    empty() {
        return count == 0;
    }
    /*!
     * Check if pool is not empty.
     */
    inline
    bool
    notEmpty() {
        return count > 0;
    }
    /*!
     * Check if pool is full.
     */
    inline
    bool
    full() {
        return count == size;
    }
    /*!
     * Clear pool.
     */
    inline
    void
    clear() {
        count = 0;
    }
    /*!
     * Get best solution in pool
     */
    inline
    bool
    best(MLSolution *sol) {
        if(count == 0)
            return false;
        sol->assign(buffer[0],flag);
        return true;
    }
    /*!
     * Get pointer to best solution in pool
     */
    inline
    MLSolution *
    best() {
        if(count == 0)
            return NULL;
        return buffer[0];
    }
    /*!
     * Add solution to pool.
     */
    bool
    add(MLSolution *sol) {
        MLSolution *t;
        if((t = submit(sol->cost)) != NULL)
            t->assign(sol,flag);
        return (t != NULL);
    }
    /*!
     * Get solution slot for a given cost.
     */
    MLSolution *
    submit(uint cost) {
        MLSolution *t;
        int         i,j;

        // If cost greater than worst cost
        if((count == size) && (cost >= buffer[count - 1]->cost))
            return NULL;

        i = 0;
        while((i < count) && (buffer[i]->cost < cost))
            i++;
        if((count > 0) && (buffer[i]->cost == cost))
            return NULL;

        if(count == size)
            count--;

        if(i < count) {
            j = count;
            t = buffer[j];
            while(j > i) {
                buffer[j] = buffer[j - 1];
                j--;
            }
            buffer[i] = t;
        }
        count++;
        return buffer[i];
    }
    /*!
     * Set solution slot for a given cost.
     */
    bool
    submit(MLSolution *sol) {
        MLSolution *t;
        int         i;

        if(count == 0) {
            buffer[count++]->assign(sol,flag);
            return true;
        }

        if(sol->cost < buffer[0]->cost) {
            if(count < size)
                count++;

            t = buffer[count - 1];

            for(i=count - 1;i > 0;i--)
                buffer[i] = buffer[i -1];

            buffer[0] = t;
            t->assign(sol,flag);

            return true;
        }

        return false;
    }

    bool
    submit2(MLSolution *sol) {
        MLSolution *t;
        int         i,j;

        // If cost greater than worst cost
        if((count == size) && (sol->cost >= buffer[count - 1]->cost))
            return false;

        i = 0;
        while((i < count) && (buffer[i]->cost < sol->cost))
            i++;
        if((count > 0) && (buffer[i]->cost == sol->cost))
            return false;

        if(count == size)
            count--;

        if(i < count) {
            j = count;
            t = buffer[j];
            while(j > i) {
                buffer[j] = buffer[j - 1];
                j--;
            }
            buffer[i] = t;
        }
        count++;
        buffer[i]->assign(sol,flag);
        return true;
    }
    /*!
     * Remove solution from pool.
     */
    bool
    del(int index) {
        if(count == 0)
            return false;
        count--;
        if(index < count) {
            MLSolution *t = buffer[index];
            for(;index < count;index++)
                buffer[index] = buffer[index + 1];
            buffer[index] = t;
        }
        return true;
    }

#if 1
    /*!
     * BEST SELECT
     * Get solution with cost better that \a cost.
     */
    bool
    select(uint cost, MLSolution *sol) {
        for(int i=0;i < count;i++) {
            if(buffer[i]->cost < cost) {
                sol->assign(buffer[i],flag);
                return true;
            }
        }
        return false;
    }
#else
    /*!
     * RANDOM SELECT
     * Get solution with cost better that \a cost.
     */
    bool
    select(uint cost, MLSolution *sol) {
        int  i;

        for(i=0;i < count;i++) {
            if(buffer[i]->cost >= cost)
                break;
        }
        if(i > 0) {
            i = rng.rand(0,i - 1);
            sol->assign(buffer[i],flag);
            return true;
        }
        return false;
    }
#endif
    /*!
     * Show pool on stream.
     */
    void
    show(const char *prompt = NULL) {
        if(prompt)
            std::cout << prompt;
        std::cout << '{';

        for(int i=0;i < count;i++) {
            std::cout << buffer[i]->cost;
            if(i + 1 < count)
                std::cout << ',';
        }
        std::cout << '}' << std::endl;
    }
};


#endif	// __mlkrnpool_hpp
