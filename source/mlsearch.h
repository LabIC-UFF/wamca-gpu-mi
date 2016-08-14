/**
 * @file   mlsearch.h
 *
 * @brief  MLP local searches.
 *
 * @author Eyder Rios
 * @date   2014-06-01
 */

#ifndef __mlsearch_h
#define __mlsearch_h

#include <math.h>
#include "gpu.h"
#include "mlads.h"
#include "mlsolution.h"


// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //

// ################################################################################ //
// ##                                                                            ## //
// ##                                  DATA TYPES                                ## //
// ##                                                                            ## //
// ################################################################################ //

class   MLGPUTask;


// ################################################################################ //
// ##                                                                            ## //
// ##                                CLASS MLSearch                              ## //
// ##                                                                            ## //
// ################################################################################ //

class MLSearch
{
protected:
    MLGPUTask           &gpuTask;

    MLMoveId             searchId;
    const char          *name;

    uint                 callCount;
    uint                 imprvCount;

public:
    /*!
     * Create a MLSearch instance.
     */
    MLSearch(MLGPUTask &parent, MLMoveId id) : gpuTask(parent) {
        searchId   = id;
        name = nameMove[id];
        callCount = 0;
        imprvCount = 0;
    }
    /*!
     * Destroy a MLSearch instance
     */
    virtual
    ~MLSearch() {
    }
    /*!
     * Reset search calls counter.
     */
    inline
    void
    resetStats() {
        callCount = 0;
        imprvCount = 0;
    }
    /*!
     * Perform a 2Opt local search
     */
    virtual
    void
    launchSearch(MLSolution *solBase, MLMove &move) = 0;
    /*!
     * Apply movement on a solution.
     */
    virtual
    void
    applyMove(MLSolution *sol, MLMove &move) = 0;
    /*!
     * Friend classes
     */
    friend class MLSolver;
    friend class MLGPUTask;
};

// ################################################################################ //
// ##                                                                            ## //
// ##                           CLASS MLSearchSwap                               ## //
// ##                                                                            ## //
// ################################################################################ //

class MLSearchSwap : public MLSearch
{
public:
    /*!
     * Create a MLSearchSwap instance.
     */
    MLSearchSwap(MLGPUTask &parent) : MLSearch(parent,MLMI_SWAP) {
    }
    /*!
     * Perform a Swap local search
     */
    void
    launchSearch(MLSolution *solBase, MLMove &move);
    /*!
     * Apply movement on a solution.
     */
    void
    applyMove(MLSolution *sol, MLMove &move);
};

// ################################################################################ //
// ##                                                                            ## //
// ##                           CLASS MLSearch2Opt                               ## //
// ##                                                                            ## //
// ################################################################################ //

class MLSearch2Opt : public MLSearch
{
public:
    /*!
     * Create a MLSearch2Opt instance.
     */
    MLSearch2Opt(MLGPUTask &parent) : MLSearch(parent,MLMI_2OPT) {
    }
    /*!
     * Perform a 2Opt local search
     */
    void
    launchSearch(MLSolution *solBase, MLMove &move);
    /*!
     * Apply movement on a solution.
     */
    void
    applyMove(MLSolution *sol, MLMove &move);
};

// ################################################################################ //
// ##                                                                            ## //
// ##                           CLASS MLSearchOrOpt                              ## //
// ##                                                                            ## //
// ################################################################################ //

class MLSearchOrOpt : public MLSearch
{
protected:
    uint        k;

public:
    /*!
     * Create a MLSearchSwap instance.
     */
    MLSearchOrOpt(MLGPUTask &parent, uint k) : MLSearch(parent,MLMoveId(MLMI_OROPT(k))) {
        this->k = k;
    }
    /*!
     * Perform a OrOpt-k local search
     */
    void
    launchSearch(MLSolution *solBase, MLMove &move);
    /*!
     * Apply movement on a solution.
     */
    void
    applyMove(MLSolution *sol, MLMove &move);
};

#endif
