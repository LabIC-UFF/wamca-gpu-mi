/**
 * @file	mlsolver.h
 *
 * @brief    Minimum Latency Problem solver.
 *
 * @author	Eyder Rios
 * @date    2015-05-28
 */

#ifndef __mlsolver_h
#define __mlsolver_h

#include <stdio.h>
#include "consts.h"
#include "mtrand.h"
#include "utils.h"
#include "mlproblem.h"
#include "mlsearch.h"
#include "mlsolpool.hpp"


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

/*
 * Incomplete classes
 */
class MLSolver;
class MLTask;
class MLGPUTask;
class MLKernel;

/*!
 * MLPSolver
 */
typedef MLSolver      *PMLSolver;

/*!
 * MLExecStats
 */
struct MLExecStats {
    ullong  timeStart;                  ///< Time application started
    ullong  timeExec;                   ///< Application execution time
    ullong  timeCPUSolver;              ///< Application CPU solving time
    ullong  timeGPUSolver;              ///< Application GPU solving time
    ullong  timeCommTx;                 ///< Time transfering data to GPU
    ullong  timeCommRx;                 ///< Time transfering data from GPU

    uint    cpuSolCount;                ///< Number of generated solutions
    uint    gpuSolCount;                ///< Number of generated solutions
};

/*
 * MLSolverStats
 */
struct MLSolverStats {
    uint    calls[MLP_MAX_NEIGHBOR];
    uint    callsTotal;

    uint    imprv[MLP_MAX_NEIGHBOR];
    uint    imprvTotal;

    uint    merge[MLP_MAX_NEIGHBOR];
    uint    mergeTotal;
    ullong  mergeTime;

    ullong  time;
};

/*!
 * MLKernelHistory
 */
typedef MLSolutionPool<MLP_MAX_HISTORY,true>    MLKernelHistory;


// ################################################################################ //
// ##                                                                            ## //
// ##                                CLASS MLSolver                              ## //
// ##                                                                            ## //
// ################################################################################ //

class MLSolver
{
protected:
    MLParams        &params;            ///< Application parameters
    MLProblem       &problem;           ///< Problem instance

    MTRandom         rng;               ///< Random number generator
    uint             rngSeed;           ///< Random number generator seed

    MLHeuristic      heuristic;
    MLArch           architecture;
    MLImprovement    improvement;

    MLExecStats      stats;             ///< Execution statistics
    FILE            *logExec;           ///< Log file

    MLKernelHistory *history;           ///< Solution history

    MLGPUTask      **gpuTasks;          ///< GPU manager threads
    uint             gpuTaskCount;      ///< Number of GPU solver threads

    MLSolution      *solSolver;         ///< Best solution

    float           alpha[MLP_MAX_ALPHA]; ///<< Alpha values for random constructor
    uint            iterDiver;            ///<< Current diversification iteration
    uint            maxDiver;             ///<< Max diversification iteration
    uint            iterInten;            ///<< Current intensification iteration
    uint            maxInten;             ///<< Max diversification iteration


#ifdef MLP_COST_LOG
    FILE            *logCostFile;       ///< Log cost file
    pthread_mutex_t  logCostMutex;      ///< Mutex for log file writing
    ullong           logCostRef;        ///< Start time
    ullong           logCostPeriod;     ///< Period
    ullong           logCostTime;       ///< Last time data was logged
    uint             logCostLast;       ///< Last cost logged
#endif

private:
    /*!
     * Entry point for GPU manager threads.
     *
     * @param   data    CPU task instance
     */
    static
    void *
    threadMain(void *data);
    /*!
     * DVND (GPU)
     */
    ullong
    launchDvnd(MLSolution *solBase, MLSolution *solDvnd, MLSolverStats *stats);
    /*!
     * GDVND
     */
    ullong
    gdvnd(const MLArch arch, const MLImprovement improv, MLSolverStats *stats);
    /*!
     * RVND (GPU)
     */
    ullong
    launchRvnd(MLSolution *solBase, MLSolution *solRvnd, MLSolverStats *stats);
    /*!
     * GRVND
     */
    ullong
    grvnd(const MLArch arch, const MLImprovement improv, MLSolverStats *stats);
    /*!
     * Initialize solver allocating necessary resources.
     */
    void
    init();
    /*!
     * Finalize solver releasing any allocated resources.
     */
    void
    term();
    /*!
     * Create log files.
     */
    void
    logCreate();
    /*!
     * Close log files.
     */
    void
    logClose();
    /*!
     * Write log header to file.
     */
    void
    logHeader();
    /*!
     * Write log footer to file.
     */
    void
    logFooter();
    /*!
     * Write log execution result to file.
     */
    void
    logResult();
    /*!
     * Experiment header
     */
    void
    experHeader(const MLHeuristic heur, const MLArch arch, const MLImprovement improv);

#ifdef MLP_COST_LOG
    /*!
     * Create log solution file.
     */
    void
    logCostCreate();
    /*!
     * Close log solution file.
     */
    void
    logCostClose();
    /*!
     * Log cost data to file.
     */
    void
    logCostWrite(uint cost, int improv, MLKernel *kernel, uint gpuId);
    /*!
     * Reset log cost timer.
     */
    inline
    void
    logCostStart() {
        logCostRef = sysTimer();
        logCostPeriod = MS2US(params.logCostPeriod);
    }
    /*!
     * Can log cost data to file?
     */
    inline
    bool
    logCost() {
        if((logCostFile == NULL) || (logCostPeriod == 0))
            return false;
        return (sysTimer() - logCostTime) >= logCostPeriod;
    }
#endif
    /*!
     * Solver main method.
     */
    void
    main();
    /*!
     * Initialize tasks.
     */
    void
    initTasks(MLHeuristic heur, MLArch arch, MLImprovement movsel);
    /*!
     * Finalzie tasks.
     */
    void
    termTasks();
    /*!
     * Performs a local search around a solution.
     */
    void
    localSearch(MLSolution *solBase, MLSolution *solBest);
    /*!
     *  Launch solver tasks.
     */
    void
    launchTasks();
    /*!
     *  Join solver tasks.
     */
    void
    joinTasks();
    /*!
     * Generate, if necessary, solutions to experiments.
     */
    bool
    generateSolutions(uint max, bool create = false);
    /*!
     * Experiments
     */
    void
    execExperiments(const MLHeuristic heur, const MLArch arch, const MLImprovement improv);
    /*!
     * Experiment GDVND
     */
    void
    execGdvnd(const MLArch arch, const MLImprovement improv);
    /*!
     * Experiment GRVND
     */
    void
    execGrvnd(const MLArch arch, const MLImprovement improv);
    /*!
     * Perturbation (double bridge)
     */
    void
    perturb(MLSolution &solBase, MLSolution &solPert);
    /*!
     * Tests.
     */
    void
    checkBestMoveCPUxGPU();
    void
    showSearchTime();
    void
    showKernelTime();
    void
    testKernelSearch();
    void
    testMergeMove();
    void
    testShowDist();
    void
    testDvnd();

public:
    /*!
     * Create a MLSolver instance.
     */
    MLSolver(MLProblem &prob);
    /*!
     * Destroy a MLSolver instance.
     */
    ~MLSolver();
    /*!
     * Get time since solver starts to solve problem.
     */
    inline
    ullong
    timer() {
        return sysTimer() - stats.timeStart;
    }
    /*!
     * Solve a instance stored.
     */
    void
    solve();
    /*!
     * Friend classes
     */
    friend class MLTask;
    friend class MLGPUTask;
};

#endif	// __mlsolver_h
