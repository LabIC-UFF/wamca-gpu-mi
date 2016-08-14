/**
 * @file	mlgputask.h
 *
 * @brief   Handle a GPU monitoring thread.
 *
 * @author	Eyder Rios
 * @date    2015-05-28
 */

#ifndef __mlgputask_h
#define __mlgputask_h

#include "gpu.h"
#include "mtrand.h"
#include "mlqueue.hpp"
#include "mlsolver.h"
#include "mltask.h"
#include "mlsearch.h"
#include "mlkernel.h"
#include "mlsolution.h"


// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //

#define NS_BITS_ALL        uint(~0)
#define NS_BITS_SWAP       uint(1 << 0)
#define NS_BITS_2OPT       uint(1 << 1)
#define NS_BITS_OROPT1     uint(1 << 2)
#define NS_BITS_OROPT2     uint(1 << 3)
#define NS_BITS_OROPT3     uint(1 << 4)


// ################################################################################ //
// ##                                                                            ## //
// ##                                  DATA TYPES                                ## //
// ##                                                                            ## //
// ################################################################################ //

class MLGPUTask;

/*!
 * PMLGPUTask
 */
typedef MLGPUTask *PMLGPUTask;

/*!
 * MLKernelQueue
 */
typedef MLLockQueue<MLKernel *,MLP_MAX_NEIGHBOR> MLKernelQueue;


// ################################################################################ //
// ##                                                                            ## //
// ##                               CLASS MLGPUTask                              ## //
// ##                                                                            ## //
// ################################################################################ //

class MLGPUTask : public MLTask
{
protected:
    MLSolver        &solver;                    ///< Solver reference
    MLKernelHistory *history;                   ///< Solution history

    MTRandom        rng;                        ///< Random number generator

    uint            gpuId;                      ///< GPU id
    uint            gpuMemFree;                 ///< Amount of free memory on GPU
    cudaDeviceProp  gpuProps;                   ///< Device properties

    MLKernel       *kernels[MLP_MAX_NEIGHBOR];  ///< Kernel searches
    uint            kernelCount;                ///< Number of kernels
    uint            kernelBits;                 ///< Bit mask for active kernels

    MLSearch       *searches[MLP_MAX_NEIGHBOR]; ///< CPU searches
    uint            searchCount;                ///< Number of searches
    uint            searchBits;                 ///< Bit mask for active searches

    MLKernelQueue   kernelQueue;                ///< Finished kernels queue

    uint            searchSet[MLP_MAX_NEIGHBOR];///< Neighborhood set for CPU
    uint            searchSetCount;             ///< Number of elements in neighborhood set

    ullong          lsCall;                     ///< Local search call no.

    MLSolution     *solDevice;                  ///< Base solution
    MLSolution     *solResult;                  ///< Resulting solution
    MLSolution     *solAux;                     ///< Aux solution

    MLMove64       *mergeBuffer;                ///< Independent movements buffer

    float           alpha[MLP_MAX_ALPHA];       ///<< Alpha values for random constructor
    uint            iterDiver;
    uint            maxDiver;
    uint            iterInten;
    uint            maxInten;

    ullong          timeExec;                   ///<< Exec time
    ullong          timeSync;                   ///<< Time spent in synchronizing w/ GPU
    ullong          timeIdle;                   ///<< Time CPU was idle

    MLArch          architecture;               ///< Architecture
    MLHeuristic     heuristic;                  ///< Heuristic
    MLImprovement   improvement;                ///<< Movement selection

private:
    /*!
     * Entry point for kernel launcher thread.
     *
     * @param   data    CPU task instance
     */
    static
    void *
    threadLauncher(void *data);
    /*!
     * Initialize neighborhood set
     */
    void
    searchSetInit(uint count);
    /*!
     * Callback function called when a stream work is completed.
     *
     * @param stream    Stream that was finalized
     * @param status    Error code
     * @param data      User data
     */
    static
    void
    kernelCallback(cudaStream_t stream, cudaError_t status, void *data);
    /*!
     * Remove an element of index \a s from neighborhood set
     */
    void
    searchSetRemove(uint i);
    void
    kernelSetRemove(uint i);

protected:
    /*!
     * Prepare GPU device for execution allocating and initializing resources.
     */
    void
    initDevice();
    /*!
     * Reset device releasing any allocated resources.
     */
    void
    resetDevice();
    /*!
     * Random VND
     */
    ullong
    cpuRvnd(MLSolution *solBase, MLSolution *solRvnd);
    ullong
    gpuRvnd(MLSolution *solBase, MLSolution *solRvnd);
    ullong
    gpuRvndAll(MLSolution *solBase, MLSolution *solRvnd);
    /*!
     * DVND Multi-Improvement
     */
    ullong
    dvnd(MLSolution *solDvnd, ullong maxTime = 0);
    ullong
    dvndAll(MLSolution *solDvnd, ullong maxTime = 0);
    /*!
     * Algorithm based on GDVND
     */
    ullong
    gdvnd(MLSolution *solBest);
    /*!
     * Perturbs a solution.
     */
    void
    perturb(MLSolution &solBase, MLSolution &solPert);
    /*!
     * Algorithm based on GRASP+ILS
     */
    void
    graspILS();
    /*!
     * Compare with RVND performance.
     */
    void
    compRvnd();

public:
    /*!
     * Create a CLCPUSearch instance.
     *
     * @param   frame   Parent framework
     * @param   rseed   Random seed
     * @param   ns      Active kernels (bit mask)
     */
    MLGPUTask(MLSolver &solver, uint id,
              MLArch arch,
              MLHeuristic heur,
              MLImprovement improv,
              uint kns = NS_BITS_ALL);
    /*!
     * Create a CLCPUSearch instance.
     */
    ~MLGPUTask();
    /*!
     * Set device solution
     */
    void
    setSolution(MLSolution *sol) {
        solDevice->assign(sol,true);
    }
    /*!
     * Clear device solution
     */
    void
    clearSolution() {
        solDevice->clear();
    }
    /*!
     * Reset search calls.
     */
    void
    resetSearchStats() {
        for(int i=0;i < MLP_MAX_NEIGHBOR;i++)
            searches[i]->resetStats();
    }
    /*!
     * Reset kernel calls.
     */
    void
    resetKernelStats() {
        for(int i=0;i < MLP_MAX_NEIGHBOR;i++)
            kernels[i]->resetStats();
    }
    /*!
     * Add a search instance to pool.
     */
    void
    searchInit(uint bits);
    /*!
     * Add a search instance to pool.
     */
    void
    kernelInit(uint bits);
    /*!
     * Main procedure of CPU task.
     */
    void
    main();
    /*!
     * Write log execution result to file.
     */
    void
    logResult();
    /*!
     * Friend classes
     */
    friend class  MLSolver;
    friend class  MLTask;
    friend class  MLKernel;
};

#endif	// __mlgputask_h
