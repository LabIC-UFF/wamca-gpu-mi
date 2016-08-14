/**
 * @file	mlgputask.cpp
 *
 * @brief   Handle a GPU monitoring thread.
 *
 * @author	Eyder Rios
 * @date    2015-05-28
 */

#include <algorithm>
#include <unistd.h>
#include <ctype.h>
#include "mlgputask.h"
#include "mlsolution.h"
#include "mlkswap.h"
#include "mlk2opt.h"
#include "mlkoropt.h"
#include "mlads.h"

using namespace std;

// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //

#ifdef MLP_COST_LOG

#define LOG_COST_BEGIN(c,i,k,g)     { solver.logCostStart(); solver.logCostWrite(c,i,k,g); }
#define LOG_COST_END(c,i,k,g)       { solver.logCostWrite(c,i,k,g);                        }

#define LOG_COST_PERIOD(c,i,k,g)    { if(solver.logCost()) solver.logCostWrite(c,i,k,g);   }
#define LOG_COST_IMPROV(c,i,k,g)    { if(solver.logCostPeriod == 0) solver.logCostWrite(c,i,k,g); }

#else

#define LOG_COST_BEGIN(c,i,k,g)
#define LOG_COST_END(c,i,k,g)

#define LOG_COST_PERIOD(c,i,k,g)
#define LOG_COST_IMPROV(c,i,k,g)

#endif


// ################################################################################ //
// ##                                                                            ## //
// ##                                  DATA TYPES                                ## //
// ##                                                                            ## //
// ################################################################################ //

// ################################################################################ //
// ##                                                                            ## //
// ##                                GLOBAL VARIABLES                            ## //
// ##                                                                            ## //
// ################################################################################ //

// ################################################################################ //
// ##                                                                            ## //
// ##                               CLASS MLGPUTask                              ## //
// ##                                                                            ## //
// ################################################################################ //

MLGPUTask::MLGPUTask(MLSolver &s, uint id,
                MLArch arch,
                MLHeuristic heur,
                MLImprovement  improv,
                uint kns) :
                MLTask(s,TTYP_GPUTASK,id + 1),
                solver(s)
{
    l4printf("GPU%u\n",id);

    this->architecture = arch;
    this->heuristic    = heur;
    this->improvement  = improv;

    history = solver.history;

    if(params.rngSeed)
        rng.seed(params.rngSeed + id + 1);
    else
        rng.seed(uint(sysTimer()) + id + 1);

    gpuId = id;
    gpuMemFree = 0;

    memset(&gpuProps,0,sizeof(gpuProps));

    solDevice = new MLSolution(problem,cudaHostAllocDefault);
    solResult = new MLSolution(problem,cudaHostAllocDefault);
    solAux    = new MLSolution(problem,cudaHostAllocDefault);

    searchBits  = ~0;
    searchCount = 0;
    memset(searches,0,sizeof(searches));

    searchSetCount = 0;
    memset(searchSet,0,sizeof(searchSet));

    lsCall = 0;

    kernelBits  = kns;
    kernelCount = 0;
    memset(kernels,0,sizeof(kernels));

    timeExec = 0;
    timeIdle = 0;
    timeSync = 0;

    // Diversification iterations
    maxDiver = problem.params.maxDiver;
    iterDiver = 0;

    // Intensification iterations
    maxInten = (problem.size < problem.params.maxInten) ? problem.size : problem.params.maxInten;
    iterInten = 0;

    for(int i=0;i < MLP_MAX_ALPHA;i++)
        alpha[i] = i / 100.0F;

    // Move merge buffer
    mergeBuffer = new MLMove64[problem.size];
}

MLGPUTask::~MLGPUTask()
{
    l4trace();

    for(int i=0;i < kernelCount;i++)
        delete kernels[i];

    for(int i=0;i < searchCount;i++)
        delete searches[i];

    if(solDevice)
        delete solDevice;

    if(solResult)
        delete solResult;

    if(solAux)
        delete solAux;

    if(mergeBuffer)
        delete[] mergeBuffer;

    // If not reset, do it
    if(gpuMemFree > 0)
        resetDevice();

    l4printf("GPU%u\n",gpuId);
}

void
MLGPUTask::initDevice()
{
    MLSolution *sol;
    size_t      free,
                size;
    bool        flag;

    l4printf("GPU%u\n",gpuId);

    // Get GPU properties
    gpuGetDeviceProperties(&gpuProps,gpuId);
    // Get free memory size
    gpuMemGetInfo(&free,&size);
    gpuMemFree = free;

    l4printf("GPU%u (%s)\n",gpuId,gpuProps.name);
    l4printf("allocFlags = %d\n",params.allocFlags);

    if((heuristic == MLSH_RVND) && (architecture != MLSA_CPU))
        flag = false;
    else
        flag = true;

    // Initialize kernel
    l4printf("GPU%u, initializing %d kernels\n",gpuId,kernelCount);
    for(int i=0;i < kernelCount;i++) {
        l4printf("GPU%u, initializing kernel %s\n",gpuId,kernels[i]->name);
        kernels[i]->init(flag);
    }
}

void
MLGPUTask::resetDevice()
{
    size_t  free,
            total;

    if(gpuMemFree == 0)
        return;

    l4printf("GPU%u (%s)\n",gpuId,gpuProps.name);

    memset(&gpuProps,0,sizeof(gpuProps));

    // Finalize kernels
    for(int i=0;i < kernelCount;i++)
        kernels[i]->term();

#if 0
  gpuMemGetInfo(&free,&total);
    if(free != gpuMemFree)
        WARNING("Possible error releasing GPU memory (before=%llu, after=%u)",gpuMemFree,free);
#endif

    gpuMemFree = 0;
}

void
MLGPUTask::searchInit(uint bits)
{
    l4printf("GPU%u: searchAdd(%x), count=%u, this=%p\n",gpuId,bits,searchCount,this);

    for(int i=0;i < MLP_MAX_NEIGHBOR;i++) {
        if(searches[i])
            delete searches[i];
        searches[i] = NULL;
    }

    searchCount = 0;
    for(int sid=0;sid < MLP_MAX_NEIGHBOR;sid++) {
        if(bits & (1 << sid)) {
            switch(sid) {
            case MLMI_SWAP:
                searches[searchCount++] = new MLSearchSwap(*this);
                break;
            case MLMI_2OPT:
                searches[searchCount++] = new MLSearch2Opt(*this);
                break;
            case MLMI_OROPT1:
                searches[searchCount++] = new MLSearchOrOpt(*this,1);
                break;
            case MLMI_OROPT2:
                searches[searchCount++] = new MLSearchOrOpt(*this,2);
                break;
            case MLMI_OROPT3:
                searches[searchCount++] = new MLSearchOrOpt(*this,3);
                break;
            default:
                EXCEPTION("Invalid move id: %d",sid);
            }
        }
    }
}

void
MLGPUTask::kernelInit(uint bits)
{
    l4printf("GPU%u: kernelAdd(%x), count=%u\n",gpuId,bits,kernelCount);

    for(int i=0;i < MLP_MAX_NEIGHBOR;i++) {
        if(kernels[i])
            delete kernels[i];
        kernels[i] = NULL;
    }

    kernelCount = 0;
    for(int kid=0;kid < MLP_MAX_NEIGHBOR;kid++) {
        if(bits & (1 << kid)) {
            switch(kid) {
            case MLMI_SWAP:
                kernels[kernelCount++] = new MLKernelSwap(*this);
                break;
            case MLMI_2OPT:
                kernels[kernelCount++] = new MLKernel2Opt(*this);
                break;
            case MLMI_OROPT1:
                kernels[kernelCount++] = new MLKernelOrOpt(*this,1);
                break;
            case MLMI_OROPT2:
                kernels[kernelCount++] = new MLKernelOrOpt(*this,2);
                break;
            case MLMI_OROPT3:
                kernels[kernelCount++] = new MLKernelOrOpt(*this,3);
                break;
            default:
                EXCEPTION("Invalid move id: %d",kid);
            }
        }
    }
}

void
MLGPUTask::searchSetInit(uint count)
{
    for(int i=0;i < count;i++)
        searchSet[i] = i;
    searchSetCount = count;
}

void
MLGPUTask::searchSetRemove(uint i)
{
    for(;i < searchSetCount - 1;i++)
        searchSet[i] = searchSet[i + 1];
    searchSetCount--;
}

ullong
MLGPUTask::cpuRvnd(MLSolution *solBase, MLSolution *solRvnd)
{
    MLSearch   *search;
    MLMove      move;
    ullong      time;
    uint        s;

    // Initialize statistics
    time = sysTimer();

    // Initialize statistics
    resetSearchStats();

    // initialize neighborhood set
    searchSetInit(searchCount);

    // update best solution so far
    solRvnd->assign(solBase,false);

    // update solution cost evaluation data
    solRvnd->ldsUpdate();

    while(searchSetCount > 0) {

        // randomly selects a neighborhood
        s = rng.rand(searchSetCount - 1);
        search = searches[ searchSet[s] ];

        // Performs a neighborhood search
        search->launchSearch(solRvnd,move);

        // Compare search result
        if(move.cost < 0) {
            // update best solution
            search->applyMove(solRvnd,move);

            // update evaluation data
            solRvnd->ldsUpdate();

            // initialize neighborhood set
            searchSetInit(searchCount);

            // Updates statistics
            search->imprvCount++;
        }
        else {
            // remove neighborhood from set
            searchSetRemove(s);
        }
    }

    return sysTimer() - time;
}

#if 0

ullong
MLGPUTask::gpuRvndBI(MLSolution *solBase, MLSolution *solRvnd)
{
    MLKernel   *kernel;
    MLMove      move;
    ullong      timeStart;
    uint        k;

    lprintf("Running gpuRvndBI()\n");

    // Initialize statistics
    timeStart = sysTimer();

    // Initialize statistics
    resetKernelStats();

    // initialize neighborhood set
    searchSetInit(kernelCount);

    // All kernels points to 'solRvnd' solution
    for(k=0;k < kernelCount;k++)
        kernels[k]->solution = solRvnd;

    // update best solution so far
    solRvnd->assign(solBase,false);

    // update solution cost evaluation data
    solRvnd->ldsUpdate();

    LOG_COST_BEGIN(solRvnd->cost,0,NULL,gpuId);

    while(searchSetCount > 0) {

        // randomly selects a neighborhood
        k = rng.rand(searchSetCount - 1);
        kernel = kernels[ searchSet[k] ];

        // Performs a neighborhood search
        kernel->sendSolution();
        kernel->launchKernel();
        kernel->recvResult();
        kernel->sync();
        // Get best search move
        kernel->bestMove(move);

        // Compare search result
        if(move.cost < 0) {
            // update best solution
            kernel->applyMove(move);

            // update evaluation data
            solRvnd->ldsUpdate();

            // initialize neighborhood set
            searchSetInit(kernelCount);

            // Updates statistics
            kernel->imprvCount++;

            LOG_COST_IMPROV(solRvnd->cost,move.cost,kernel,gpuId);
        }
        else {
            // remove neighborhood from set
            searchSetRemove(k);
        }

        LOG_COST_PERIOD(solRvnd->cost,move.cost,kernel,gpuId);
    }

    LOG_COST_END(solRvnd->cost,0,NULL,gpuId);

    return sysTimer() - timeStart;
}

ullong
MLGPUTask::gpuRvndMI(MLSolution *solBase, MLSolution *solRvnd)
{
    MLKernel   *kernel;
    MLMove      move;
    ullong      time;
    uint        k;
    int         mergeCost,
                mergeMoves;

    l4printf("Running gpuRvndMI\n");

    // Initialize statistics
    time = sysTimer();

    // Initialize statistics
    resetKernelStats();

    // initialize neighborhood set
    searchSetInit(kernelCount);

    // All kernels points to 'solRvnd' solution
    for(k=0;k < kernelCount;k++)
        kernels[k]->solution = solRvnd;

    // update best solution so far
    solRvnd->assign(solBase,false);

    // update solution cost evaluation data
    solRvnd->ldsUpdate();

    l4printf("\tgpuRvndMI: start=%u\n",solRvnd->cost);

    LOG_COST_BEGIN(solRvnd->cost,0,NULL,gpuId);

    while(searchSetCount > 0) {

        // randomly selects a neighborhood
        k = rng.rand(searchSetCount - 1);
        kernel = kernels[ searchSet[k] ];

        // Performs a neighborhood search

        // Added 2016-05-10 16:00
        kernel->setSolution(solRvnd);
        // End

        kernel->sendSolution();
        kernel->launchKernel();
        kernel->recvResult();
        kernel->sync();
        // Get best kernel merged moves
        mergeCost = kernel->mergeGreedy(mergeBuffer,mergeMoves);

        l4printf("GPU RvndMI: %s, improv=%d\n",kernel->name,mergeCost);

        // Compare search result
        if(mergeCost < 0) {
            // Update merge move counter
            kernel->mergeCount += mergeMoves;

            // Apply improved movement to 'solBest'
            while(mergeMoves > 0) {
                move64ToMove(move,mergeBuffer[--mergeMoves]);
                kernel->applyMove(move);
            }
            // update evaluation data
            solRvnd->ldsUpdate();

            // initialize neighborhood set
            searchSetInit(kernelCount);

            // Updates statistics
            kernel->imprvCount++;

            LOG_COST_IMPROV(solRvnd->cost,move.cost,kernel,gpuId);
        }
        else {
            // remove neighborhood from set
            searchSetRemove(k);
        }

        LOG_COST_PERIOD(solRvnd->cost,move.cost,kernel,gpuId);
    }

    LOG_COST_END(solRvnd->cost,0,NULL,gpuId);

    return sysTimer() - time;
}

#endif

ullong
MLGPUTask::gpuRvnd(MLSolution *solBase, MLSolution *solRvnd)
{
    MLKernel   *kernel;
    MLMove      move;
    ullong      time;
    uint        k;
    int         cost,
                moveCount;

    EXCEPTION("Regular RVND cannot be called for VNS'2016");

    l4printf("Running %s/%s/%s\n",
                    nameHeuristic[heuristic],
                    nameArch[architecture],
                    nameImprov[improvement]);

    // Initialize statistics
    time = sysTimer();

    // Initialize statistics
    resetKernelStats();

    // initialize neighborhood set
    searchSetInit(kernelCount);

    // All kernels points to 'solRvnd' solution
    for(k=0;k < kernelCount;k++)
        kernels[k]->solution = solRvnd;

    // update best solution so far
    solRvnd->assign(solBase,false);

    // update solution cost evaluation data
    solRvnd->ldsUpdate();

    l4printf("###### gpuRvnd_%s: start=%u\n",nameImprov[improvement],solRvnd->cost);

    LOG_COST_BEGIN(solRvnd->cost,0,NULL,gpuId);

    while(searchSetCount > 0) {

        // randomly selects a neighborhood
        k = rng.rand(searchSetCount - 1);
        kernel = kernels[ searchSet[k] ];

        // Not necessary because all kernel->solution points to solRvnd
        // kernel->setSolution(solRvnd);

        // Performs a neighborhood search
        kernel->sendSolution();
        kernel->launchKernel();
        kernel->recvResult();
        kernel->sync();

        if(improvement == MLSI_MULTI) {
            // Get best kernel merged moves
            cost = kernel->mergeGreedy(mergeBuffer,moveCount);

            l4printf("\tgpuRvnd_MI: %s\timprov=%-6d\tmoves=%d\n",
                            kernel->name,
                            cost,moveCount);

            // Apply improvement moves
            if(cost < 0) {
                // Update merge move counter
                kernel->mergeCount += moveCount;

                // Apply improved movement to 'solBest'
                while(moveCount > 0) {
                    move64ToMove(move,mergeBuffer[--moveCount]);
                    kernel->applyMove(move);
                }
            }
        }
        else {
            // Get best move
            cost = kernel->bestMove(move);

            l4printf("\tgpuRvnd_BI: %s\timprov=%-6d\tmoves=%d\t%s(%d,%d)\n",
                            kernel->name,
                            cost,1,
                            kernel->name,move.i,move.j);

            //solRvnd->show("RVND_SOL =");
            //kernel->solution->show("1.KRN_SOL=");
            // Apply improvement move
            if(cost < 0) {
                // Apply move
                kernel->applyMove(move);
                // Update merge move counter
                kernel->mergeCount++;
            }
            //kernel->solution->show("2.KRN_SOL=");
        }

        // Compare search result
        if(cost < 0) {
            // Copy solution
            kernel->getSolution(solRvnd,false);
            // update evaluation data
            solRvnd->ldsUpdate();

            // initialize neighborhood set
            searchSetInit(kernelCount);

            // Updates statistics
            kernel->imprvCount++;

            LOG_COST_IMPROV(solRvnd->cost,move.cost,kernel,gpuId);
        }
        else {
            // remove neighborhood from set
            searchSetRemove(k);
        }

        LOG_COST_PERIOD(solRvnd->cost,move.cost,kernel,gpuId);
    }

    LOG_COST_END(solRvnd->cost,0,NULL,gpuId);

    return sysTimer() - time;
}

ullong
MLGPUTask::gpuRvndAll(MLSolution *solBase, MLSolution *solRvnd)
{
    MLKernel   *kernel;
    MLMove      move;
    ullong      time,
                timeMI,
                timeBI,
                timeNS;
    llong       gMin,gMax,gAvg, gSum;
    uint        i,k,mic;
    int         cost,
                moveCount;

    lsCall++;

    l4printf("Running %s/%s/%s\n",
                    nameHeuristic[heuristic],
                    nameArch[architecture],
                    nameImprov[improvement]);

    // Initialize statistics
    time = sysTimer();

    // Initialize statistics
    resetKernelStats();

    // initialize neighborhood set
    searchSetInit(kernelCount);

    // All kernels points to 'solRvnd' solution
    for(k=0;k < kernelCount;k++)
        kernels[k]->solution = solRvnd;

    // update best solution so far
    solRvnd->assign(solBase,false);

    // update solution cost evaluation data
    solRvnd->ldsUpdate();

    l4printf("###### gpuRvnd_%s: start=%u\n",nameImprov[improvement],solRvnd->cost);

    LOG_COST_BEGIN(solRvnd->cost,0,NULL,gpuId);

    while(searchSetCount > 0) {

        // randomly selects a neighborhood
        k = rng.rand(searchSetCount - 1);
        kernel = kernels[ searchSet[k] ];

        // Save current solution
        solAux->assign(solRvnd,true);

        // Performs a neighborhood search

        //
        // MULTI IMPROVEMENT
        //
        timeNS = sysTimer();

        // Not necessary because kernel->solution points to solRvnd
        // kernel->setSolution(solRvnd);

        kernel->sendSolution();
        kernel->launchKernel();
        kernel->recvResult();
        kernel->sync();

        timeNS = sysTimer() - timeNS;

        timeMI = sysTimer();
        // Get best kernel merged moves
        cost = kernel->mergeGreedy(mergeBuffer,moveCount);
        mic = moveCount;

        // Apply improvement moves
        if(cost < 0) {
            // Update merge move counter
            kernel->mergeCount += moveCount;

            // Apply improved movement to 'solBest'
            while(moveCount > 0) {
                move64ToMove(move,mergeBuffer[--moveCount]);
                kernel->applyMove(move);
            }
        }

        timeMI = sysTimer() - timeMI;

        if(cost < 0) {
            move64ToMove(move,mergeBuffer[0]);
            gMin = gMax = move.cost;
            gSum = 0;
            for(i=0;i < mic;i++) {
                move64ToMove(move,mergeBuffer[i]);
                if(move.cost < gMin)
                    gMin = move.cost;
                if(move.cost > gMax)
                    gMax = move.cost;
                gSum += move.cost;
            }
            gAvg = gSum / mic;
        }
        else {
            gSum = gMax = gMin = gAvg = 0;
        }

        //
        // BEST IMPROVEMENT
        //
        kernel->setSolution(solAux,true);
        kernel->sendSolution();
        kernel->launchKernel();
        kernel->recvResult();
        kernel->sync();

        timeBI = sysTimer();

        // Get best move
        cost = kernel->bestMove(move);

        if(cost < 0) {
            // Apply move
            kernel->applyMove(move);
            // Update merge move counter
            kernel->mergeCount++;
        }

        timeBI = sysTimer() - timeBI;

        if(cost < 0) {
            // Time, Call, Sin, Sout, MIC, Gmin, Gmax, Gavg, Gsum, Tkrn, Tbi, Tmi
            fprintf(logExec,"%llu\t%llu\t%u\t%u\t%s\t%d\t%lld\t%lld\t%lld\t%lld\t%llu\t%llu\t%llu\n",
                            sysTimer(),
                            lsCall,
                            solAux->cost,
                            kernel->solution->cost,
                            kernel->name,
                            mic,
                            gMin,
                            gMax,
                            gAvg,
                            gSum,
                            timeNS,
                            timeBI,
                            timeMI
                            );
        }

        // Compare search result
        if(cost < 0) {
            // Copy solution
            kernel->getSolution(solRvnd,false);
            // update evaluation data
            solRvnd->ldsUpdate();

            l4printf("Update\t%u\n",solRvnd->cost);

            // initialize neighborhood set
            searchSetInit(kernelCount);

            // Updates statistics
            kernel->imprvCount++;

            LOG_COST_IMPROV(solRvnd->cost,move.cost,kernel,gpuId);
        }
        else {
            // remove neighborhood from set
            searchSetRemove(k);
        }

        LOG_COST_PERIOD(solRvnd->cost,move.cost,kernel,gpuId);
    }

    LOG_COST_END(solRvnd->cost,0,NULL,gpuId);

    return sysTimer() - time;
}

ullong
MLGPUTask::dvndAll(MLSolution *solDvnd, ullong maxTime)
{
    MLKernel    *kernel;
    MLSolution  *solStart;
    MLMove       move;
    ullong       time;
    uint         prevCost;
    uint         i,h,k;
    int          optimaBits,
                 optimaCount;
    int          cost,count;
    bool         hflag;

    ullong       timeMI,
                 timeBI,
                 timeNS;
    llong        gMin,gMax,gAvg, gSum;
    uint         mic;



    l4printf("Running %s/%s/%s\n",
                    nameHeuristic[heuristic],
                    nameArch[architecture],
                    nameImprov[improvement]);

    // Save starting time
    time = sysTimer();

    // DVND best solution
    solDvnd->clear();

    // Time spent locked
    timeIdle = 0;

    // Clear kernel queue
    kernelQueue.clear();

    solStart = history->best();
    if(solStart == NULL)
        EXCEPTION("*** Empty DVND History");

    // Initialize kernels solutions
    for(i=0;i < kernelCount;i++) {
        kernels[i]->resetStats();
        kernels[i]->setSolution(solStart);
        l4printf("GPU %d, kernel %s(%u)\n",gpuId,kernels[i]->name,solStart->cost);
    }

    // Copy initial solution to GPU
    for(i=0;i < kernelCount;i++)
        kernels[i]->sendSolution();

    // Launch kernel
    for(i=0;i < kernelCount;i++)
        kernels[i]->launchKernel();

    // Receive result
    for(i=0;i < kernelCount;i++) {
        kernels[i]->recvResult();
        kernels[i]->addCallback(kernelCallback);
    }

    // Local optima kernels
    optimaBits  = 0;
    optimaCount = 0;

    while(optimaCount < kernelCount) {

        if(maxTime && (sysTimer() - time >= maxTime))
            break;

        // Gets the last task completed
        // If no task, thread is blocked until a task is completed
        timeNS = sysTimer();
        kernelQueue.pop(kernel);
        kernel->sync();
        timeNS = sysTimer() - time;
        timeIdle += timeNS;

        l4printf("GPU %d, kernel %s\n",gpuId,kernel->name);

        // Previous solution cost
        prevCost = kernel->solution->cost;

        gSum = gMax = gMin = gAvg = 0;

        if(improvement == MLSI_MULTI) {
            // Get best kernel merged moves
            cost = kernel->mergeGreedy(mergeBuffer,count);
            mic  = count;

            l4printf("GPU %u, Merged %-6s moves=%d\tcost=%d\n",
                            gpuId,kernel->name,count,cost);

            // Has improvement?
            if(cost < 0) {
                // Update move merge counter
                kernel->mergeCount += count;

                move64ToMove(move,mergeBuffer[0]);
                gMin = gMax = move.cost;
                gSum = 0;

                // Apply improved movement to 'solBest'
                while(count > 0) {
                    move64ToMove(move,mergeBuffer[--count]);
                    l4printf(">> Apply %s(%u,%u)=%d\n",kernel->name,move.i,move.j,move.cost);
                    kernel->applyMove(move);

                    if(move.cost < gMin)
                        gMin = move.cost;
                    if(move.cost > gMax)
                        gMax = move.cost;
                    gSum += move.cost;
                }
                gAvg = gSum / mic;
            }
        }
        else {
            // Get best kernel move
            cost = kernel->bestMove(move);
            mic  = 1;

            l4printf("%s\tmove=%d\n",kernel->name,move.cost);
            l4printf("optCount=%d\n",optimaCount);

            // Has improvement?
            if(cost < 0) {
                l4printf("\tGPU%u: %s(%d,%d)=%d\tbase=%d\tBEFORE applyMove()\n",
                                gpuId,
                                kernel->name,move.i,move.j,move.cost,
                                kernel->solution->cost);

                // Apply improved movement to 'solBest'
                kernel->applyMove(move);
                // Update merge move counter
                kernel->mergeCount++;
            }
        }

        timeBI = sysTimer();
        kernel->bestMove(move);
        timeBI = sysTimer() - timeBI;

        if(cost < 0) {
            kernel->solution->ldsUpdate();

            // Submit solution to history
            history->lock();
            history->submit(kernel->solution);
            history->unlock();

            // Update statistics
            kernel->imprvCount++;

            // Time, Call, Sin, Sout, MIC, Gmin, Gmax, Gavg, Gsum, Tkrn, Tbi, Tmi
            fprintf(logExec,"%llu\t%llu\t%u\t%u\t%s\t%d\t%lld\t%lld\t%lld\t%lld\t%llu\t%llu\t%llu\n",
                            sysTimer(),
                            0LL,
                            prevCost,
                            kernel->solution->cost,
                            kernel->name,
                            mic,
                            gMin,
                            gMax,
                            gAvg,
                            gSum,
                            timeNS,
                            timeBI,
                            timeMI);
        }

        history->lock();
        hflag = history->select(prevCost,kernel->solution);
        history->unlock();

        if(hflag) {
            // Send solution, process it and copy results
            kernel->sendSolution();
            kernel->launchKernel();
            kernel->recvResult();
            kernel->addCallback(kernelCallback);
        }
        else {
            // Kernel if local optima
            BIT_SET(optimaBits,kernel->id);
            optimaCount++;
        }

        // If any kernel in local optima, process it
        if(optimaBits) {
            // Is current kernel in local optima?
            for(k=0;k < kernelCount;k++) {
                if(BIT_CHECK(optimaBits,k)) {
                    // Select solution from history better than local one
                    history->lock();
                    hflag = history->select(kernels[k]->solution->cost,kernels[k]->solution);
                    history->unlock();

                    if(hflag) {
                        // L = L - { (s,N(k)) }
                        BIT_CLEAR(optimaBits,kernels[k]->id);
                        optimaCount--;

                        // Send solution to stream
                        kernels[k]->sendSolution();
                        kernels[k]->launchKernel();
                        kernels[k]->recvResult();
                        kernels[k]->addCallback(kernelCallback);
                    }
                }
            }
        }
    }

    return sysTimer() - time;
}


#define MLP_DVND_QUEUE
#ifdef  MLP_DVND_QUEUE

void
MLGPUTask::kernelCallback(cudaStream_t stream, cudaError_t status, void *data)
{
    MLKernel    *kernel = (MLKernel *) data;

    l4printf("QUEUE << %s\n",kernel->name);
    kernel->task()->kernelQueue.push(kernel);
}

ullong
MLGPUTask::dvnd(MLSolution *solDvnd, ullong maxTime)
{
    MLKernel    *kernel;
    MLSolution  *solStart;
    MLMove       move;
    ullong       time,tick;
    uint         prevCost;
    uint         i,h,k;
    int          optimaBits,
                 optimaCount;
    int          cost,count;
    bool         hflag;


    EXCEPTION("Regular RVND cannot be called for VNS'2016");

    l4printf("Running %s/%s/%s\n",
                    nameHeuristic[heuristic],
                    nameArch[architecture],
                    nameImprov[improvement]);

    // Save starting time
    time = sysTimer();

    // DVND best solution
    solDvnd->clear();

    // Time spent locked
    timeIdle = 0;

    // Clear kernel queue
    kernelQueue.clear();

    solStart = history->best();
    if(solStart == NULL)
        EXCEPTION("*** Empty DVND History");

    // Initialize kernels solutions
    for(i=0;i < kernelCount;i++) {
        kernels[i]->resetStats();
        kernels[i]->setSolution(solStart);
        l4printf("GPU %d, kernel %s(%u)\n",gpuId,kernels[i]->name,solStart->cost);
    }

    // Copy initial solution to GPU
    for(i=0;i < kernelCount;i++)
        kernels[i]->sendSolution();

    // Launch kernel
    for(i=0;i < kernelCount;i++)
        kernels[i]->launchKernel();

    // Receive result
    for(i=0;i < kernelCount;i++) {
        kernels[i]->recvResult();
        kernels[i]->addCallback(kernelCallback);
    }

    // Local optima kernels
    optimaBits  = 0;
    optimaCount = 0;

    while(optimaCount < kernelCount) {

        if(maxTime && (sysTimer() - time >= maxTime))
            break;

        // Gets the last task completed
        // If no task, thread is blocked until a task is completed
        tick = sysTimer();
        kernelQueue.pop(kernel);
        timeIdle += sysTimer() - tick;

        kernel->sync();

        l4printf("GPU %d, kernel %s\n",gpuId,kernel->name);

        // Previous solution cost
        prevCost = kernel->solution->cost;

        if(improvement == MLSI_MULTI) {
            // Get best kernel merged moves
            cost = kernel->mergeGreedy(mergeBuffer,count);
            l4printf("GPU %u, Merged %-6s moves=%d\tcost=%d\n",
                            gpuId,kernel->name,count,cost);

            // Has improvement?
            if(cost < 0) {
                // Update move merge counter
                kernel->mergeCount += count;
                // Apply improved movement to 'solBest'
                while(count > 0) {
                    move64ToMove(move,mergeBuffer[--count]);
                    l4printf(">> Apply %s(%u,%u)=%d\n",kernel->name,move.i,move.j,move.cost);
                    kernel->applyMove(move);
                }
            }
        }
        else {
            // Get best kernel move
            cost = kernel->bestMove(move);

            l4printf("%s\tmove=%d\n",kernel->name,move.cost);
            l4printf("optCount=%d\n",optimaCount);

            // Has improvement?
            if(cost < 0) {
                l4printf("\tGPU%u: %s(%d,%d)=%d\tbase=%d\tBEFORE applyMove()\n",
                                gpuId,
                                kernel->name,move.i,move.j,move.cost,
                                kernel->solution->cost);

                // Apply improved movement to 'solBest'
                kernel->applyMove(move);
                // Update merge move counter
                kernel->mergeCount++;
            }
        }

        if(cost < 0) {
            kernel->solution->ldsUpdate();

            // Submit solution to history
            history->lock();
            history->submit(kernel->solution);
            history->unlock();

            // Update statistics
            kernel->imprvCount++;
        }

        history->lock();
        hflag = history->select(prevCost,kernel->solution);
        history->unlock();

        if(hflag) {
            // Send solution, process it and copy results
            kernel->sendSolution();
            kernel->launchKernel();
            kernel->recvResult();
            kernel->addCallback(kernelCallback);
        }
        else {
            // Kernel if local optima
            BIT_SET(optimaBits,kernel->id);
            optimaCount++;
        }

        // If any kernel in local optima, process it
        if(optimaBits) {
            // Is current kernel in local optima?
            for(k=0;k < kernelCount;k++) {
                if(BIT_CHECK(optimaBits,k)) {
                    // Select solution from history better than local one
                    history->lock();
                    hflag = history->select(kernels[k]->solution->cost,kernels[k]->solution);
                    history->unlock();

                    if(hflag) {
                        // L = L - { (s,N(k)) }
                        BIT_CLEAR(optimaBits,kernels[k]->id);
                        optimaCount--;

                        // Send solution to stream
                        kernels[k]->sendSolution();
                        kernels[k]->launchKernel();
                        kernels[k]->recvResult();
                        kernels[k]->addCallback(kernelCallback);
                    }
                }
            }
        }
    }

#if 0

    lprintf("GPU %d\tCalls\tImprov\tMerge\tId\n",gpuId);
    for(i=0;i < kernelCount;i++) {
        kernel = kernels[i];
        lprintf("%s\t%u\t%u\t%u\t%u\n",kernel->name,kernel->callCount,
                                  kernel->imprvCount,kernel->mergeCount,kernel->id);
    }
    lprintf("------------------------\n");

#endif


/*
 * This was commented in 2018-05-10 09:45
 * This code was moved to MLSolver::launchDVND()
 *
    history->lock();
    if(!history->best(solDvnd))
        EXCEPTION("No best solution in DVND? Something wrong!");
    history->unlock();
*/

    return sysTimer() - time;
}

#if 0
ullong
MLGPUTask::dvndMI(MLSolution *solDvnd, ullong maxTime)
{
    MLKernel    *kernel;
    MLMove       move;
    ullong       time,tick;
    uint         prevCost;
    uint         i,h,k;
    int          cost;
    int          mergeMoves,
                 mergeCost;
    int          optimaBits;
    int          optimaCount;
    bool         hflag;

    l4trace();
    l4printf("%s\n",__FUNCTION__);

    // Save starting time
    time = sysTimer();

    // Time spent locked
    timeIdle = 0;

    // Clear kernel queue
    kernelQueue.clear();

//    // Initialize history
//    history->lock();
//    history->clear();
//    history->add(solBase);
//    history->unlock();

    // Initialize kernels solutions
    for(i=0;i < kernelCount;i++) {
        kernels[i]->resetStats();
        kernels[i]->setSolution(history->best());
    }

    // Copy initial solution to GPU
    for(i=0;i < kernelCount;i++)
        kernels[i]->sendSolution();

    // Launch kernel
    for(i=0;i < kernelCount;i++)
        kernels[i]->launchKernel();

    // Receive result
    for(i=0;i < kernelCount;i++) {
        kernels[i]->recvResult();
        kernels[i]->addCallback(kernelCallback);
    }

    // Local optima kernels
    optimaBits  = 0;
    optimaCount = 0;

    while(optimaCount < kernelCount) {

        if(maxTime && (sysTimer() - time >= maxTime))
            break;

        // Gets the last task completed
        // If no task, thread is blocked until a task is completed
        tick = sysTimer();
        kernelQueue.pop(kernel);
        timeIdle += sysTimer() - tick;

        kernel->sync();

        // Previous solution cost
        prevCost = kernel->solution->cost;

        // Get best kernel merged moves
        mergeCost = kernel->mergeGreedy(mergeBuffer,mergeMoves);
        l4printf("Merged %-6s moves=%d\tcost=%d\n",
                        kernel->name,mergeMoves,mergeCost);

        // Has improvement?
        if(mergeCost < 0) {
            // Update move merge counter
            kernel->mergeCount += mergeMoves;
            // Apply improved movement to 'solBest'
            while(mergeMoves > 0) {
                move64ToMove(move,mergeBuffer[--mergeMoves]);
                l4printf(">> Apply %s(%u,%u)=%d\n",kernel->name,move.i,move.j,move.cost);
                kernel->applyMove(move);
            }
            kernel->solution->ldsUpdate();

            // Submit solution to history
            history->lock();
            hflag = history->submit(kernel->solution);
            history->unlock();

            // Update statistics
            kernel->imprvCount++;
        }

        history->lock();
        hflag = history->select(prevCost,kernel->solution);
        history->unlock();

        if(hflag) {
            // Send solution, process it and copy results
            kernel->sendSolution();
            kernel->launchKernel();
            kernel->recvResult();
            kernel->addCallback(kernelCallback);
        }
        else {
            // Kernel if local optima
            BIT_SET(optimaBits,kernel->id);
            optimaCount++;
        }

        // If any kernel in local optima, process it
        if(optimaBits) {
            // Is current kernel in local optima?
            for(k=0;k < kernelCount;k++) {
                if(BIT_CHECK(optimaBits,k)) {
                    // Select solution from history better than local one
                    history->lock();
                    hflag = history->select(kernels[k]->solution->cost,kernels[k]->solution);
                    history->unlock();

                    if(hflag) {
                        // L = L - { (s,N(k)) }
                        BIT_CLEAR(optimaBits,kernels[k]->id);
                        optimaCount--;

                        // Send solution to stream
                        kernels[k]->sendSolution();
                        kernels[k]->launchKernel();
                        kernels[k]->recvResult();
                        kernels[k]->addCallback(kernelCallback);
                    }
                }
            }
        }
    }

    history->lock();
    if(!history->best(solDvnd))
        EXCEPTION("No best solution in DVND? Something wrong!");
    history->unlock();

    l4trace();

    return sysTimer() - time;
}

#endif

#else

ullong
MLGPUTask::dvnd(MLSolution *solBase, MLSolution *solDvnd, ullong maxTime)
{
    MLKernel    *kernel;
    uint         krand[MLP_MAX_NEIGHBOR];
    MLMove       move;
    ullong       time;
    uint         prevCost;
    uint         i,h,k;
    int          historyBits;
    int          optimaCount;

    l4trace();

    // Save starting time
    time = sysTimer();

    // Update eval data
    solBase->evalUpdate();

    // Initialize history
    history->clear();
    history->add(solBase);

    // Initialize kernels solutions
    for(i=0;i < kernelCount;i++) {
        krand[i] = i;

        kernels[i]->resetStats();
        kernels[i]->setSolution(solBase);
        kernels[i]->flagOptima = false;
    }

    // Copy initial solution to GPU
    for(i=0;i < kernelCount;i++)
        kernels[i]->sendSolution();

    // Launch kernel
    for(i=0;i < kernelCount;i++)
        kernels[i]->launchKernel();

    // Receive result
    for(i=0;i < kernelCount;i++)
        kernels[i]->recvResult();

    // Local optima kernels
    optimaCount = 0;

    // History access: initially only kernel 0 (swap) has access
    historyBits = BIT_WORD(0);

    while(optimaCount < kernelCount) {

        // Shuffle kernel checking order
        rng.shuffle(krand,kernelCount);

        for(k=0;k < kernelCount;k++) {

            if(maxTime && (sysTimer() - time >= maxTime)) {
                optimaCount = kernelCount;
                break;
            }

            // Get current kernel
            kernel = kernels[ krand[k] ];

            //printf("\n");
            l4printf("1.kernel %s: best=%s, tag=%d\n",
                            kernel->name,
                            kernel->solution->strCost().c_str(),
                            kernel->flagOptima);

            // If N(k) in L
            if(kernel->flagOptima) {
                l4printf("kernel %s: best=%u\n",kernel->name,kernel->solution->cost);
                // Select solution from history better than local one
                l4printf("\ttry select(%u)\n",kernel->solution->cost);
                if(history->randSelect(kernel->solution->cost,kernel->solution)) {
                    l4printf("\tselected(2) -> %u\n",kernel->solution->cost);
                    // L = L - { (s,N(k)) }
                    kernel->flagOptima = false;
                    optimaCount--;

                    // Send solution to stream
                    kernel->sendSolution();
                    kernel->launchKernel();
                    kernel->recvResult();
                }
            }
            // If N(k) not in L
            else {
                // Check if stream is done
                if(cudaStreamSynchronize(kernel->stream) == cudaSuccess) {
                    // Previous solution cost
                    prevCost = kernel->solution->cost;
                    // Get best kernel move
                    kernel->bestMove(move);

                    l4printf("2.kernel %s: move(%u,%u)=%d ==> cost=%u\n",
                                    kernel->name,
                                    uint(move.i),uint(move.j),move.cost,
                                    kernel->solution->cost + move.cost);
                    //history->show("H1=");

                    // Has improvement?
                    if(move.cost < 0) {
                        // Apply improved movement to 'solBest'
                        kernel->applyMove(move);
                        kernel->solution->evalUpdate();

                        // Submit solution to history
                        history->submit(kernel->solution);
                        //history->show("H2=");

                        // Update statistics
                        kernel->imprvCount++;
                        kernel->imprvCost += -move.cost;
                    }

                    l4printf("kernel %s: try select(%u)\n",kernel->name,prevCost);
                    //history->show("H1=");
                    if(history->randSelect(prevCost,kernel->solution)) {
                        l4printf("\tselected(1) -> %u\n",kernel->solution->cost);
                        // Send solution, process it and copy results (async)
                        kernel->sendSolution();
                        kernel->launchKernel();
                        kernel->recvResult();
                    }
                    else {
                        kernel->flagOptima = true;
                        optimaCount++;
                        l4printf("Kernel %s: OPTIMA, best=%u\t|L|=%u\n",
                                        kernel->name,kernel->solution->cost,
                                        optimaCount);
                    }
                }
            }
        }
    }

    // Update statistics
    for(i=0;i < kernelCount;i++)
        kernels[i]->updateStats();

    if(!history->best(solDvnd))
        EXCEPTION("No best solution in DVND? Something wrong!");

    l4trace();

    return sysTimer() - time;
}

#endif

void
MLGPUTask::perturb(MLSolution &solBase, MLSolution &solPert)
{
    uint    cuts[4];
    uint    i;

    if (solBase.size < 10) {
        solPert.assign(solBase,false);
        return;
    }

    // Generate four random cut points and sort then
    rng.set(0,solBase.clientCount - 2,cuts,4);
    sort(cuts,cuts + 4);

    solPert.clear();

    // [0,c0]
    for(i=0;i <= cuts[0];i++)
        solPert.add(solBase.clients[i]);

    // [c2+1,c3]
    for(i=cuts[2]+1;i <= cuts[3];i++)
        solPert.add(solBase.clients[i]);

    // [c1+1,c2]
    for(i=cuts[1]+1;i <= cuts[2];i++)
        solPert.add(solBase.clients[i]);

    // [c0+1,c1]
    for(i=cuts[0]+1;i <= cuts[1];i++)
        solPert.add(solBase.clients[i]);

    // [c3+1,n]
    for(i=cuts[3]+1;i < solBase.clientCount;i++)
        solPert.add(solBase.clients[i]);

    // Update solution cost
    solPert.update();
}

#if 0
ullong
MLGPUTask::gdvnd(MLSolution *solBest)
{
    MLSolution  solGrasp(problem),
                solDvnd(problem),
                solIls(problem);
    ullong      time;
    uint        i;

    // Save starting time
    time = sysTimer();

    // initially best solution is empty (cost = +oo)
    solBest->clear();

    // Intensification loop
    iterDiver = 0;
    while(iterDiver < maxDiver) {

        // Diversification iterations counter
        iterDiver++;
        //stats.diverIters++;

        // pick an alpha value from alpha set
        i = rng.rand(MLP_MAX_ALPHA - 1);

        // create a new solution
        solGrasp.random(rng,alpha[i]);

        // current best solution
        solIls.assign(solGrasp,false);

        lprintf("GDVND CONSTR/ILS --> %u\t%u\n",solGrasp.cost,solIls.cost);

        // Intensification loop
        iterInten = 0;
        while(iterInten < maxInten) {

            // Intensification iterations counter
            iterInten++;
            //stats.intenIters++;

            // performs a DVND search over 'lsolCurr'
            if(improvement == MLSI_BEST) {
                dvndBI(&solGrasp,&solDvnd);
                l4printf("GDVND dvndBI(%u) --> %u\n",solGrasp.cost,solDvnd.cost);
            }
            else {
                dvndMI(&solGrasp,&solDvnd);
                lprintf("GDVND dvndMI(%u) --> %u\n",solGrasp.cost,solDvnd.cost);
            }

            // if 'rvndSol' cost is best than 'lsolBest', update best solution
            if(solDvnd.cost < solIls.cost) {
                // Update best solution
                solIls.assign(solDvnd,false);

                lprintf("GDVND IMPROV(%u) --> %u\n",solGrasp.cost,solDvnd.cost);

                // Reset intensify load
                iterInten = 0;
            }

            // perturb current solution
            if(iterInten < maxInten - 1) {
                perturb(solIls,solGrasp);
                lprintf("\tPERT(%u) --> %u\n",solIls.cost,solGrasp.cost);
            }
        }

        // update best device solution
        if(solIls.cost < solBest->cost)
            solBest->assign(solIls,false);

        l4printf("device best=%u\n",solBest->cost);
    }

    return sysTimer() - time;
}

void
MLGPUTask::graspILS()
{
    MLSolution  lsolCurr(problem),
                lsolDvnd(problem),
                lsolBest(problem);
    uint        i;

    // initially best solution is empty (cost = +oo)
    solDevice->clear();

    // Intensification loop
    iterDiver = 0;
    while(iterDiver < maxDiver) {

        // Diversification iterations counter
        iterDiver++;
        //stats.diverIters++;

        // pick an alpha value from alpha set
        i = rng.rand(MLP_MAX_ALPHA - 1);

        // create a new solution
        lsolCurr.random(rng,alpha[i]);

        // current diversification best solution
        lsolBest.assign(lsolCurr,false);

        // Intensification loop
        iterInten = 0;
        while(iterInten < maxInten) {

            // Intensification iterations counter
            iterInten++;
            //stats.intenIters++;

            // performs a RVND search over 'lsolCurr'
            lprintf("DVND base=%u\n",lsolCurr.cost);
            dvndBI(&lsolCurr,&lsolDvnd);

            // if 'rvndSol' cost is best than 'lsolBest', update best solution
            if(lsolDvnd.cost < lsolBest.cost) {
                // Update best solution
                lsolBest.assign(lsolDvnd,false);
                // Reset intensify load
                iterInten = 0;
            }

            lprintf("DVND dvnd=%u\titrDiver=%u\titrInten=%u\n",
                            lsolDvnd.cost,iterDiver,iterInten);

            // perturb current solution
            if(iterInten < maxInten - 1)
                perturb(lsolBest,lsolCurr);
        }

        // update best device solution
        if(lsolBest.cost < solDevice->cost)
            solDevice->assign(lsolBest,false);

        lprintf("device best=%u\n",solDevice->cost);
    }
}
#endif

//#define MLP_GPU_TESTING
#ifndef MLP_GPU_TESTING

void
MLGPUTask::main()
{
    l4tracef("GPU%u: ENTER main()\n",gpuId);

    // Initialize CPU search instances
    if(architecture == MLSA_CPU)
        searchInit(searchBits);
    else
        searchInit(0);

    // Initialize GPU kernel instances
    if(architecture != MLSA_CPU)
        kernelInit(kernelBits);
    else
        kernelInit(0);

    // Initialize device
    initDevice();

    l4printf("GPU%u: %s/%s/%s\n",gpuId,
                    nameHeuristic[heuristic],
                    nameArch[architecture],
                    nameImprov[improvement]);

#if 0
    char buffer[512],*p;
    p = buffer;
    p += sprintf(p,"GPU%u: ",gpuId);
    for(int i=0;i < kernelCount;i++)
        p += sprintf(p," %s",kernels[i]->name);
    lprintf("%s\n",buffer);
#endif

#ifdef MLP_CPU_ADS
    switch(heuristic) {
    case MLSH_RVND:
        switch(architecture) {
        case MLSA_CPU:
            timeExec = cpuRvnd(solDevice,solResult);
            break;
        case MLSA_SGPU:
            timeExec = gpuRvndAll(solDevice,solResult);
            break;
        case MLSA_MGPU:
            EXCEPTION("RVND does not run on multi-GPU architecture");
        default:
            EXCEPTION("Invalid architecture: %d\n",architecture);
        }
        break;
    case MLSH_DVND:
        timeExec = dvndAll(solResult);
        break;
    default:
        EXCEPTION("Invalid heuristic: %d\n",int(heuristic));
    }
#else
    l4printf("No CPU ADS -- not run\n");
    solResult->assign(*solDevice,false);
    timeExec = 0;
#endif

    // Reset device
    resetDevice();

    l4tracef("GPU%u: LEAVE main()\n",gpuId);
}

#else

void
MLGPUTask::main()
{
    ltrace();

    searchInit(0);
    kernelInit(1);

    // Initialize device
    initDevice();

    lprintf("kernels=%p, kernels[0]=%p\n",kernels,kernels[0]);

    //gpuMemcpyAsync(adsData,solution->adsData,adsDataSize,cudaMemcpyHostToDevice,stream);

    kernels[0]->solution->sequence();
    kernels[0]->solution->show();
    kernels[0]->solution->ldsUpdate();

    lprintf("LDS=%p\tsize=%u\n",kernels[0]->adsData,kernels[0]->adsDataSize);

    kernels[0]->setSolution(solDevice,true);
    kernels[0]->sendSolution();
    kernels[0]->launchShowDistKernel();

    ltrace();

    // Reset device
    resetDevice();

    ltrace();
}

#endif
