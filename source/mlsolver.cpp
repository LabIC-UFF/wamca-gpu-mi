/**
 * @file	mlsolver.cpp
 *
 * @brief    Minimum Latency Problem solver.
 *
 * @author	Eyder Rios
 * @date    2015-05-28
 */

#include <unistd.h>
#include <algorithm>
#include "except.h"
#include "log.h"
#include "gpu.h"
#include "mlsolution.h"
#include "mlsolver.h"
#include "mlgputask.h"


// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //

#ifdef MLP_COST_LOG

#define LOG_COST_CREATE()       logCostCreate()
#define LOG_COST_CLOSE()        logCostClose()

#else

#define LOG_COST_CREATE()
#define LOG_COST_CLOSE()

#endif

// ################################################################################ //
// ##                                                                            ## //
// ##                                  DATA TYPES                                ## //
// ##                                                                            ## //
// ################################################################################ //

// ################################################################################ //
// ##                                                                            ## //
// ##                                  FUNCTIONS                                 ## //
// ##                                                                            ## //
// ################################################################################ //

// ################################################################################ //
// ##                                                                            ## //
// ##                               CLASS MLSolver                               ## //
// ##                                                                            ## //
// ################################################################################ //

MLSolver::MLSolver(MLProblem &prob) :
            params(prob.params),
            problem(prob)
{
    int     r;

#ifndef GPU_CUDA_DISABLED
    // Check for GPU support
    if(params.maxGPU > 0) {
        r = 0;

        switch(cudaGetDeviceCount(&r)) {
        case cudaSuccess:
            for(int dev;dev < r;dev++) {
                cudaDeviceProp  prop;

                if(cudaGetDeviceProperties(&prop,dev) == cudaSuccess) {
                    if(prop.major < 2)
                        EXCEPTION("Device '%s' is not suitable to this application: "
                                  "Device capability %d.%d < 3.0\n",
                                   prop.name,prop.major,prop.minor);
                }
            }
            break;
        case cudaErrorNoDevice:
            EXCEPTION("No GPU Device detected");
            break;
        case cudaErrorInsufficientDriver:
            EXCEPTION("No CUDA driver installed");
            break;
        default:
            EXCEPTION("Unknown error detecting GPU devices");
        }

        if(params.maxGPU > r)
            EXCEPTION("There are no sufficient GPU devices to allocate (req=%u, max=%d)",
                      params.maxGPU,r);
    }
#endif

    architecture = MLSA_CPU;
    improvement  = MLSI_BEST;
    heuristic    = MLSH_RVND;

    history = NULL;

    memset(&stats,0,sizeof(stats));

    gpuTasks = NULL;
    gpuTaskCount = 0;

    // Define random seed
    if(params.rngSeed == 0)
        rngSeed = sysTimer();
    else
        rngSeed = params.rngSeed;

    // Initialize RNG
    rng.seed(rngSeed);

    // Diversification iterations
    maxDiver = 0;
    iterDiver = 0;

    // Intensification iterations
    maxInten = 0;
    iterInten = 0;

    for(int i=0;i < MLP_MAX_ALPHA;i++)
        alpha[i] = i / 100.0F;

    logExec = NULL;

#ifdef MLP_COST_LOG
    logCostFile = NULL;
    logCostRef  = 0;
    logCostTime = 0;
    logCostPeriod = MS2US(params.logCostPeriod);
    logCostLast   = 0;
    pthread_mutex_init(&logCostMutex,NULL);
#endif

    solSolver = NULL;
}

MLSolver::~MLSolver()
{
#ifdef MLP_COST_LOG
    pthread_mutex_destroy(&logCostMutex);
#endif

    logClose();

    term();

    for(int id=0;id < gpuTaskCount;id++) {
        gpuSetDevice(id);
        gpuDeviceReset();
    }
}

void
MLSolver::logCreate()
{
   const
   char *odir;
   char  path[256],
         bname[64];

   if(params.logDisabled)
       return;

   if(params.logResult) {
       strcpy(bname,problem.filename);
       basename(bname);
       stripext(bname);

       odir = params.outputDir;
        if(odir == NULL)
           odir = ".";

       sprintf(path,"%s/%s.log",odir,bname);

        if((logExec = fopen(path,"wt")) == NULL)
            EXCEPTION("Error creating log file '%s'.\n",path);
   }
   else
       logExec = stdout;
}

void
MLSolver::logClose()
{
    if(logExec && (logExec != stdout))
        fclose(logExec);

    logExec = NULL;
}

void
MLSolver::logHeader()
{
    time_t     now;
    struct tm  tm;
    char       buffer[64];

    if(params.logDisabled)
        return;

    // Log command line
    fprintf(logExec,"<<< COMMAND\n");

    for(int i=0;i < params.argc;i++)
        fprintf(logExec,"%s ",params.argv[i]);

    fprintf(logExec,"\n>>> COMMAND\n");

    // Log app parameters
    fprintf(logExec,"\n<<< HEADER\n");

    // Get build date/time
    strptime(__DATE__ " " __TIME__,"%b %2d %Y %H:%M:%S",&tm);
    strftime(buffer,sizeof(buffer),"%Y-%m-%d %H:%M:%S",&tm);
    fprintf(logExec,"DATE_BUILD\t: %s\n",buffer);

    // Get current date/time
    now = time(NULL);
    tm  = *localtime(&now);

    strftime(buffer,sizeof(buffer),"%Y-%m-%d %H:%M:%S",&tm);
    fprintf(logExec,"DATE_EXEC\t: %s\n",buffer);

    fprintf(logExec,"INSTANCE\t: %s\n",problem.filename);
    fprintf(logExec,"SIZE\t\t: %u\n",problem.size);
//    fprintf(logExec,"CANDS\t\t: %u\n",problem.cands);
//    fprintf(logExec,"WIDTH\t\t: %u\n",problem.width);
//    fprintf(logExec,"HEIGHT\t\t: %u\n",problem.height);
    fprintf(logExec,"TIME_LOAD\t: %llu ms\n",US2MS(problem.timeLoad));
    fprintf(logExec,"NODE_ID\t\t: 0\n");

    gethostname(buffer,sizeof(buffer));
    fprintf(logExec,"NODE_NAME\t: %s\n",buffer);

    fprintf(logExec,"NODE_QTY\t: 1\n");
    fprintf(logExec,"THREAD_ID\t: %u\n",0);
    //fprintf(logExec,"THREAD_QTY\t: %u\n",taskCount + 1);
    fprintf(logExec,"RAND_SEED\t: %u\n",rngSeed);
    fprintf(logExec,">>> HEADER\n\n");
}

void
MLSolver::logFooter()
{
    if(params.logDisabled)
        return;

    // Log solution
//    fprintf(logExec,"<<< SOLUTION\n");
//    fprintf(logExec,"%s\n",taskTypeNames[bestArch]);
//    fprintf(logExec,"%u\n",bestSoluton->cost);
//    for(uint i=0;i < bestSoluton->size();i++) {
//        fprintf(logExec,"%u",bestSoluton->getCandNo(i));
//        if(i < bestSoluton->size() - 1)
//            fprintf(logExec,",");
//        else
//            fprintf(logExec,"\n");
//    }
    //    fprintf(logExec,">>> SOLUTION\n");
    fprintf(logExec,"#####################################################\n");
}

void
MLSolver::logResult()
{
    float  speedup;

    if(params.logDisabled)
        return;

/*
    fprintf(logExec,"<<< CPU\n");
    fprintf(logExec,"CPU_SOLUTIONS\t: %u\n",stats.cpuSolCount);
    fprintf(logExec,"CPU_BEST_COST\t: %s\n",cpuSolution->strCost().c_str());
    fprintf(logExec,"CPU_SOLVE_TIME\t: %llu\n",US2MS(stats.timeCPUSolver));
    fprintf(logExec,">>> CPU\n\n");

    fprintf(logExec, "<<< GPU\n");
    fprintf(logExec, "GPU_SOLUTIONS\t: %u\n",stats.gpuSolCount);
    fprintf(logExec, "GPU_BEST_COST\t: %s\n",gpuSolution->strCost().c_str());
    fprintf(logExec, "GPU_SOLVE_TIME\t: %llu\n", US2MS(stats.timeGPUSolver));
    fprintf(logExec, "GPU_COMMTX_TIME\t: %llu\n", US2MS(stats.timeCommTx));
    fprintf(logExec, "GPU_COMMRX_TIME\t: %llu\n", US2MS(stats.timeCommRx));

    fprintf(logExec, "GPU_BLOCKS\t: %u\n",groupCount);
    fprintf(logExec, "GPU_THEADS/BLOCK: %u\n",groupSize);
    fprintf(logExec, "GPU_THEADS\t: %u\n",solutionCount);
    fprintf(logExec, "GPU_SHARED_MEM\t: %0.2f KB\n", gpuData.sharedSize / 1024.0F);
    fprintf(logExec, "GPU_OCCUPANCY\t: %.1f%%\n", gpuOccupancy * 100.0F);
    fprintf(logExec, ">>> GPU\n\n");

    speedup = stats.timeGPUSolver + stats.timeCommRx + stats.timeCommTx;
    speedup = (speedup > 0) ? float(stats.timeCPUSolver) / speedup : 0.0F;

    fprintf(logExec, "<<< CPU x GPU\n");
    fprintf(logExec, "BEST_COST\t: %s\n",bestSoluton->strCost().c_str());
    fprintf(logExec, "BEST_ARCH\t: %s\n",taskTypeNames[bestArch]);
    fprintf(logExec, "SPEEDUP\t\t: %0.1fX\n", speedup);
    fprintf(logExec, ">>> CPU x GPU\n\n");
*/
}

#ifdef MLP_COST_LOG

void
MLSolver::logCostCreate()
{
    char   path[128];

    if(params.logCostFile == NULL) {
        logCostFile = NULL;
        logCostRef  = 0;
        logCostTime = 0;
        return;
    }

    if(*params.logCostFile == '/')
        sprintf(path,"%s",params.logCostFile);
    else
        sprintf(path,"%s/%s",params.outputDir,params.logCostFile);
    l4printf("Creating log cost file: %s\n",path);
    if((logCostFile = fopen(path,"wt")) == NULL)
        EXCEPTION("Error creating improvement log file: %s",path);

    // Instance name
    fprintf(logCostFile,"%s\n",problem.name);

    // time, cost, improv, kernel, GPU, BI/MI
    fprintf(logCostFile,"TIME\tCOST\tIMPROV\tNS\tGPU\n");
}

void
MLSolver::logCostClose()
{
    if(logCostFile)
        fclose(logCostFile);

    logCostFile = NULL;
    logCostRef  = 0;
    logCostTime = 0;
}

void
MLSolver::logCostWrite(uint cost, int improv, MLKernel *kernel, uint gpuId)
{
    pthread_mutex_lock(&logCostMutex);

    logCostTime = sysTimer();

    lprintf("logCostFile=%p\n",logCostFile);

    // time, cost, improv, kernel, GPU
    fprintf(logCostFile,"%llu\t%u\t%d\t%d\t%u\n",
                    US2MS(sysTimer() - logCostRef),
                    cost,
                    improv,
                    kernel ? kernel->id : -1,
                    gpuId);

    pthread_mutex_unlock(&logCostMutex);
}

#endif

void
MLSolver::init()
{
    uint    nsk;
    int     count;

    l4printf("seed=%u\n",rngSeed);

    // Get number of CPU cores on system
    count = sysconf(_SC_NPROCESSORS_ONLN);

    // Check for CPU usage limits
    if((params.maxCPU > 0) && (count > params.maxCPU))
        count = params.maxGPU;
    // Number of GPU devices to use
    gpuTaskCount = count;

    // Get number of GPU devices on system
    gpuGetDeviceCount(&count);

    // Check for GPU usage limits
    if((params.maxGPU > 0) && (count > params.maxGPU))
        count = params.maxGPU;
#ifndef GPU_CUDA_DISABLED
    // Number of GPU devices to use
    gpuTaskCount = count;
#else
    // Number of GPU devices to use
    gpuTaskCount = 1;
#endif

    // Destroy any history from some previous instance
    if(history)
        delete history;

    // Create history
    history = new MLKernelHistory(problem,rng,gpuTaskCount > 1);

    // Best solution
    solSolver = new MLSolution(problem,cudaHostAllocPortable);
}

void
MLSolver::term()
{
    if(history)
        delete history;
    history = NULL;

    if(solSolver)
        delete solSolver;
    solSolver = NULL;

    termTasks();
}

void
MLSolver::initTasks(MLHeuristic heur, MLArch arch, MLImprovement improv)
{
    termTasks();

    // Create GPU manager tasks
    gpuTasks = new PMLGPUTask[gpuTaskCount];
    for(uint i=0;i < gpuTaskCount;i++) {
        gpuTasks[i] = new MLGPUTask(*this,i,
                                    arch,
                                    heur,
                                    improv,
                                    params.nsKernel[i]);
        gpuTasks[i]->lsCall = 0;
        gpuTasks[i]->logCreate();
        fprintf(gpuTasks[i]->logExec,"Time\tCall\tSin\tSout\tNS\tMIC\tGmin\tGmax\tGavg\tGsum\tTkrn\tTbi\tTmi\n");

    }
}

void
MLSolver::termTasks()
{
    if(gpuTasks == NULL)
        return;

    for(uint i=0;i < gpuTaskCount;i++) {
        delete gpuTasks[i];
        gpuTasks[i] = NULL;
    }
    delete[] gpuTasks;
    gpuTasks = NULL;
}

void
MLSolver::launchTasks()
{
    pthread_attr_t attr;
    int            error,
                   i;

    /*
     * Launch threads
     */
    // Initialize and set thread detached attribute
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

    for(i=0;i < gpuTaskCount;i++) {
        l4printf("Launching task id=%u\n",gpuTasks[i]->gpuId);
        if((error = pthread_create(&gpuTasks[i]->thread,&attr,MLSolver::threadMain,(void *) gpuTasks[i])) != 0)
            EXCEPTION("Error launching GPU thread (error = %d)", error);
    }
    pthread_attr_destroy(&attr);
}

void
MLSolver::joinTasks()
{
    void   *status;
    int     error;
    /*
     * Join threads
     */
    for(uint i=0;i < gpuTaskCount;i++) {
        if((error = pthread_join(gpuTasks[i]->thread,&status)) != 0)
            EXCEPTION("Error joining thread (error = %d): %s",error,strerror(error));
        if(status == NULL)
            EXCEPTION("Invalid thread result");
    }
}

void *
MLSolver::threadMain(void *data)
{
    l4tracef("GPU%d\n",PMLGPUTask(data)->gpuId);

    if(PMLTask(data)->type == TTYP_GPUTASK) {
        MLGPUTask *gtask;

        // Get task
        gtask = PMLGPUTask(data);

        // Change status to RUNNING
        gtask->status = TSTA_RUNNING;

        // Set GPU device related to this host thread
        gpuSetDevice(gtask->gpuId);

#ifdef GPU_PROFILE
        char name[64];
        // Set name for GPU device managed by this thread
        sprintf(name,"GPU%u",gtask->gpuId);
        nvtxNameCuDevice(gtask->gpuId,name);
        // Set name for GPU manager thread
        sprintf(name,"GPU%u_MAIN",gtask->gpuId);
        nvtxNameOsThread(gtask->thread,name);
#endif

        // Call main method
        gtask->main();

        // Change status to STOPPED
        gtask->status = TSTA_STOPPED;
    }

    l4tracef("GPU%d\n",PMLGPUTask(data)->gpuId);

    return data;
}

/*
ullong
MLSolver::run(MLSolution *solBase, MLSolution *solResult, MLHeuristic heur, MLArch arch, MLMoveSelect movsel)
{
    MLSolution *solBest;
    ullong      time;

    l4printf("Running %s/%s/%s\n",
                    nameHeuristic[heur],
                    nameArch[arch],
                    nameMoveSel[movsel]);

    for(uint i=0;i < gpuTaskCount;i++)
        gpuTasks[i]->setSolution(solBase);

    launchTasks();
    joinTasks();

    time    = gpuTasks[0]->timeExec;
    solBest = gpuTasks[0]->solResult;
    for(uint i=1;i < gpuTaskCount;i++) {
        if(gpuTasks[i]->solResult->cost < solBest->cost)
            solBest = gpuTasks[i]->solResult;
    }
    solResult->assign(solBest,false);

    return time;
}
*/

void
MLSolver::solve()
{
    ullong  time;

    if(problem.filename == NULL)
        EXCEPTION("Error solving problem: no problem instance loaded");

    l4printf("Executable %s\n",problem.params.argv[0]);
    l4printf("Solving %s\n",problem.filename);

#ifdef GPU_PROFILE
    // Set name for main thread
    nvtxNameOsThread(pthread_self(),"CPU_MAIN");
#endif

    // Register starting time
    stats.timeStart = sysTimer();

    // Allocate resources
    init();

    // Create app log file
    logCreate();
    // Log header
    logHeader();

    // Run method
    main();

    // Compute execution time
    stats.timeExec = stats.timeStart - sysTimer();

    // Log result
    logResult();
    // Log footer
    logFooter();
    // Close log file
    logClose();
}

void
MLSolver::main()
{
    /*
     * execExperiment(MLHeuristic heur, int arch, bool merge);
     */
    l4printf("Experiment #%d\n",params.experNo);

    switch(params.experNo) {
    case 0:
        // Nothing to do
        break;
    case 1:
        // RVND/CPU/BI
        execExperiments(MLSH_RVND,MLSA_CPU,MLSI_BEST);
        break;
    case 2:
        // RVND/GPU/BI
        execExperiments(MLSH_RVND,MLSA_SGPU,MLSI_BEST);
        break;
    case 3:
        // RVND/GPU/MI
        execExperiments(MLSH_RVND,MLSA_SGPU,MLSI_MULTI);
        break;
    case 4:
        // DVND/SGPU/BI
        execExperiments(MLSH_DVND,MLSA_SGPU,MLSI_BEST);
        break;
    case 5:
        // DVND/SGPU/MI
        execExperiments(MLSH_DVND,MLSA_SGPU,MLSI_MULTI);
        break;
    case 6:
        // DVND/MGPU/BI
        execExperiments(MLSH_DVND,MLSA_MGPU,MLSI_BEST);
        break;
    case 7:
        // DVND/MGPU/MI
        execExperiments(MLSH_DVND,MLSA_MGPU,MLSI_MULTI);
        break;
    case 8:
        // GDVND/SGPU/MI
        execGdvnd(MLSA_SGPU,MLSI_MULTI);
        break;
    case 9:
        // GDVND/MGPU/MI
        execGdvnd(MLSA_MGPU,MLSI_MULTI);
        break;
    case 10:
        // GRVND/SGPU/BI
        execGrvnd(MLSA_SGPU,MLSI_BEST);
        break;
    case 11:
        // GRVND/SGPU/MI
        execGrvnd(MLSA_SGPU,MLSI_MULTI);
        break;
    case 14:
        // testKernelTime()
        showSearchTime();
        break;
    case 15:
        // testKernelTime()
        showKernelTime();
        break;
    case 90:
        // testShowDist
        testShowDist();
        break;
    case 91:
        // testKernelSearch()
        testKernelSearch();
        break;
    case 92:
        // testKernelSearch()
        testDvnd();
        break;
    default:
        EXCEPTION("Invalid experiment number %d. Define option '--exper'",problem.params.experNo);
    }
}

float
calcGap(double v1, double v2)
{
    return float((v2 - v1) / v1);
}

void
MLSolver::perturb(MLSolution &solBase, MLSolution &solPert)
{
    uint    cuts[4];
    uint    i;

    if (solBase.size < 6) {
        solPert.assign(solBase,false);
        return;
    }

    // Generate four random cut points and sort then
    rng.set(0,solBase.clientCount - 2,cuts,4);
    std::sort(cuts,cuts + 4);

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

ullong
MLSolver::launchDvnd(MLSolution *solBase, MLSolution *solDvnd, MLSolverStats *stats)
{
    ullong      time;

    // Update LDS data
    solBase->ldsUpdate();

    // Initialize history
    history->clear();
    history->add(solBase);

//    lprintf("GDVND dvndMI(%u)\n",solBase->cost);
//    history->show("\tBefore\tH=");

    // Launch DVND tasks
    launchTasks();
    // Join DVND tasks
    joinTasks();

    if(!history->best(solDvnd))
        EXCEPTION("Something wrong: no best solution in History!");

//    history->show("\tAfter\tH=");
//    lprintf("GDVND dvndMI(%u) --> %u\n",solBase->cost,solDvnd->cost);

    time = 0;
    for(int i=0;i < gpuTaskCount;i++) {
        MLGPUTask *task;
        uint       id;

        task = gpuTasks[i];

        if(task->timeExec > time)
            time = task->timeExec;

        if(stats) {
            for(int j=0;j < task->kernelCount;j++) {

                id = task->kernels[j]->id;

                stats->calls[id]  += task->kernels[j]->callCount;
                stats->callsTotal += task->kernels[j]->callCount;

                stats->imprv[id]  += task->kernels[j]->imprvCount;
                stats->imprvTotal += task->kernels[j]->imprvCount;

                stats->merge[id]  += task->kernels[j]->mergeCount;
                stats->mergeTotal += task->kernels[j]->mergeCount;

                stats->mergeTime  += task->kernels[j]->timeMove;
            }
        }
    }

    // Returns elapsed time
    return time;
}

ullong
MLSolver::gdvnd(const MLArch arch, const MLImprovement improv, MLSolverStats *stats)
{
    MLSolution  solGrasp(problem),
                solDvnd(problem),
                solIls(problem);
    ullong      time;
    uint        i;

    if(arch == MLSA_CPU)
        EXCEPTION("There's no GDVND for CPU");

    heuristic    = MLSH_DVND;
    architecture = arch;
    improvement  = improv;

    maxDiver = problem.params.maxDiver;
    maxInten = (problem.size < problem.params.maxInten) ? problem.size : problem.params.maxInten;

    l4printf("Running GDVND(diver=%u,inten=%u)\n",maxDiver,maxInten);

    // LOG_COST_CREATE();

    // Create tasks
    initTasks(heuristic,architecture,improvement);

    // Initialize tasks base solution
    for(i=0;i < gpuTaskCount;i++)
        gpuTasks[i]->clearSolution();

    // Save starting time
    time = sysTimer();

    // initially best solution is empty (cost = +oo)
    solSolver->clear();

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

        // Intensification loop
        iterInten = 0;
        while(iterInten < maxInten) {

            // Intensification iterations counter
            iterInten++;
            //stats.intenIters++;

            // performs a DVND search over 'lsolCurr'
            launchDvnd(&solGrasp,&solDvnd,stats);

            // if 'rvndSol' cost is best than 'lsolBest', update best solution
            if(solDvnd.cost < solIls.cost) {
                // Update best solution
                solIls.assign(solDvnd,false);

                l4printf("GDVND IMPROV(%u) --> %u\n",solGrasp.cost,solIls.cost);

                // Reset intensify load
                iterInten = 0;
            }

            // perturb current solution
            if(iterInten < maxInten - 1) {
                perturb(solIls,solGrasp);
                l4printf("\tPERTB(%u) --> %u\n",solIls.cost,solGrasp.cost);
            }
        }

        // update best device solution
        if(solIls.cost < solSolver->cost) {
            solSolver->assign(solIls,false);
            lprintf("GDVND best=%u\n",solSolver->cost);
        }
    }

    time = sysTimer() - time;

    l4printf("GDVND best=%u\n",solSolver->cost);

    termTasks();

    //LOG_COST_CLOSE();

    return time;
}

ullong
MLSolver::launchRvnd(MLSolution *solBase, MLSolution *solRvnd, MLSolverStats *stats)
{
    MLSolution *solBest;
    ullong      time;

    for(uint i=0;i < gpuTaskCount;i++)
        gpuTasks[i]->setSolution(solBase);

    // Launch DVND tasks
    launchTasks();
    // Join DVND tasks
    joinTasks();

    solBest = gpuTasks[0]->solResult;
    time = 0;
    for(int i=0;i < gpuTaskCount;i++) {
        MLGPUTask *task;
        uint       id;

        task = gpuTasks[i];

        if(task->solResult->cost < solBest->cost)
            solBest = task->solResult;

        if(task->timeExec > time)
            time = task->timeExec;

        if(stats) {
            for(int j=0;j < MLP_MAX_NEIGHBOR;j++) {
                id = task->kernels[j]->id;

                stats->calls[id]  += task->kernels[j]->callCount;
                stats->callsTotal += task->kernels[j]->callCount;

                stats->imprv[id]  += task->kernels[j]->imprvCount;
                stats->imprvTotal += task->kernels[j]->imprvCount;

                stats->merge[id]  += task->kernels[j]->mergeCount;
                stats->mergeTotal += task->kernels[j]->mergeCount;

                stats->mergeTime  += task->kernels[j]->timeMove;
            }
        }
    }

    solRvnd->assign(solBest,false);

    // Returns elapsed time
    return time;
}


ullong
MLSolver::grvnd(const MLArch arch, const MLImprovement improv, MLSolverStats *stats)
{
    MLSolution  solGrasp(problem),
                solRvnd(problem),
                solIls(problem);
    ullong      time;
    uint        i;

    heuristic    = MLSH_RVND;
    architecture = arch;
    improvement  = improv;

    maxDiver = problem.params.maxDiver;
    maxInten = (problem.size < problem.params.maxInten) ? problem.size : problem.params.maxInten;

    l4printf("Running GRVND(diver=%u,inten=%u)\n",maxDiver,maxInten);

    // LOG_COST_CREATE();

    // Create tasks
    initTasks(heuristic,architecture,improvement);

    // Initialize tasks base solution
    for(i=0;i < gpuTaskCount;i++)
        gpuTasks[i]->clearSolution();

    // Save starting time
    time = sysTimer();

    // initially best solution is empty (cost = +oo)
    solSolver->clear();

    // Intensification loop
    iterDiver = 0;
    while(iterDiver < maxDiver) {
        l4printf("****************************************\n");
        // Diversification iterations counter
        iterDiver++;
        //stats.diverIters++;

        // pick an alpha value from alpha set
        i = rng.rand(MLP_MAX_ALPHA - 1);

        // create a new solution
        solGrasp.random(rng,alpha[i]);

        // current best solution
        solIls.assign(solGrasp,false);

        // Intensification loop
        iterInten = 0;
        while(iterInten < maxInten) {

            // Intensification iterations counter
            iterInten++;
            //stats.intenIters++;

            l4printf("@@@@@ diver=%u\tinten=%u\n",iterDiver,iterInten);

            // performs a RVND search over 'lsolCurr'
            l4printf("<<< GRVND\trvnd(%u)\n",solGrasp.cost);
            launchRvnd(&solGrasp,&solRvnd,stats);
            l4printf(">>> GRVND\trvnd(%u) --> %u\n",solGrasp.cost,solRvnd.cost);

            // if 'rvndSol' cost is best than 'lsolBest', update best solution
            //TODO Compare with solSolver, the global best solution
            if(solRvnd.cost < solIls.cost) {
                // Update best solution
                solIls.assign(solRvnd,false);

                l4printf("GRVND IMPROV(%u) --> %u\n",solGrasp.cost,solIls.cost);

                // Reset intensify load
                iterInten = 0;
            }

            // perturb current solution
            if(iterInten < maxInten) {
                perturb(solIls,solGrasp);
                l4printf("\tPERTB(%u) --> %u\n",solIls.cost,solGrasp.cost);
            }
        }

        // update best device solution
        if(solIls.cost < solSolver->cost) {
            solSolver->assign(solIls,false);
            lprintf("GRVND best=%u\n",solSolver->cost);
        }
    }

    l4printf("GRVND best*=%u\n",solSolver->cost);

    time = sysTimer() - time;

    termTasks();

    //LOG_COST_CLOSE();

    return time;
}


void
MLSolver::experHeader(const MLHeuristic heur, const MLArch arch, const MLImprovement improv)
{
    char    buffer[1024];
    int     i,n;

    n = (arch == MLSA_CPU) ? 0 : gpuTaskCount;

    lprintf("Exper #%d\t%s/%s/%s\n",
                    problem.params.experNo,
                    nameHeuristic[heur],
                    nameArch[arch],
                    nameImprov[improv]);

    sprintf(buffer,"%s/%s/%s",
                    nameHeuristic[heur],
                    nameArch[arch],
                    nameImprov[improv]);

    lprintf("EXEC_FILE\t%s\n",params.argv[0]);
    lprintf("RAND_SEED\t%u\n",rng.getSeed());
    lprintf("SOLUTIONS\t%u\n",params.maxExec);
    lprintf("INSTANCE\t%s\t%s\n",problem.name,buffer);
    lprintf("CHECK\t%s\n",yesNo(params.checkCost));
    lprintf("ARCH\t%s\t%d\n",nameArch[architecture],n);
    lprintf("IMPROV\t%s\n",nameImprov[improv]);

    *buffer = '\0';
    if(params.cpuAds)
        strcat(buffer,"CPU");
    if(params.gpuAds) {
        if(*buffer)
            strcat(buffer,"\t");
        strcat(buffer,"GPU");
    }
    lprintf("ADS\t%s\n",buffer);

    /*
     * Print header
     */
    lprintf("Run\tCost\tCost\tCost\tTime");
    for(i=0;i <= MLP_MAX_NEIGHBOR;i++)
        lprintf("\tCalls");
    for(i=0;i <= MLP_MAX_NEIGHBOR;i++)
        lprintf("\tImprov");
    for(i=0;i <= MLP_MAX_NEIGHBOR;i++)
        lprintf("\t%s",nameImprov[improv]);
    lprintf("\t%s\n",nameImprov[improv]);

    lprintf("\tBase\t%s\tGap\t%s",
                    nameHeuristic[heur],
                    nameHeuristic[heur]);
    for(i=0;i < MLP_MAX_NEIGHBOR;i++)
        lprintf("\t%s",nameMove[i]);
    lprintf("\tTotal");
    for(i=0;i < MLP_MAX_NEIGHBOR;i++)
        lprintf("\t%s",nameMove[i]);
    lprintf("\tTotal");
    for(i=0;i < MLP_MAX_NEIGHBOR;i++)
        lprintf("\t%s",nameMove[i]);
    lprintf("\tTotal\tTime\n");
}

void
MLSolver::execGdvnd(const MLArch arch, const MLImprovement improv)
{
    MLGPUTask      *task;
    MLSolverStats   stats;
    ullong          time;
    uint            e,i,id;

    experHeader(MLSH_DVND,arch,improv);

    for(e=0;e < params.maxExec;e++) {

        // Reset stats
        memset(&stats,0,sizeof(stats));

        // Execute GDVND
        time = gdvnd(arch,improv,&stats);

        lprintf("%u\t%u\t%u\t%0.4f\t%llu",
                        e + 1,
                        0,
                        solSolver->cost,
                        0.0F,
                        US2MS(time)
        );

        for(i=0;i < MLP_MAX_NEIGHBOR;i++)
            lprintf("\t%u",stats.calls[i]);
        lprintf("\t%u",stats.callsTotal);

        for(i=0;i < MLP_MAX_NEIGHBOR;i++)
            lprintf("\t%u",stats.imprv[i]);
        lprintf("\t%u",stats.imprvTotal);

        for(i=0;i < MLP_MAX_NEIGHBOR;i++)
            lprintf("\t%u",stats.merge[i]);
        lprintf("\t%u\t%llu",stats.mergeTotal,
                             US2MS(stats.mergeTime));

        lprintf("\n");
    }
}

void
MLSolver::execGrvnd(const MLArch arch, const MLImprovement improv)
{
    MLGPUTask      *task;
    MLSolverStats   stats;
    ullong          time;
    uint            e,i,id;

    if(gpuTaskCount != 1)
        EXCEPTION("GRVND runs with 1 GPU only.");

    //experHeader(MLSH_RVND,arch,improv);

    for(e=0;e < params.maxExec;e++) {

        // Reset stats
        memset(&stats,0,sizeof(stats));

        // Execute GDVND
        time = grvnd(arch,improv,&stats);

        /*
        lprintf("%u\t%u\t%u\t%0.4f\t%llu",
                        e + 1,
                        0,
                        solSolver->cost,
                        0.0F,
                        US2MS(time)
        );

        for(i=0;i < MLP_MAX_NEIGHBOR;i++)
            lprintf("\t%u",stats.calls[i]);
        lprintf("\t%u",stats.callsTotal);

        for(i=0;i < MLP_MAX_NEIGHBOR;i++)
            lprintf("\t%u",stats.imprv[i]);
        lprintf("\t%u",stats.imprvTotal);

        for(i=0;i < MLP_MAX_NEIGHBOR;i++)
            lprintf("\t%u",stats.merge[i]);
        lprintf("\t%u\t%llu",stats.mergeTotal,
                             US2MS(stats.mergeTime));

        lprintf("\n");
        */
    }
}

void
MLSolver::execExperiments(const MLHeuristic heur, const MLArch arch, const MLImprovement improv)
{
    MLSolution     *solVnd;
    MLGPUTask      *task;
    MLSolverStats   stats;
    ullong          time;
    uint            i,j,k,n,v;
    uint            gpus;
    int             id;
    char            buffer[1024];

    heuristic    = heur;
    architecture = arch;
    improvement  = improv;

    l4printf("Running %s/%s/%s\n",
                    nameHeuristic[heuristic],
                    nameArch[architecture],
                    nameImprov[improvement]);

//    if((heuristic != MLSH_DVND) && (params.maxGPU > 1))
//        EXCEPTION("CPU heuristics can run in one thread (--max-gpu=%u)",params.maxGPU);

    if(architecture == MLSA_CPU) {
        if(improvement == MLSI_MULTI)
            EXCEPTION("There's no RVND CPU with move merge");
        if(heuristic == MLSH_DVND)
            EXCEPTION("There's no DVND CPU");

        gpus = 0;
    }
    else
        gpus = gpuTaskCount;

    LOG_COST_CREATE();

    // Create tasks
    initTasks(heuristic,architecture,improvement);

    /*
     * Print header
     */
    experHeader(heuristic,architecture,improvement);

#if 0
    lprintf("Exper #%d\t%s/%s/%s\n",
                    problem.params.experNo,
                    nameHeuristic[heuristic],
                    nameArch[architecture],
                    nameImprov[improvement]);

    sprintf(buffer,"%s/%s/%s",
                    nameHeuristic[heuristic],
                    nameArch[architecture],
                    nameImprov[improvement]);

    lprintf("EXEC_FILE\t%s\n",params.argv[0]);
    lprintf("RAND_SEED\t%u\n",rng.getSeed());
    lprintf("SOLUTIONS\t%u\n",params.maxExec);
    lprintf("INSTANCE\t%s\t%s\n",problem.name,buffer);
    lprintf("CHECK\t%s\n",yesNo(params.checkCost));
    lprintf("ARCH\t%s\t%u\n",nameArch[architecture],gpus);
    lprintf("IMPROV\t%s\n",nameImprov[improvement]);

    *buffer = '\0';
    if(params.cpuAds)
        strcat(buffer,"CPU");
    if(params.gpuAds) {
        if(*buffer)
            strcat(buffer,"\t");
        strcat(buffer,"GPU");
    }
    lprintf("ADS\t%s\n",buffer);

    lprintf("Run\tCost\tCost\tCost\tTime");
    for(j=0;j <= MLP_MAX_NEIGHBOR;j++)
        lprintf("\tCalls");
    for(j=0;j <= MLP_MAX_NEIGHBOR;j++)
        lprintf("\tImprov");
    for(j=0;j <= MLP_MAX_NEIGHBOR;j++)
        lprintf("\tMerge");
    lprintf("\n");

    lprintf("\tBase\t%s\tGap\t%s",
                    nameHeuristic[heuristic],
                    nameHeuristic[heuristic]);
    for(j=0;j < MLP_MAX_NEIGHBOR;j++)
        lprintf("\t%s",nameMove[j]);
    lprintf("\tTotal");
    for(j=0;j < MLP_MAX_NEIGHBOR;j++)
        lprintf("\t%s",nameMove[j]);
    lprintf("\tTotal");
    for(j=0;j < MLP_MAX_NEIGHBOR;j++)
        lprintf("\t%s",nameMove[j]);
    lprintf("\tTotal");
    lprintf("\n");
#endif

    /*
     * Run 'max' times loading previously solutions.
     */
    for(i=0;i < params.maxExec;i++) {
        sprintf(buffer,"%s/%s_%03u.sol",
                        params.inputDir ? params.inputDir : ".",
                        problem.name,
                        i + 1);

        l4printf("Solution %3u: ",i + 1);
        if(access(buffer,R_OK) == 0) {
            l4printf("loading from file '%s'\n",buffer);
            solSolver->load(buffer);
        }
        else {
            l4printf("creating randomly\n");
            solSolver->random(rng,0.50);
        }
        //solSolver->save(std::cout);
        solSolver->ldsUpdate();

        // Initialize tasks base solution
        for(j=0;j < gpuTaskCount;j++)
            gpuTasks[j]->setSolution(solSolver);

        // Initialize history
        history->clear();
        history->add(solSolver);

        // Launch threads
        launchTasks();
        // Wait for threads completion
        joinTasks();

        // Initialize execution statistics
        memset(&stats,0,sizeof(stats));

        l4printf("gpuTaskCount=%u\n",gpuTaskCount);

        // Collect statistics data
        solVnd = NULL;
        time   = 0;
        for(j=0;j < gpuTaskCount;j++) {
            // Get task
            task = gpuTasks[j];

            if(j > 0) {
                // min(cost)
                if(task->solResult->cost < solVnd->cost)
                    solVnd = task->solResult;

                // max(time)
                if(task->timeExec > time)
                    time = task->timeExec;
            }
            else {
                time   = task->timeExec;
                solVnd = task->solResult;
            }
            l4printf("j=%d: solVnd=%u\n",j,solVnd->cost);

            if(architecture == MLSA_CPU) {
                for(k=0;k < task->searchCount;k++) {
                    id = task->searches[k]->searchId;

                    stats.calls[id]  += task->searches[k]->callCount;
                    stats.callsTotal += task->searches[k]->callCount;

                    stats.imprv[id]  += task->searches[k]->imprvCount;
                    stats.imprvTotal += task->searches[k]->imprvCount;
                }
            }
            else {
                for(k=0;k < task->kernelCount;k++) {
                    id = task->kernels[k]->id;

                    stats.calls[id]  += task->kernels[k]->callCount;
                    stats.callsTotal += task->kernels[k]->callCount;

                    stats.imprv[id]  += task->kernels[k]->imprvCount;
                    stats.imprvTotal += task->kernels[k]->imprvCount;

                    stats.merge[id]  += task->kernels[k]->mergeCount;
                    stats.mergeTotal += task->kernels[k]->mergeCount;

                    stats.mergeTime  += task->kernels[k]->timeMove;
                }
            }
        }

//        sprintf(buffer,"/tmp/eil101_%02d.sol",i);
//        solVnd->save(buffer);

        k = (solSolver->cost != COST_INFTY) ? solSolver->cost : 0;
        lprintf("%u\t%u\t%u\t%0.4f\t%llu",
                        i + 1,
                        k,
                        solVnd->cost,
                        calcGap(solSolver->cost,solVnd->cost),
                        US2MS(time)
        );

        for(j=0;j < MLP_MAX_NEIGHBOR;j++)
            lprintf("\t%u",stats.calls[j]);
        lprintf("\t%u",stats.callsTotal);

        for(j=0;j < MLP_MAX_NEIGHBOR;j++)
            lprintf("\t%u",stats.imprv[j]);
        lprintf("\t%u",stats.imprvTotal);

        for(j=0;j < MLP_MAX_NEIGHBOR;j++)
            lprintf("\t%u",stats.merge[j]);
        lprintf("\t%u\%llu",stats.mergeTotal,
                            US2MS(stats.mergeTime));

        lprintf("\n");
    }

    termTasks();

    LOG_COST_CLOSE();
}

void
MLSolver::testShowDist()
{
    initTasks(heuristic,architecture,improvement);

    launchTasks();
    joinTasks();

    termTasks();
}

void
MLSolver::testDvnd()
{
#if 0
    MLKernel    *kernel;
    MLSolution  *solVnd;
    MLMove       move;
    uint         i,n,max;
    ullong       time;
    char         buffer[128];

    lprintf("GPU%u ENTER Experiment %s()\n\tkernels:",gpuId,__FUNCTION__);
    for(i=0;i < kernelCount;i++)
        printf(" %s",kernels[i]->name);
    printf("\n");

    solVnd = new MLSolution(problem);

#if 0
        short   ids[] = { 0,5,1,6,3,2,4 };
        int     n = sizeof(ids) / sizeof(*ids);

        solDevice->clear();
        for(int i=0;i < n;i++)
            solDevice->add(ids[i]);
        if(problem.params.costTour)
            solDevice->add(ids[0]);
        solDevice->update();

        if(solDevice->clientCount != problem.size)
            EXCEPTION("Invalid solution size: sol=%u, prob=%u\n",solDevice->clientCount,problem.size);
#else
        solDevice->random(rng,0.50);
#endif
        solDevice->adsUpdate();

    for(uint m=0;m < params.maxDiver;m++) {

        lprintf("----------------------------------------\n");
        lprintf("\tGPU%u: Exec #%u\n",gpuId,m + 1);

        dvnd(solDevice,solVnd);

        n = 0;
        for(i=0;i < kernelCount;i++)
            n += kernels[i]->callCount;


        lprintf("\tGPU%d: base=%u\tdvnd=%u\tcalls=%u\n",
                        gpuId,solDevice->cost,solVnd->cost,n);
    }

    delete solVnd;

    l4printf("GPU%u LEAVE Experiment %s()\n",gpuId,__FUNCTION__);
#endif
}

void
MLSolver::testKernelSearch()
{
#if 0
    MLKernel    *kernel;
    MLSolution  *solVnd;
    MLMove       move;
    uint         i,max;
    ullong       time;
    char         buffer[128];

    lprintf("GPU%u ENTER Experiment %s()\n",gpuId,__FUNCTION__);

    solVnd = new MLSolution(problem);

    l4printf("GPU%u Solution created\n",gpuId);

    for(uint m=0;m < params.maxDiver;m++) {

        lprintf("GPU%u: Exec #%u\n",gpuId,m + 1);

#if 0
        short   ids[] = { 0,5,1,6,3,2,4 };
        int     n = sizeof(ids) / sizeof(*ids);

        solDevice->clear();
        for(int i=0;i < n;i++)
            solDevice->add(ids[i]);
        if(problem.params.costTour)
            solDevice->add(ids[0]);
        solDevice->update();

        if(solDevice->clientCount != problem.size)
            EXCEPTION("Invalid solution size: sol=%u, prob=%u\n",solDevice->clientCount,problem.size);
#else
        solDevice->random(rng,0.50);
#endif
        solDevice->adsUpdate();

        lprintf("GPU%u: kernelCount = %u\n",gpuId,kernelCount);

        for(int k=0;k < kernelCount;k++) {
            kernel = kernels[k];

            lprintf("GPU%d: kernel %s\n",gpuId,kernel->name);

            kernel->setSolution(solDevice);
            kernel->sendSolution();
            kernel->launchKernel();
            kernel->recvResult();
            kernel->sync();

            l4tracef("GPU%d\n",gpuId);
            kernel->bestMove(move);
            l4tracef("GPU%d\n",gpuId);
            kernel->applyMove(move);
            l4tracef("GPU%d\n",gpuId);
        }

        lprintf("GPU%d: -----------------------------------------\n",gpuId);
    }

    delete solVnd;

    lprintf("GPU%u LEAVE Experiment %s()\n",gpuId,__FUNCTION__);
#endif
}

void
MLSolver::testMergeMove()
{
#if 0
    MLKernel    *kernel;
    MLSolution  *solVnd;
    MLMove       move;
    uint         i,max;
    ullong       time,timeAvg;
    int          movesCount,
                 movesCost;
    char         buffer[128];

    solVnd = new MLSolution(problem);

    lprintf("RAND_SEED\t: %u\n",rng.getSeed());

    timeAvg = 0;
    for(uint m=0;m < problem.params.maxDiver;m++) {

        l4printf("***\n* Solution #%u\n***\n",m + 1);

#if 1
        short   ids[] = { 0,5,1,6,3,2,4 };
        int     n = sizeof(ids) / sizeof(*ids);

        solDevice->clear();
        for(int i=0;i < n;i++)
            solDevice->add(ids[i]);
        if(problem.params.costTour)
            solDevice->add(ids[0]);

        if(solDevice->clientCount != problem.size)
            EXCEPTION("Invalid solution size: sol=%u, prob=%u\n",solDevice->clientCount,problem.size);

        solDevice->update();
#else
        solDevice->random(rng,0.50);
#endif
        solDevice->adsUpdate();

#if 0
        for(int k=MLMI_SWAP;k <= MLMI_OROPT3;k++) {
            kernel = kernels[k];

            kernel->setSolution(solDevice);
            kernel->sendSolution();
            kernel->launchKernel();
            kernel->recvResult();
            kernel->sync();

            l4printf("time = %llu us = %llu ms\n",time,time / 1000);

            movesCost = kernel->mergeGreedy(mergeBuffer,movesCount);

            lprintf("-- Merged %4d %s moves\tmerge_cost=%d\tsol_cost=%d\n",
                                movesCount,kernel->name,movesCost,kernel->solution->cost);
//            for(int j=0;j < mergeCount;j++)
//                lprintf("(%d,%d,%d)",mergeBuffer[j].i,mergeBuffer[j].j,mergeBuffer[j].cost);
//            lprintf("\n");

            while(movesCount > 0) {
                move64ToMove(move,mergeBuffer[--movesCount]);
                l4printf("Apply %s(%d,%d) = %d\n",kernel->name,move.i,move.j,move.cost);
                kernel->applyMove(move);
            }
            l4printf("Cost improvement: %d --> %d = %d\n",
                            int(solDevice->cost),
                            int(kernel->solution->cost),
                            int(kernel->solution->cost) - int(solDevice->cost));
        }
        lprintf("-----------------------------------------\n");

#else
        ullong rtime;

        time  = rvndCPU(solDevice,solVnd);
        rtime = time;
        lprintf("RVND CPU\t: %6d --> %6d\t%6.2f%%\t%6llu ms\t%+8.2lf%%\n",
                        solDevice->cost,solVnd->cost,
                       (solDevice->cost - solVnd->cost) / float(solDevice->cost) * 100.0F,
                       US2MS(time),
                       (double(time) - double(rtime)) / double(rtime) * 100.0);

        time = dvnd(solDevice,solVnd);
        lprintf("BEST MOVE\t: %6d --> %6d\t%6.2f%%\t%6llu ms\t%+8.2lf%%\n",
                        solDevice->cost,solVnd->cost,
                       (solDevice->cost - solVnd->cost) / float(solDevice->cost) * 100.0F,
                       US2MS(time),
                       (double(time) - double(rtime)) / double(rtime) * 100.0);

        time = dvndMerge(solDevice,solVnd);
        lprintf("MERGE MOVE\t: %6d --> %6d\t%6.2f%%\t%6llu ms\t%+8.2lf%%\n",
                        solDevice->cost,solVnd->cost,
                       (solDevice->cost - solVnd->cost) / float(solDevice->cost) * 100.0F,
                       US2MS(time),
                       (double(time) - double(rtime)) / double(rtime) * 100.0);
        lprintf("\n");

#endif
    }

    delete solVnd;
#endif
}

void
MLSolver::showSearchTime()
{
#if 0
    MLSearch    *search;
    MLMove       move;
    ullong       time,
                 timeSum[MLP_MAX_NEIGHBOR];
    int          i,m,max;

    max = params.maxDiver;

    memset(timeSum,0,sizeof(timeSum));

    lprintf("PROBLEM\t%s\n",problem.name);
    lprintf("TIME\tmicroseconds (us)\n");
    lprintf("ARCH\tCPU\n");
    lprintf("ADS\t%s\n",yesNo(params.cpuAds));

    lprintf("Run");
    for(i=0;i < searchCount;i++)
        lprintf("\t%s",searches[i]->name);
    lprintf("\n");

    for(m=0;m < max;m++) {
        lprintf("%d",m + 1);

        solDevice->random(rng,0.5);
        solDevice->adsUpdate();

        for(i=0;i < searchCount;i++) {
            search = searches[i];

            time = sysTimer();

            search->launchSearch(solDevice,move);

            time = sysTimer() - time;

            lprintf("\t%llu",time);

            timeSum[i] += time;
        }
        lprintf("\n");
    }

    lprintf("\nAVG");
    for(i=0;i < kernelCount;i++)
        lprintf("\t%llu",timeSum[i] / max);

    lprintf("\n");
#endif
}

void
MLSolver::showKernelTime()
{
    MLKernel    *kernel;
    MLMove       move;
    ullong       time,
                 timeSum[MLP_MAX_NEIGHBOR];
    int          i,m,max;

    max = params.maxDiver;

    memset(timeSum,0,sizeof(timeSum));

    lprintf("PROBLEM\t%s\n",problem.name);
    lprintf("TIME\tmicroseconds (us)\n");
    lprintf("ARCH\tGPU\n");
    lprintf("ADS\t%s\n",yesNo(params.gpuAds));

    lprintf("Run");
    for(i=0;i < MLP_MAX_NEIGHBOR;i++)
        lprintf("\t%s",nameMove[i]);
    lprintf("\n");

#if 0
    for(m=0;m < max;m++) {
        lprintf("%d",m + 1);

        solDevice->random(rng,0.5);
        solDevice->adsUpdate();

        for(i=0;i < kernelCount;i++) {
            kernel = kernels[i];

            time = sysTimer();

            kernel->setSolution(solDevice);
            kernel->sendSolution();
            kernel->launchKernel();
            kernel->recvResult();
            kernel->sync();
            kernel->bestMove(move);

            time = sysTimer() - time;

            lprintf("\t%llu",time);

            timeSum[i] += time;
        }
        lprintf("\n");
    }

    lprintf("\nAVG");
    for(i=0;i < kernelCount;i++)
        lprintf("\t%llu",timeSum[i] / max);
    lprintf("\n");
#endif
}

void
MLSolver::checkBestMoveCPUxGPU()
{
#if 0
    MLSearch *search;
    MLKernel *kernel;
    MLMove    smove,kmove;
    int       i,d,k;

    for(d=0;d < problem.params.maxDiver;d++) {
        printf("-------------------------------\n");

        solDevice->clear();
#if 0
        //ushort    base[] = { 0,2,9,4,1,5,3,7,6,8 };
        ushort    base[] = { 0,5,1,6,3,2,4 };
        int       baseSize = sizeof(base) / sizeof(*base);

        for(i=0;i < baseSize;i++)
            solDevice->add(base[i]);
        if(problem.params.costTour)
            solDevice->add(base[0]);
        solDevice->update();
#else
        solDevice->random(rng,0.50);
#endif

        solDevice->adsUpdate();
        solDevice->show();

        printf("\n");

        for(k=0;k < kernelCount;k++) {
            search = searches[k];
            kernel = kernels[k];

            search->launchSearch(solDevice,smove);

            kernel->setSolution(solDevice);
            kernel->sendSolution();
            kernel->launchKernel();
            kernel->recvResult();
            gpuDeviceSynchronize();
            kernel->bestMove(kmove);

            printf("CPU_%s(%u,%u)\t%5d\tGPU_%s(%u,%u)\t%5d\n",
                            search->name,smove.i,smove.j,smove.cost,
                            kernel->name,kmove.i,kmove.j,kmove.cost);

            if(smove.cost != kmove.cost)
                EXCEPTION("Wrong CPU/GPU search result");
        }
    }
#endif
}
