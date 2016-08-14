#ifndef WAMCA_EXPERIMENT_HPP
#define WAMCA_EXPERIMENT_HPP

#include "except.h"
#include "log.h"
#include "gpu.h"
#include "utils.h"

#include "mlproblem.h"
#include "mlsolution.h"
//#include "mlparser.h"
//#include "mlsolver.h"
#include "mlkernel.h"

#include "mlkswap.h"
#include "mlkoropt.h"
#include "mlk2opt.h"


class WAMCAExperiment
{
public:

	WAMCAExperiment(MLProblem& _problem) :
		problem(_problem)
	{
	}


	MLProblem problem;

	MTRandom         rng;               ///< Random number generator

	uint            gpuId;                      ///< GPU id
	uint            gpuMemFree;                 ///< Amount of free memory on GPU
	cudaDeviceProp  gpuProps;                   ///< Device properties

	MLKernel       *kernels[MLP_MAX_NEIGHBOR];  ///< Kernel searches
	uint            kernelCount;                ///< Number of kernels
	uint            kernelBits;                 ///< Bit mask for active kernels


	ullong          lsCall;                     ///< Local search call no.

	MLSolution     *solDevice;                  ///< Base solution
	MLSolution     *solResult;                  ///< Resulting solution
	MLSolution     *solAux;                     ///< Aux solution

	MLMove64       *mergeBuffer;                ///< Independent movements buffer


	ullong          timeExec;                   ///<< Exec time
	ullong          timeSync;                   ///<< Time spent in synchronizing w/ GPU
	ullong          timeIdle;                   ///<< Time CPU was idle



	/*!
	 * Initialize environment.
	 */
	void
	envInit()
	{
	    int     error;

	    // Initializes log system
	    logInit();

	    cudaDeviceProp  prop;
	    int             count;

	    // Set thread GPU
	    cudaSetDevice(0);

	    // Detect CUDA driver and GPU devices
	    switch(cudaGetDeviceCount(&count)) {
	    case cudaSuccess:
	        for(int d;d < count;d++) {
	            if(cudaGetDeviceProperties(&prop,d) == cudaSuccess) {
	                if(prop.major < 2)
	                    WARNING("Device '%s' is not suitable to this application. Device capability %d.%d < 2.0\n",
	                            prop.name,prop.major,prop.minor);
	            }
	        }
	        break;
	    case cudaErrorNoDevice:
	        WARNING("No GPU Devices detected.");
	        break;
	    case cudaErrorInsufficientDriver:
	        WARNING("No CUDA driver installed.");
	        break;
	    default:
	        EXCEPTION("Unknown error detecting GPU devices.");
	    }
	}



	void generalInit(int id = 0, int rngSeed = -1)
	{
	    l4printf("GPU%u\n",id);

	    if(rngSeed != -1)
	        rng.seed(rngSeed);
	    else
	        rng.seed(uint(sysTimer()) + id + 1);

	    gpuId = id;
	    gpuMemFree = 0;

	    memset(&gpuProps,0,sizeof(gpuProps));

	    solDevice = new MLSolution(problem,cudaHostAllocDefault);
	    solResult = new MLSolution(problem,cudaHostAllocDefault);
	    solAux    = new MLSolution(problem,cudaHostAllocDefault);


	    lsCall = 0;

	    kernelCount = 0;
	    memset(kernels,0,sizeof(kernels));

	    timeExec = 0;
	    timeIdle = 0;
	    timeSync = 0;

	    // Move merge buffer
	    mergeBuffer = new MLMove64[problem.size];
	}

	void initDevice()
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

	    // Initialize kernel
	    l4printf("GPU%u, initializing %d kernels\n",gpuId,kernelCount);
	    for(int i=0;i < kernelCount;i++) {
	        l4printf("GPU%u, initializing kernel %s\n",gpuId,kernels[i]->name);
	        kernels[i]->init(flag);
	    }
	}

	void kernelInit()
	{
	    l4printf("GPU%u: kernelAdd, count=%u\n",gpuId,kernelCount);

	    for(int i=0;i < MLP_MAX_NEIGHBOR;i++) {
	        if(kernels[i])
	            delete kernels[i];
	        kernels[i] = NULL;
	    }

	    kernelCount = 0;
	    for(int kid=0;kid < MLP_MAX_NEIGHBOR;kid++) {
	            switch(kid) {
	            case MLMI_SWAP:
	                kernels[kernelCount++] = new MLKernelSwap(problem);
	                break;
	            case MLMI_2OPT:
	                kernels[kernelCount++] = new MLKernel2Opt(problem);
	                break;
	            case MLMI_OROPT1:
	                kernels[kernelCount++] = new MLKernelOrOpt(problem,1);
	                break;
	            case MLMI_OROPT2:
	                kernels[kernelCount++] = new MLKernelOrOpt(problem,2);
	                break;
	            case MLMI_OROPT3:
	                kernels[kernelCount++] = new MLKernelOrOpt(problem,3);
	                break;
	            default:
	                EXCEPTION("Invalid move id: %d",kid);
	            }
	   	}
	}




	void runExperiment()
	{

	    MLKernel    *kernel;
	    MLSolution  *solVnd;
	    MLMove       move;
	    uint         i,max;
	    ullong       time,timeAvg;
	    int          movesCount,
	                 movesCost;
	    char         buffer[128];

	    MLSolution* solDevice = new MLSolution(problem);

	    lprintf("RAND_SEED\t: %u\n",rng.getSeed());

	    timeAvg = 0;
	    for(uint m=0;m < 10; m++) {

	        l4printf("***\n* Solution #%u\n***\n",m + 1);

	        solDevice->random(rng,0.50);

	        //solDevice->adsUpdate();

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

	    }

	    delete solVnd;
	}



};




#endif

