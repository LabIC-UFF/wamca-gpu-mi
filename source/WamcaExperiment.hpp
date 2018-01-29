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

#include "mlkernel.h"

class WAMCAExperiment
{
public:

    WAMCAExperiment(MLProblem& _problem, int seed) :
        problem(_problem)
    {
        generalInit(0, seed); // SEED
        kernelInit();
        initDevice();
    }

    MLProblem& problem;

    MTRandom         rng;               ///< Random number generator

    uint            gpuId;                      ///< GPU id
    uint            gpuMemFree;                 ///< Amount of free memory on GPU
    cudaDeviceProp  gpuProps;                   ///< Device properties

    MLKernel       *kernels[MLP_MAX_NEIGHBOR];  ///< Kernel searches
    uint            kernelCount;                ///< Number of kernels
    uint            kernelBits;                 ///< Bit mask for active kernels

    MLKernel       *tkernels[MLP_MAX_NEIGHBOR];  ///< Kernel searches
    uint            tkernelCount;                ///< Number of kernels
    uint            tkernelBits;                 ///< Bit mask for active kernels


    ullong          lsCall;                     ///< Local search call no.

    MLSolution     *solDevice;                  ///< Base solution
    MLSolution     *solResult;                  ///< Resulting solution
    MLSolution     *solAux;                     ///< Aux solution

    MLMove64       *mergeBuffer;                ///< Independent movements buffer
    MLMove64       *tmergeBuffer;                ///< Independent movements buffer


    ullong          timeExec;                   ///<< Exec time
    ullong          timeSync;                   ///<< Time spent in synchronizing w/ GPU
    ullong          timeIdle;                   ///<< Time CPU was idle


    void generalInit(int id = 0, int rngSeed = 0)
    {
        lprintf("GPU%u\n",id);

        if(rngSeed != 0) {
            rng.seed(rngSeed);
            lprintf("set seed %d\n", rngSeed);
        } else {
            rng.seed(uint(sysTimer()) + id + 1);
            lprintf("set random seed!\n");
        }

        gpuId = id;
        gpuMemFree = 0;

        memset(&gpuProps,0,sizeof(gpuProps));

        solDevice = new MLSolution(problem,cudaHostAllocDefault);
        solResult = new MLSolution(problem,cudaHostAllocDefault);
        solAux    = new MLSolution(problem,cudaHostAllocDefault);


        lsCall = 0;

        kernelCount = 0;
        memset(kernels,0,sizeof(kernels));

        tkernelCount = 0;
        memset(tkernels,0,sizeof(tkernels));

        timeExec = 0;
        timeIdle = 0;
        timeSync = 0;

        // Move merge buffer
        mergeBuffer = new MLMove64[problem.size];

        tmergeBuffer = new MLMove64[problem.size*problem.size];
    }


    void kernelInit()
    {
        l4printf("GPU%u: kernelAdd, count=%u\n", gpuId, kernelCount);

        for (int i = 0; i < MLP_MAX_NEIGHBOR; i++) {
            if (kernels[i]) {
                printf("ERRO!\n");
                delete kernels[i];
            }
            kernels[i] = NULL;
        }

        kernelCount = 0;
        for (int kid = 0; kid < MLP_MAX_NEIGHBOR; kid++) {
            switch (kid) {
            case MLMI_SWAP:
                kernels[kernelCount++] = new MLKernelSwap(problem);
                break;
            case MLMI_2OPT:
                kernels[kernelCount++] = new MLKernel2Opt(problem);
                break;
            case MLMI_OROPT1:
                kernels[kernelCount++] = new MLKernelOrOpt(problem, 1);
                break;
            case MLMI_OROPT2:
                kernels[kernelCount++] = new MLKernelOrOpt(problem, 2);
                break;
            case MLMI_OROPT3:
                kernels[kernelCount++] = new MLKernelOrOpt(problem, 3);
                break;
            default:
                EXCEPTION("Invalid move id: %d", kid);
            }
        }

        // TOTAL kernels
        for (int i = 0; i < MLP_MAX_NEIGHBOR; i++) {
            if (tkernels[i]) {
                printf("ERRO!\n");
                delete tkernels[i];
            }
            tkernels[i] = NULL;
        }

        tkernelCount = 0;
        for (int kid = 0; kid < MLP_MAX_NEIGHBOR; kid++) {
            switch (kid) {
            case MLMI_SWAP:
                tkernels[tkernelCount++] = new MLKernelSwap(problem, true); // true = TOTAL
                break;
            case MLMI_2OPT:
                tkernels[tkernelCount++] = new MLKernel2Opt(problem, true);
                break;
            case MLMI_OROPT1:
                tkernels[tkernelCount++] = new MLKernelOrOpt(problem, 1, true);
                break;
            case MLMI_OROPT2:
                tkernels[tkernelCount++] = new MLKernelOrOpt(problem, 2, true);
                break;
            case MLMI_OROPT3:
                tkernels[tkernelCount++] = new MLKernelOrOpt(problem, 3, true);
                break;
            default:
                EXCEPTION("Invalid move id: %d", kid);
            }
        }

    }


    void initDevice()
    {
        MLSolution *sol;
        size_t      free,
                    size;
        //bool        flag;  TODO: Pra que serve??? entender.

        l4printf("GPU%u\n",gpuId);

        // Get GPU properties
        gpuGetDeviceProperties(&gpuProps,gpuId);
        // Get free memory size
        gpuMemGetInfo(&free,&size);
        gpuMemFree = free;

        l4printf("GPU%u (%s)\n",gpuId,gpuProps.name);

        // Initialize kernel
        lprintf("GPU%u, initializing %d kernels\n", gpuId, kernelCount);
        for (int i = 0; i < kernelCount; i++)
            kernels[i]->init(true);

        lprintf("GPU%u, initializing %d tkernels\n", gpuId, tkernelCount);
        for (int i = 0; i < tkernelCount; i++)
            tkernels[i]->init(true);

    }


    unsigned int runWAMCA2016(int mMax = 3, int kMin = 0, int kMax = 3, int *solution = NULL, unsigned int solutionSize = 0)
    {
        lprintf("BEGIN WAMCA 2016 Experiments\n");

        //MLSolution  *solVnd;
        MLMove       move;
        uint         i,max;
        ullong       time,timeAvg;
        int          movesCount,
                     movesCost;
        char         buffer[128];
        int impr,  countImpr;  // total cost improvement
        int impr2, countImpr2; // for checking purposes
        std::clock_t start;
        double duration; // clock duration
        uint valor = 0;

        lprintf("RAND_SEED\t: %u\n",rng.getSeed());

        timeAvg = 0;
        for(uint m=0; m < mMax; m++) {
            lprintf("***\n* Solution #%u\n***\n",m + 1);

            MLSolution* solDevice = new MLSolution(problem);
            if (solution == NULL) {
                solDevice->random(rng,0.50);
            } else {
                solDevice->clientCount = solutionSize;
                for (int si = 0; si < solutionSize; si++) {
                    solDevice->clients[si] = solution[si];
                }
                solDevice->update();
            }
            solDevice->ldsUpdate();

            lprintf("random solution created!\n");
            solDevice->show(std::cout);


            lprintf("BEGIN PARTIAL - GPU-XPU\n");
            //for(int k=0;k < kernelCount;k++) {
            for(int k=0; k < kMax; k++) {
                MLKernel    *kernel = kernels[k];
                //lprintf("initializing kernel %d with &kernel:%p\n", k, kernel);
                lprintf("initializing kernel %d with &kernel:%p %s TOTAL=%d\n", k, kernel, kernel->name, kernel->isTotal);


                MLMove64* h_moves = kernel->transBuffer.p_move64;

                lprintf("&h_moves=%p\n", h_moves);

                kernel->setSolution(solDevice);
                //lprintf("kernel solution set!\n");
                //kernel->solution->show(std::cout);
                //kernel->solution->ldsShow(std::cout);

                kernel->sendSolution();
                //lprintf("kernel solution sent!\n");
                kernel->defineKernelGrid(); // TODO: precisa??
                //lprintf("defined kernel grid!\n");

                //kernel->launchShowDataKernel(5, 32);
                //lprintf("printed data!\n");


                lprintf("launching kernel k=%d %s!\n",k,kernel->name);
                lprintf("kernel moveElems=%d!\n",kernel->moveElems);
                gpuMemset(kernel->moveData, 0, kernel->moveDataSize);
                gpuDeviceSynchronize();
                start = std::clock();
                kernel->launchKernel();
                kernel->sync();
                duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
                printf("kernel time %.7f ms\n", duration);

                start = std::clock();
                // partial GPU-CPU
                kernel->recvResult();
                kernel->sync();

//	            lprintf("kernel GPU moves: ");
//	            for(unsigned i=0; i<kernel->moveElems;i++)
//	            	lprintf("%d\t",h_moves[i].cost);
//	            lprintf("\n");

                // MERGE GPU-CPU (partial)
/*
                movesCost = kernel->mergeGreedy(mergeBuffer,movesCount);
                impr = 0;
                countImpr = 0;
                for(unsigned iter=0; iter<movesCount; ++iter) {
                    move64ToMove(move, mergeBuffer[iter]);
                    if(move.cost < 0) {
                        impr += move.cost;
                        countImpr++;
                        l4printf("Apply %s(%d,%d) = %d\n", kernel->name, move.i, move.j, move.cost);
                        //kernel->applyMove(move);
                    }
                }
                duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
                printf("partial GPU-CPU time %.7f ms\n", duration);
                printf("partial GPU-CPU improvement=%d count=%d moveCount=%d\n", impr, countImpr, movesCount);
*/


                // MERGE GPU-GPU (partial)
                lprintf("kernel 2 moveElems=%d!\n",kernel->moveElems);
                start = std::clock();
                // partial GPU-GPU
                kernel->mergeGPU();
                //kernel->sync();
                kernel->recvResult();
                kernel->sync();


//	            lprintf("MERGE_GPU moves: ");
//	            for(unsigned i=0; i<kernel->moveElems;i++)
//	            	lprintf("%d\t",h_moves[i].cost);
//	            lprintf("\n");

                impr2 = 0;
                countImpr2 = 0;
                for (unsigned iter = 0; iter < kernel->moveElems; ++iter) {
                    move64ToMove(move, h_moves[iter]);
                    if (move.cost < 0) {
                        impr2 += move.cost;
                        countImpr2++;
                        l4printf("Apply %s(%d,%d) = %d\n", kernel->name, move.i, move.j, move.cost);
                        kernel->applyMove(move);
                    }
                }
                duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
                printf("partial GPU-GPU time %.7f ms\n", duration);
                printf("partial GPU-GPU improvement=%d count=%d moveCount=%d\n", impr2, countImpr2, movesCount);


                if((impr2==impr) && (countImpr == countImpr2)) {
                    lprintf("IMPR CHECKED OK!\n\n");
                } else {
                    lprintf("IMPR ERROR! :( \n\n");
                    getchar();
                    getchar();
                }



                l4printf("Cost improvement: %d --> %d = %d\n",
                         int(solDevice->cost),
                         int(kernel->solution->cost),
                         int(kernel->solution->cost) - int(solDevice->cost));
            }  // end for each kernel
            lprintf("END PARTIAL - GPU-XPU\n");
            lprintf("-----------------------------------------\n");



            //getchar();
            /*


            	        lprintf("BEGIN TOTAL - GPU-XPU\n");
            	        //for(int k=0;k < tkernelCount;k++) {
            	        for(int k=0;k < kMax;k++) {
            	        	MLKernel* tkernel = tkernels[k];
            	        	lprintf("initializing kernel %d with &tkernel:%p %s TOTAL=%d\n", k, tkernel, tkernel->name, tkernel->isTotal);

            	        	lprintf("&tkernel->transBuffer=%p\n", tkernel->transBuffer);

            	            MLMove64* h_moves = tkernel->transBuffer.p_move64;

            	            lprintf("&h_moves=%p\n", h_moves);

            	            tkernel->setSolution(solDevice);
            	            lprintf("tkernel solution set!\n");
            	            //kernel->solution->show(std::cout);
            	            //kernel->solution->ldsShow(std::cout);

            	            tkernel->sendSolution();
            	            //lprintf("kernel solution sent!\n");
            	            tkernel->defineKernelGrid(); // TODO: precisa??
            	            //lprintf("defined kernel grid!\n");

            	            //kernel->launchShowDataKernel(5, 32);
            	            //lprintf("printed data!\n");


            	            lprintf("launching kernel k=%d %s!\n",k,tkernel->name);
            	            lprintf("kernel moveElems=%d!\n",tkernel->moveElems);
            	            gpuMemset(tkernel->moveData, 0, tkernel->moveDataSize);
            	            gpuDeviceSynchronize();
                            start = std::clock();
            	            tkernel->launchKernel();
            	            tkernel->sync();
            	            duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
            	            printf("kernel time %.7f ms\n", duration);

            	            start = std::clock();
            	            // partial GPU-CPU
            	            tkernel->recvResult();
            	            tkernel->sync();
            	            lprintf("GOT RESULT OF %d ELEMS\n", tkernel->moveElems);

            //	            lprintf("kernel GPU moves: ");
            //	            for(unsigned i=0; i<kernel->moveElems;i++)
            //	            	lprintf("%d\t",h_moves[i].cost);
            //	            lprintf("\n");


            	            movesCost = tkernel->mergeGreedy(mergeBuffer,movesCount);
            	            impr = 0;
            	            countImpr = 0;
            	            for(unsigned iter=0; iter<movesCount; ++iter) {
            					move64ToMove(move, mergeBuffer[iter]);
            					if(move.cost < 0) {
            						impr += move.cost;
            						countImpr++;
            						l4printf("tApply %s(%d,%d) = %d\n", tkernel->name, move.i, move.j, move.cost);
            						tkernel->applyMove(move);
            					}
            				}
            	            duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
            	            printf("total GPU-CPU time %.7f ms\n", duration);
            	            printf("total GPU-CPU improvement=%d count=%d moveCount=%d\n", impr, countImpr, movesCount);


            	            lprintf("tkernel 2 moveElems=%d!\n",tkernel->moveElems);
            	            start = std::clock();
            	            // partial GPU-GPU
            	            tkernel->mergeGPU();
            	            //kernel->sync();
            	            tkernel->recvResult();
            	            tkernel->sync();



            //	            lprintf("MERGE_GPU moves: ");
            //	            for(unsigned i=0; i<kernel->moveElems;i++)
            //	            	lprintf("%d\t",h_moves[i].cost);
            //	            lprintf("\n");


            				impr2 = 0;
            				countImpr2 = 0;
            				for (unsigned iter = 0; iter < tkernel->moveElems; ++iter) {
            					move64ToMove(move, h_moves[iter]);
            					if (move.cost < 0) {
            						impr2 += move.cost;
            						countImpr2++;
            						l4printf("tApply %s(%d,%d) = %d\n", tkernel->name, move.i, move.j, move.cost);
            						tkernel->applyMove(move);
            					}
            				}
            				duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
            				printf("total GPU-GPU time %.7f ms\n", duration);
            				printf("total GPU-GPU improvement=%d count=%d moveCount=%d\n", impr2, countImpr2, movesCount);


            				if((impr2==impr) && (countImpr == countImpr2)) {
            					lprintf("IMPR CHECKED OK!\n\n");
            				} else {
            					lprintf("IMPR CHECK ERROR! :( \n\n");
            					//getchar();
            					//getchar();
            				}



            	            l4printf("Cost improvement: %d --> %d = %d\n",
            	                            int(solDevice->cost),
            	                            int(tkernel->solution->cost),
            	                            int(tkernel->solution->cost) - int(solDevice->cost));

            	            lprintf("finished this kernel\n");
            	            //getchar();
            	        }  // end for each kernel
            	        lprintf("-----------------------------------------\n");
            	        lprintf("END TOTAL\n");
            	        //getchar();

            */

            if (solution == NULL) {
                for (int si = 0; si < solutionSize; si++) {
                    solution[si] = solDevice->clients[si];
                }
            }

            valor = solDevice->costCalc();
        }

        // TODO: APLICAR MOVIMENTOS DIRETAMENTE NA GPU! SE FOREM INDEPENDENTES FICA FACIL :)
        // EXISTE VANTAGEM EM FAZER ISSO DENTRO DE UM KERNEL COM PARALELISMO DINÂMICO?
        // USAR FUNCAO DE COMPRESSAO DA CUDAPP
        // COMO CONTABILIZAR NUMERO DE MOVIMENTOS UTEIS?

        //delete solVnd;
        return valor;
    }



};




#endif
