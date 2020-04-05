
#include <stdio.h>
#include <sys/time.h>

#include "mlkernel.h"
#include "mlkswap.h"

void runGpuGpu(MLKernel *kernel, int &impr2, int &countImpr2, int &imprMoves2, std::vector<MLMove> *moves = NULL)
{
	//std::cout << "runGPUGPU" << std::endl;
	int movesCount, movesCost;
	MLMove move;

	// MERGE GPU-GPU (partial)
	////lprintf("kernel 2 moveElems=%d!\n",kernel->moveElems);
	//			std::clock_t start = std::clock();
	// partial GPU-GPU
	MLMove64 *h_moves = kernel->transBuffer.p_move64;

	kernel->mergeGPU();
	//kernel->sync();
	kernel->recvResult();
	kernel->sync();

	//	            lprintf("MERGE_GPU moves: ");
	//	            for(unsigned i=0; i<kernel->moveElems;i++)
	//	            	lprintf("%d\t",h_moves[i].cost);
	//	            lprintf("\n");

	imprMoves2 = 0;
	impr2 = 0;
	countImpr2 = 0;
	for (unsigned iter = 0; iter < kernel->moveElems; ++iter)
	{
		move64ToMove(move, h_moves[iter]);
		if (move.cost < 0)
		{
			impr2 += move.cost;
			countImpr2++;
			//std::cout << "GOOD!" << move.cost << std::endl;
			////l4printf("Apply %s(%d,%d) = %d\n", kernel->name, move.i, move.j, move.cost);
			//                        printf("Apply %s(%d,%d) = %d\n", kernel->name, move.i, move.j, move.cost);
			kernel->applyMove(move);
			if (moves)
			{
				moves->push_back(MLMove(move));
				//MLMove temp;
				//temp.id = move.id;
				//temp.i = move.i;
				//temp.j = move.j;
				//temp.cost = move.cost;
				//moves->push_back(temp);
			}
		}
	}
	imprMoves2 = impr2;

	//                kernel->getSolution(solDevice);
	//                lprintf("solution updated!\n");
	//				solDevice->show(std::cout);

	//				duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
	////printf("partial GPU-GPU time %.7f ms\n", duration);
	////printf("partial GPU-GPU improvement=%d count=%d moveCount=%d\n", impr2, countImpr2, movesCount);

	//printf("CHECK (impr=%d impr2=%d) (count=%d count2=%d) (imprMoves=%d imprMoves2=%d)\n", impr, impr2, countImpr, countImpr2, imprMoves, imprMoves2);
}

TEST(WamcaExp, WamcaExperiment_InitKernel)
{
	MLProblem &problem = TestBerlin52::getProblem();
	EXPECT_EQ(problem.size, 53);

	// --------
	MTRandom rng;
	rng.seed(0);

	MLSolution *solDevice = new MLSolution(problem, cudaHostAllocDefault);
	solDevice->random(rng, 0.50);
	solDevice->update();
	solDevice->ldsUpdate();
	uint valor1 = solDevice->costCalc();
	EXPECT_EQ(valor1, 494309);
	// ---------
	MLKernel *kernel = new MLKernelSwap(problem);
	kernel->init(true);

	// ============== runKernel
	std::vector<MLMove> *moves = NULL;
	//void runKernel(int k, std::vector<MLMove> *moves = NULL) {

	double duration; // clock duration
	std::clock_t start;
	int movesCount, movesCost;
	int impr, countImpr;   // total cost improvement
	int impr2, countImpr2; // for checking purposes
	uint valor = 0;
	MLMove move;
	int imprMoves = 0;
	int imprMoves2 = 0;

	

	//MLKernel *kernel = kernels[k];
	kernel->setSolution(solDevice);
	kernel->sendSolution();
	valor1 = solDevice->costCalc();
	EXPECT_EQ(valor1, 494309);
	//
	struct timeval tv1;
	gettimeofday(&tv1, NULL);
	long curtime1 = tv1.tv_sec * 1000000 + tv1.tv_usec;
	//
	kernel->defineKernelGrid(); // TODO: precisa??

	gpuMemset(kernel->moveData, 0, kernel->moveDataSize);
	gpuDeviceSynchronize();
	start = std::clock();
	//
	kernel->launchKernel();
	kernel->sync();
	duration = ((std::clock() - start) / (double)CLOCKS_PER_SEC) * 1000;

	//
	runGpuGpu(kernel, impr2, countImpr2, imprMoves2, moves);


	kernel->getSolution(solDevice);
	solDevice->update();
	solDevice->ldsUpdate();
	//
	valor1 = solDevice->costCalc();
	EXPECT_EQ(valor1, 367403);
}
