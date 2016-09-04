/**
 * @file	mlkernel.cpp
 *
 * @brief   Handle a kernel.
 *
 * @author	Eyder Rios
 * @date    2015-05-28
 */

#include <algorithm>
#include <fstream>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include "log.h"
//#include "mlgputask.h"
#include "mlkernel.h"
#include "mlsolution.h"


using namespace std;

// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //

#define MINMAX(a,b,min,max)     if(a < b) { min = a; max = b; } \
                                else      { min = b; max = a; }

#define tx      threadIdx.x



#define def_canSwapMerge(m1,m2) (  ( GPU_ABS(int(m1.i - m2.i)) > 1 ) && \
							       ( GPU_ABS(int(m1.i - m2.j)) > 1 ) && \
							       ( GPU_ABS(int(m1.j - m2.i)) > 1 ) && \
							       ( GPU_ABS(int(m1.j - m2.j)) > 1 ) )

__forceinline__ __device__ bool f_canSwapMerge(MLMove64& m1, MLMove64& m2) {
	return     ( GPU_ABS(int(m1.i - m2.i)) > 1 ) &&
		       ( GPU_ABS(int(m1.i - m2.j)) > 1 ) &&
		       ( GPU_ABS(int(m1.j - m2.i)) > 1 ) &&
		       ( GPU_ABS(int(m1.j - m2.j)) > 1 );
}

__forceinline__ __device__ bool f_can2OptMerge(MLMove64& m1, MLMove64& m2) {
    return (m1.j < m2.i - 1) ||
           (m1.i > m2.j + 1) ||
           (m2.j < m1.i - 1) ||
           (m2.i > m1.j + 1);
}

__forceinline__ __device__ bool f_canOrOptMerge(MLMove64& m1, MLMove64& m2) {
    register int k1,min1,max1, k2,min2,max2;

    MINMAX(m1.i,m1.j,min1,max1);
    k1 = MLMI_OROPT_K(m1.id);
    MINMAX(m2.i,m2.j,min2,max2);
    k2 = MLMI_OROPT_K(m2.id);

    return (max1 + k1 < min2) || (min1 > max2 + k2);
}

__global__ void testSwap(MLMove64* g_move64, int numElems)
{
	if(tx >= numElems)
		return;
	extern __shared__ MLMove64 s_move64[];

	register MLMove64 m = g_move64[tx];
	//if(m >= 0){
	//	m.id = 1000; // giant value
	//	return;
	//}
	s_move64[tx] = m;
	if(s_move64[tx].cost >= 0) // optional
		return;                // optional
	syncthreads();

	for(int i=0; i<=tx; ++i) { // tx 'or' numElems (?) symmetry?
		if(i != tx) {
			register MLMove64 mi = s_move64[i];
			if(mi.cost < 0) {
				register MLMove64 mx = s_move64[tx];
				if(!f_canSwapMerge(mi, mx))
					s_move64[tx].cost = 0;
			}
		}
		syncthreads();
	}

	if(s_move64[tx].cost == 0)
		g_move64[tx] = s_move64[tx];

	// end kernel
}


__global__ void test2Opt(MLMove64* g_move64, int numElems)
{
	if(tx >= numElems)
		return;
	extern __shared__ MLMove64 s_move64[];

	register MLMove64 m = g_move64[tx];
	//if(m >= 0){
	//	m.id = 1000; // giant value
	//	return;
	//}
	s_move64[tx] = m;
	if(s_move64[tx].cost >= 0) // optional
		return;                // optional
	syncthreads();

	for(int i=0; i<=tx; ++i) { // tx 'or' numElems (?) symmetry?
		if(i != tx) {
			register MLMove64 mi = s_move64[i];
			if(mi.cost < 0) {
				register MLMove64 mx = s_move64[tx];
				if(!f_can2OptMerge(mi, mx))
					s_move64[tx].cost = 0;
			}
		}
		syncthreads();
	}

	if(s_move64[tx].cost == 0)
		g_move64[tx] = s_move64[tx];

	// end kernel
}


__global__ void testOrOpt(MLMove64* g_move64, int numElems)
{
	if(tx >= numElems)
		return;
	extern __shared__ MLMove64 s_move64[];

	register MLMove64 m = g_move64[tx];
	//if(m >= 0){
	//	m.id = 1000; // giant value
	//	return;
	//}
	s_move64[tx] = m;
	if(s_move64[tx].cost >= 0) // optional
		return;                // optional
	syncthreads();

	for(int i=0; i<=tx; ++i) { // tx 'or' numElems (?) symmetry?
		if(i != tx) {
			register MLMove64 mi = s_move64[i];
			if(mi.cost < 0) {
				register MLMove64 mx = s_move64[tx];
				if(!f_canOrOptMerge(mi, mx))
					s_move64[tx].cost = 0;
			}
		}
		syncthreads();
	}

	if(s_move64[tx].cost == 0)
		g_move64[tx] = s_move64[tx];

	// end kernel
}


void
MLKernel::mergeGPU() {
   	   thrust::device_ptr<MLMovePack> td_moveData(moveData);
   	   thrust::sort(td_moveData, td_moveData+moveElems, OBICmp());

/*
       gpuMemcpyAsync(transBuffer.p_void,moveData,moveElems * sizeof(MLMove64),cudaMemcpyDeviceToHost,stream);
       MLMove64* p_pack = transBuffer.p_move64;
       sync();

       printf("FIRST PRINT!\n");
       int count = 0;
       for(unsigned i=0; i<moveElems; i++) {
    	   printf("%d\t",p_pack[i].cost);
    	   if(p_pack[i].cost < 0)
    		   count++;
       }
       printf("\n");
       printf("count negative=%d moveElems=%d\n", count, moveElems);
*/

       dim3 grid, block;
       grid.x = grid.y = grid.z = 1;
       block.x = moveElems;
       block.y = block.z = 1;

       //printf("kernel_id=%d\n",this->id);

       /*
    MLMI_SWAP,
    MLMI_2OPT,
    MLMI_OROPT1,
    MLMI_OROPT2,
    MLMI_OROPT3,
        */
		int minGridSize;
		int blockSize;
		size_t sMemSize = moveElems * sizeof(MLMove64);
		int blockSizeLimit = moveElems;

	switch (this->id)
	{
	case MLMI_SWAP:
		gpuOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, testSwap, __cudaOccupancyB2DHelper(sMemSize), blockSizeLimit);
		//printf("minGridSize=%d blockSize=%d moveElems=%d\n",minGridSize, blockSize, moveElems);

		//block.x = blockSize;
		block.x = ::max(blockSize, moveElems);

		testSwap<<<grid, block, sMemSize, stream>>>((MLMove64*) moveData, moveElems);
		break;
	case MLMI_2OPT:
		gpuOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, test2Opt, __cudaOccupancyB2DHelper(sMemSize), blockSizeLimit);
		//printf("minGridSize=%d blockSize=%d moveElems=%d\n",minGridSize, blockSize, moveElems);

		//block.x = blockSize;
		block.x = ::max(blockSize, moveElems);

		test2Opt<<<grid, block, sMemSize, stream>>>((MLMove64*) moveData, moveElems);
		break;
	case MLMI_OROPT1:
	case MLMI_OROPT2:
	case MLMI_OROPT3:
		gpuOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, testOrOpt, __cudaOccupancyB2DHelper(sMemSize), blockSizeLimit);
		//printf("minGridSize=%d blockSize=%d moveElems=%d\n",minGridSize, blockSize, moveElems);

		//block.x = blockSize;
		block.x = ::max(blockSize, moveElems);

		testOrOpt<<<grid, block, sMemSize, stream>>>((MLMove64*) moveData, moveElems);
		break;
	default:
		printf("NO KERNEL TO CALL.. YET!\n");
		getchar();
	}
	// TODO: djdiusdisjd

       //cudaOccupancyMaxPotentialBlockSizeVariableSMem(  // TODO:USE!
}



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
// ##                              CLASS MLKernelTask                            ## //
// ##                                                                            ## //
// ################################################################################ //

MLKernel::MLKernel(MLProblem& _problem, bool _isTotal, int kid, uint ktag) :
                problem(_problem), isTotal(_isTotal)
{
    id   = kid;
    tag  = ktag;
    name = nameMove[kid];

    lprintf("MLKernel id=%d TOTAL=%d\n", kid, isTotal);

    callCount = 0;
    mergeCount = 0;
    imprvCount = 0;
    timeMove = 0;

    reset(); // adsData = NULL, etc...
}

MLKernel::~MLKernel()
{
    if(stream)
        term();
}


void
MLKernel::reset()
{
    memset(&grid,0,sizeof(grid));
    memset(&block,0,sizeof(block));

    shared = 0;

    time = 0;

    stream = 0;
    evtStart = 0;
    evtStop = 0;

    solSize = problem.size;

    adsData = NULL;
    adsDataSize = 0;
    adsRowElems = 0;

    moveData = NULL;
    moveDataSize = 0;
    moveElems = 0;

    maxMerge = 0;

    transBuffer.p_void = NULL;
    transBufferSize = 0;

    solution = NULL;
    solDestroy = true;

    flagOptima = 0;
}

void
MLKernel::init(bool solCreate)
{
    lprintf("Kernel INIT: %s TOTAL:%d\n",name, isTotal);

	size_t  free,
            size;

    int gpuId = 0; // TODO: fix


    // Kernel solution
    //solBase = new MLSolution(problem,cudaHostAllocDefault);
    //solBase->evalAlloc();

    // Kernel best solution
    solDestroy = solCreate;
    if(solCreate) {
        solution = new MLSolution(problem,cudaHostAllocDefault);
        solution->adsAlloc();
    }

    /*
     * Kernel data
     *
     * +--------+--------+--------+--------+--------+
     * |  SWAP  |  2OPT  | OROPT1 | OROPT2 | OROPT3 |
     * +--------+--------+--------+--------+--------+
     */

    // Kernel stream
    gpuStreamCreate(&stream);

//#ifdef GPU_PROFILE
//    // Set stream name for NVIDIA Profiler
//    nvtxNameCuStreamA(stream,name);
//#endif

    // Kernel events
    gpuEventCreate(&evtStart);
    gpuEventCreate(&evtStop);

    /*
     * ADS buffer
     *
     *     info      coords[x,y]        [W|Id]              T               C[0]                C[size-1]
     * +---------+----------------+----------------+----------------+----------------+-----+----------------+
     * | b elems | size elems |gap| size elems |gap| size elems |gap| size elems |gap| ... | size elems |gap|
     * +---------+----------------+----------------+----------------+----------------+-----+----------------+
     *
     * b = MLP_ADSINFO_ELEMS = GPU_GLOBAL_ALIGN/sizeof(uint) = 128/4 = 32 elems
     *
     * info[0]   : size (ADS buffer size in bytes)
     * info[1]   : rowElems
     * info[2]   : solElems (sol size)
     * info[3]   : solCost
     * info[4]   : tour
     * info[5]   : k
     * info[6]   : reserved
     * ...
     * info[b-1] : reserved
     */
#ifdef GPU_EVAL_ALIGNED
    adsRowElems  = GPU_BLKSIZE_GLOBAL(solSize * sizeof(uint)) / sizeof(uint);
#else
    adsRowElems  = solSize;
#endif

    //if(problem.params.gpuAds)
        adsDataSize = ADS_INFO_SIZE + (solSize + 3) * adsRowElems * sizeof(uint);
    //else
    //    adsDataSize = ADS_INFO_SIZE +            3  * adsRowElems * sizeof(uint);

    gpuMalloc(&adsData,adsDataSize);

    /*
     * Movement buffer
     *
     *        0                       size - 1
     * +-------------+------------+-------------+
     * |  (m,i,j,c)  |    . . .   |  (m,i,j,c)  |
     * +-------------+------------+-------------+
     *
     *  m   neighborhood (swap,2opt,oropt1,oropt2,oropt3)
     *  i   solution index i
     *  j   solution index j
     *  c   solution cost
     */
    if(!isTotal)
    	moveDataSize = solSize * sizeof(ullong);
    else
    	moveDataSize = solSize * solSize * sizeof(ullong);
    transBufferSize = moveDataSize;

    gpuMalloc(&moveData,moveDataSize);
    gpuHostMalloc(&transBuffer.p_void,transBufferSize,0);//???gpuTask.params.allocFlags); TODO: fix


    // Define kernel grid
    defineKernelGrid();

    // Move merge graph
    if(!isTotal)
    	graphMerge.resize(solSize);
    else
    	graphMerge.resize(solSize); // TODO: make solSize*solSize

    lprintf("transBuffer size%d\n", moveDataSize);
    lprintf("transBuffer Pointer %p\n",transBuffer);
    //getchar();

#if LOG_LEVEL > 4
    lprintf("<<< kernel %s\n",name);
    lprintf("grid(%d,%d,%d)\tblck(%d,%d,%d)\tshared=%u (%u KB)\n",
                    grid.x,grid.y,grid.z,
                    block.x,block.y,block.z,
                    shared,shared / 1024);
    lprintf("GPU Id\t\t: %u\n", gpuId);
    lprintf("Kernel Id\t: %u\n",id);
    lprintf("Kernel Name\t: %s\n",name);
    lprintf("solSize\t\t: %6u elems\n",solSize);
    lprintf("adsRowElems\t: %6u elems\t%6lu bytes\n",adsRowElems,adsRowElems * sizeof(uint));
    lprintf("adsData\t\t: %6lu elems\t%6u bytes\t%p\n",adsDataSize / sizeof(uint),adsDataSize,adsData);
    lprintf("adsSolution\t: %6u elems\t%6lu bytes\t%p\n",adsRowElems,adsRowElems * sizeof(uint),ADS_SOLUTION_PTR(adsData,adsRowElems));
    lprintf("adsTime\t\t: %6u elems\t%6lu bytes\t%p\n",adsRowElems,adsRowElems * sizeof(uint),ADS_TIME_PTR(adsData,adsRowElems));
    lprintf("adsCost\t\t: %6u elems\t%6lu bytes\t%p\n",adsRowElems * solSize,adsRowElems * sizeof(uint) * solSize,ADS_COST_PTR(adsData,adsRowElems));
    //lprintf("moveBuffer\t: %6u elems\t%6lu bytes\t%p\n",solSize,solSize * sizeof(ullong),moveBuffer);
    lprintf(">>>> kernel %s\n",name);
#endif
}

void
MLKernel::term()
{
	int gpuId = 0; // TODO: fix
    l4printf("GPU%u Kernel %d: %s\n",gpuId,id,name);

    gpuStreamDestroy(stream);
    gpuEventDestroy(evtStart);
    gpuEventDestroy(evtStop);

    gpuFree(adsData);
    gpuFree(moveData);

    gpuHostFree(transBuffer.p_void,0);//????gpuTask.params.allocFlags); TODO: fix

    if(solDestroy && solution)
        delete solution;

    reset();
}

int
MLKernel::bestMove(MLMove &move)
{
    const
    MLMove64   *moves;
    ullong      time;
    uint        i,b;

    time = sysTimer();

    moves = transBuffer.p_move64;

    l4printf("baseCost  = %u\n",solution->cost);
    l4printf("moveElems = %u\n",moveElems);

//    for(i=0;i < moveElems;i++)
//        printf("%s(%u,%u)=%-6d\ti=%u\n",name,moves[i].i,moves[i].j,moves[i].cost,i);

    b = 0;
    for(i=1;i < moveElems;i++) {
        l4printf("MOVE %u: %s(%u,%u)=%d < move(%u,%u)=%d\n",i,name,
                        uint(moves[i].i),uint(moves[i].j),moves[i].cost,
                        uint(moves[b].i),uint(moves[b].j),moves[b].cost);
        if(moves[i].cost < moves[b].cost)
            b = i;
    }
    move.id = MLMoveId(moves[b].id);
    move.i  = moves[b].i;
    move.j  = moves[b].j;
    move.cost = moves[b].cost;

    timeMove += sysTimer() - time;

    return move.cost;
}

inline
bool
canSwapMerge(MLMove64 *m1, MLMove64 *m2) // TODO: SWAP MERGE!
{
    switch(m2->id) {
    case MLMI_SWAP:
//    	return true;

        return ( abs(int(m1->i - m2->i)) > 1 ) &&
               ( abs(int(m1->i - m2->j)) > 1 ) &&
               ( abs(int(m1->j - m2->i)) > 1 ) &&
               ( abs(int(m1->j - m2->j)) > 1 );

    case MLMI_2OPT:
        return ( (m1->i < m2->i - 1) || (m1->i > m2->j + 1) ) &&
               ( (m1->j < m2->i - 1) || (m1->j > m2->j + 1) );
    default:
        int k2,min2,max2;

        MINMAX(m2->i,m2->j,min2,max2);
        k2 = MLMI_OROPT_K(m2->id);
        return (m1->j < min2 - 1)  ||
               (m1->i > max2 + k2) ||
               ( (m1->i < min2 - 1) && (m1->j > max2 + k2) );

    }
    //return false;
}

inline
bool
can2OptMerge(MLMove64 *m1, MLMove64 *m2)
{
    switch(m2->id) {
    case MLMI_SWAP:
        return canSwapMerge(m2,m1);
    case MLMI_2OPT:
        return (m1->j < m2->i - 1) ||
               (m1->i > m2->j + 1) ||
               (m2->j < m1->i - 1) ||
               (m2->i > m1->j + 1);
    default:
        int k2,min2,max2;

        MINMAX(m2->i,m2->j,min2,max2);
        k2 = MLMI_OROPT_K(m2->id);
        return (m1->i > max2 + k2) || (m1->j < min2 - 1);
    }
    //return false;
}

inline
bool
canOrOptMerge(MLMove64 *m1, MLMove64 *m2)
{
    switch(m2->id) {
    case MLMI_SWAP:
        return canSwapMerge(m2,m1);
        break;
    case MLMI_2OPT:
        return can2OptMerge(m2,m1);
        break;
    default:
        int k1,min1,max1,
            k2,min2,max2;

        MINMAX(m1->i,m1->j,min1,max1);
        k1 = MLMI_OROPT_K(m1->id);
        MINMAX(m2->i,m2->j,min2,max2);
        k2 = MLMI_OROPT_K(m2->id);

//        if(m1->i == 4 && m1->j == 1 && m2->i == 6 && m2->j == 7) {
//            lprintf("(%d,%d) x (%d,%d)\n",m1->i,m1->j,m2->i,m2->j);
//            lprintf("\t[ max(%d,%d) + %d < min(%d,%d) ] || ",m1->i,m1->j,k1,m2->i,m2->j);
//            lprintf("\t[ min(%d,%d) > max(%d,%d) + %d ] =\n",m1->i,m1->j,m2->i,m2->j,k2);
//            lprintf("\t[ %d < %d ] || [ %d > %d ] = %d\n",max1 + k1,min2,min1,max2 + k2,
//                            ((max1 + k1 < min2) || (min1 > max2 + k2)));
//        }

        return (max1 + k1 < min2) || (min1 > max2 + k2);
    }
    return false;
}

bool
canMerge(MLMove64 *move1, MLMove64 *move2)
{
    switch(move1->id) {
    case MLMI_SWAP:
        return canSwapMerge(move1,move2);
    case MLMI_2OPT:
        return can2OptMerge(move1,move2);
    default:
        return canOrOptMerge(move1,move2);
    }
    return false;
}

bool
compMove(MLMove64 m1, MLMove64 m2)
{
    return m1.cost < m2.cost;
}

int
MLKernel::mergeGreedy(MLMove64 *merge, int &count)
{
    MLMove64   *moves;
    ullong      time;
    int         i,j,n;
    int         cost;

    time = sysTimer();

    moves = transBuffer.p_move64;

    count = 0;
    cost  = 0;

//#define MLP_MERGE_SPLIT
#ifdef  MLP_MERGE_SPLIT
    n = moveElems;
    i = 0;
    j = n - 1;
    while(i < j) {
        while((i <  n) && (moves[i].cost < 0))
            i++;
        while((j >= 0) && (moves[j].cost >= 0))
            j--;
        if(i < j)
            swap(moves[i],moves[j]);
    }
    n = i;
#else
    n = moveElems;
#endif

    lprintf("moveElems=%d\n",moveElems);

	/*

    if(isTotal) {
    	i = 109723;
    	lprintf("%d (%d,%d) %d\n", i, moves[i].i, moves[i].j, moves[i].cost);
    				getchar();
		for (i = 0; i < n; i++)
		{
			lprintf("%d (%d,%d) %d\n", i, moves[i].i, moves[i].j, moves[i].cost);
			getchar();
		}
    }
		*/


    // Sort moves
    sort(moves,moves + n,compMove);


	i = 0;
	lprintf("after sort %d (%d,%d) %d\n", i, moves[i].i, moves[i].j, moves[i].cost);

    /*
    // If first moves has no gain, or ONLY first move has gain, returns
    if((moves[0].cost >= 0) || ((moves[0].cost < 0) && (moves[1].cost >= 0))) {
        merge[0] = moves[0];
        count = moves[0].cost < 0;
        l4printf("Graph  %d/%d %s moves (return)\n",count,n,name);
        return  moves[0].cost;
    }
    */

    // Assign moves to graph
    graphMerge.clear();

    int ncount = 0;
    for(i=0;i < n;i++) {
        if(moves[i].cost >= 0)
            break;
        ncount++;
    }

    n = ncount;
    // TODO: Grafo não suporta nós suficientes!! > 500000
    // TODO: fazer então mesma versão limitada da GPU!
    n = ::min(1024,n);

    graphMerge.resize(n);

    // end assign moves

    for(i=0;i < n;i++) {
#ifndef MLP_MERGE_SPLIT
        if(moves[i].cost >= 0)
            break;
#endif
        graphMerge.addVertex(moves + i,0);
        //printf("id: %d  graphMerge.cost=%d\n",i, graphMerge[i]->cost);
    }
    lprintf("Graph  %d/%d %s moves\n",graphMerge.vertexCount,n,name);

    // Set conflicts (edges)
    for(i=0;i < graphMerge.vertexCount;i++) {
//    	if(this->id==0)
//    		printf("id: %d (%d,%d)  graphMerge.cost=%d\n",i, graphMerge[i]->i, graphMerge[i]->j, graphMerge[i]->cost);
        for(j=i;j < graphMerge.vertexCount;j++) {
            if(!canMerge(graphMerge[i],graphMerge[j])){
//            	lprintf("set_edge(%d,%d)\n",i,j);
            	graphMerge.setEdge(i,j);
            }
        }
    }

    // PRINT FLAGS
    /*
    lprintf("Print FLAGS:\n");
    for(i=0;i < graphMerge.vertexCount;i++) {
   	    lprintf("PFLAG id=%d cost=%d flag=%d\n",i, graphMerge[i]->cost, graphMerge.getFlag(i));
    }
    lprintf("\n");
    */

#if 0
    /*
     * Generate movement conflict graph (Graphviz DOT file)
     */
    ofstream   os("/tmp/merge.dot",ios::trunc);

    os << "graph G {\n\tnode [shape=circle,fontsize=10];\n";
    for(int i=0;i < graphMerge.count;i++) {
        os << '\t' << i << " [label=\"" << i << "\\n" << graphMerge[i]->i << ',' << graphMerge[i]->j << "\"];\n";
    }
    for(int i=0;i < graphMerge.vertexCount;i++) {
        for(int j=i + 1;j < graphMerge.vertexCount;j++) {
            if(graphMerge.hasEdge(i,j))
                os << '\t' << i << " -- " << j << ";\n";
        }
    }
    os << "}\n";
    graphMerge.show();
#endif

    // Independent moves greedy algorithm
    for(i=0;i < graphMerge.vertexCount;i++) {
        if(graphMerge.getFlag(i))
            continue;

        merge[count] = *graphMerge[i];
        cost += graphMerge[i]->cost;
        lprintf("choose %s(%d,%d) = %d\n", this->name, graphMerge[i]->i, graphMerge[i]->j, graphMerge[i]->cost);

        ++count;
        /*
        if(count == maxMerge) {
            lprintf("maxMerge reached: %d\n",count);
            getchar();
            getchar();
            break;
        }
        */

        graphMerge.setFlag(i,1);

        for(j=i + 1;j < graphMerge.vertexCount;j++) {
            if(graphMerge.hasEdge(i,j))
                graphMerge.setFlag(j,1);

        }
    }

    timeMove += sysTimer() - time;

    return cost;
}


/*
void
MLKernel::sendSolution()
{
   uint *sol;

   sol = ADS_SOLUTION_PTR(solution->adsData,adsRowElems);
   lprintf("GPU%u: %s.sendSolution()\tadsDataSize=%u\tsol=%p\n",gpuTask.gpuId,name,adsDataSize,sol);
   printf("\tsendSolution\t<");
   for(int i=0;i < solSize;i++)
       printf(" %u",GPU_HI_USHORT(sol[i]));
   printf(" > p=%p\n",sol);

    // Copy solution to GPU
   gpuMemcpyAsync(adsData,solution->adsData,adsDataSize,cudaMemcpyHostToDevice,stream);
}
*/
