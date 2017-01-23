/**
 * @file   mlk2opt.cu
 *
 * @brief  MLP 2opt search in GPU.
 *
 * @author Eyder Rios
 * @date   2014-06-01
 */

#include <iostream>
#include <stdint.h>
#include <stdarg.h>
#include "log.h"
#include "utils.h"
#include "gpu.h"
//#include "mlgputask.h"
#include "mlk2opt.h"


// ################################################################################# //
// ##                                                                             ## //
// ##                              CONSTANTS & MACROS                             ## //
// ##                                                                             ## //
// ################################################################################# //

#define tx      threadIdx.x
#define by      blockIdx.y

// ################################################################################# //
// ##                                                                             ## //
// ##                               KERNEL FUNCTIONS                              ## //
// ##                                                                             ## //
// ################################################################################# //

//#define MLP_GPU_ADS
#ifdef  MLP_GPU_ADS

/*
 * GPU Auxiliary Data Structures (ADS) code
 */
__global__
void
kernel2Opt(const MLADSData *gm_ads, MLMovePack *gm_move, const int size)
{
    /*!
     * Shared memory variables
     */
    extern
    __shared__
    int     sm_buffer[];        // Dynamic shared memory buffer

    __shared__
    int    *sm_coordx,          // Clients x-coordinates
           *sm_coordy,          // Clients y-coordinates
           *sm_move,            // Thread movement id/cost
           *sm_time,            // ADS time
           *sm_cost,            // ADS cost
            sm_rsize,           // ADS row size
            sm_scost,           // Solution cost
            sm_tour;            // Tour/path
    __shared__
    float   sm_round;           // Round value
    /*!
     * Local memory variables
     */
    uint   *gm_data;            // Points to some ADS data
    int     dist,               // Distance
            cost,               // Solution cost
            time,               // Travel time
            wait,               // Wait
            bcost,              // Best cost in chunk
            bmove;              // Best move in chunk
    int     c,                  // Chunk no
            ctx,                // Tx index for chunk
            cmax;               // Number of chunks
    int     i,j,                // Movement indexes
            n;                  // Last solution index

    if(tx >= size)
        return;

    /*
     * Dynamic shared memory buffer usage
     *
     * buffer
     * |
     * v
     * +--------+--------+-----------------+
     * | coordx | coordy | movid | movcost |
     * +--------+--------+-----------------+
     *                   ^       ^
     *                   |       |
     *                   time    cost
     */

    // Only thread 0 initializes shared variables
    if(tx == 0) {
        sm_coordx = sm_buffer;
        sm_coordy = sm_coordx + size;
        sm_move   = sm_coordy + size;

        sm_time = sm_move;
        sm_cost = sm_move + size;

        sm_rsize = gm_ads->s.rowElems;
        sm_scost = gm_ads->s.solCost;
        sm_tour  = gm_ads->s.tour;
        sm_round = gm_ads->s.round * 0.5F;
    }
    __syncthreads();

    // Number of chunks
    cmax  = GPU_DIVCEIL(size,blockDim.x);

    /*
     * Copy clients coordinates
     */
    gm_data = ADS_COORD_PTR(gm_ads);
    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < size);c++) {
        sm_coordx[ctx] = GPU_HI_USHORT(gm_data[ctx]);
        sm_coordy[ctx] = GPU_LO_USHORT(gm_data[ctx]);
    }
    /*!
     * Copy ADS.T
     */
    // Points to ADS.T
    gm_data = ADS_TIME_PTR(gm_ads,sm_rsize);
    // Copy ADS.T data
    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < size);c++)
        sm_time[ctx] = gm_data[ctx];
    /*!
     * Copy ADS.C
     */
    // Points to ADS.C
    gm_data = ADS_COST_PTR(gm_ads,sm_rsize);
    // Row C[0] from ADS
    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < size);c++)
        sm_cost[ctx] = gm_data[ctx];         // C[0]

    // Wait all threads synchronize
    __syncthreads();

/*
    if(tx == 0 && by == 0) {
        for(i=0;i < size;i++)
            kprintf("%d\t(%d,%d)\n",i,sm_coordx[i],sm_coordy[i]);
        kprintf("\n");

        kprintf("T\t: ");
        for(i=0;i < size;i++)
            kprintf(" %d",sm_time[i]);
        kprintf("\n");

        kprintf("C[%d]\t: ",0);
        for(i=0;i < size;i++)
            kprintf(" %d",sm_cost[i]);
        kprintf("\n");
    }
*/
    // Best move/cost of chunk
    bmove = 0;
    bcost = COST_INFTY;

     __syncthreads();

     for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < size);c++) {

         if(ctx < size - sm_tour - 2) {

             n = (ctx >= by);

             // Movement indexes
             i = n*(ctx - by  + 1) + (!n)*(by - ctx);
             j = n*(ctx + 2)       + (!n)*(size + !sm_tour - ctx - 2);

             // Last solution index
             n  = size - 1;

             // Row C[j] from ADS
             gm_data = ADS_COST_PTR(gm_ads,sm_rsize) +
                       sm_rsize*j;

             /*
              * [0,i-1] + [j,i]
              */
             dist = GPU_DIST_COORD(i - 1,j);        // D[i-1,j]

             wait  = j - i + 1;                     // W[j,i] = j - i + (j > 0)
                                                    //        = j - i + 1
             cost  = sm_cost[i - 1] +               // C[0,i-1]
                     gm_data[i] +                   // C[j,i]
                     wait * (                       // W[j,i]
                     sm_time[i - 1] +               // T[0,i-1]
                     dist );                        // D[i-1,j] = D[j,i-1]

             time  = sm_time[i - 1] +               // T[0,i-1]
                     GPU_ADS_TIME(i,j) +            // T[j,i] = T[i,j]
                     dist;                          // D[i-1,j] = D[j,i-1]

             /*
              * [0,i-1] + [j,i] + [j+1,n]
              */
             if(j + 1 <= n) {

                 // Line 'j + 1' from eval matrix
                 gm_data = ADS_COST_PTR(gm_ads,sm_rsize) +
                           sm_rsize*(j + 1);

                 dist = GPU_DIST_COORD(i,j + 1);    // D[i,j+1]

                 wait  = n - j;                     // W[j+1,n] = n - j - 1 + (j+1 > 0)
                                                    //          = n - j - 1 + 1 = n - j
                 cost += gm_data[n] +               // C[j+1,n]
                         wait * (                   // W[j+1,n]
                         time +                     // T([0,i-1] + [j,i])
                         dist );                    // D[i,j+1] = D[j+1,i]
             }

             cost = cost - sm_scost;

             if(cost < bcost) {
                 bcost = cost;
                 bmove = GPU_MOVE_PACKID(i,j,MLMI_2OPT);
             }

             k4printf("GPU_2OPT(%d,%d) = %d\n",i,j,cost);
         }
     }

     __syncthreads();

     // Chunk size
     n = GPU_MIN(size,int(blockDim.x));

     sm_move[tx] = bcost;
     sm_move[tx + n] = bmove;

     __syncthreads();

     /*
      * Minimum cost reduction
      */
     for(i=GPU_DIVCEIL(n,2);i > 1;i=GPU_DIVCEIL(i,2)) {
         if(tx < i) {
             if((tx + i < n) && (sm_move[tx] > sm_move[tx + i])) {
                 sm_move[tx] = sm_move[tx + i];
                 sm_move[tx + n] = sm_move[tx + n + i];
             }
         }
         __syncthreads();
     }

     if(tx == 0) {
         // The first 2 elements was not compared
         if(sm_move[0] > sm_move[1]) {
             sm_move[0] = sm_move[1];
             sm_move[n] = sm_move[n + 1];
         }

         gm_move[by].w = GPU_MOVE_PACK64(sm_move[0],sm_move[n]);
         k4printf("Block %d: GPU_2OPT(%u,%u) = %d\n",
                         by,
                         gm_move[by].s.i,
                         gm_move[by].s.j,
                         gm_move[by].s.cost);
     }
}


__global__
void
kernel2OptTotal(const MLADSData *gm_ads, MLMovePack *gm_move, const int size)
{
    /*!
     * Shared memory variables
     */
    extern
    __shared__
    int     sm_buffer[];        // Dynamic shared memory buffer

    __shared__
    int    *sm_coordx,          // Clients x-coordinates
           *sm_coordy,          // Clients y-coordinates
           *sm_move,            // Thread movement id/cost
           *sm_time,            // ADS time
           *sm_cost,            // ADS cost
            sm_rsize,           // ADS row size
            sm_scost,           // Solution cost
            sm_tour;            // Tour/path
    __shared__
    float   sm_round;           // Round value
    /*!
     * Local memory variables
     */
    uint   *gm_data;            // Points to some ADS data
    int     dist,               // Distance
            cost,               // Solution cost
            time,               // Travel time
            wait,               // Wait
            bcost,              // Best cost in chunk
            bmove;              // Best move in chunk
    int     c,                  // Chunk no
            ctx,                // Tx index for chunk
            cmax;               // Number of chunks
    int     i,j,                // Movement indexes
            n;                  // Last solution index

    if(tx >= size)
        return;

    /*
     * Dynamic shared memory buffer usage
     *
     * buffer
     * |
     * v
     * +--------+--------+-----------------+
     * | coordx | coordy | movid | movcost |
     * +--------+--------+-----------------+
     *                   ^       ^
     *                   |       |
     *                   time    cost
     */

    // Only thread 0 initializes shared variables
    if(tx == 0) {
        sm_coordx = sm_buffer;
        sm_coordy = sm_coordx + size;
        sm_move   = sm_coordy + size;

        sm_time = sm_move;
        sm_cost = sm_move + size;

        sm_rsize = gm_ads->s.rowElems;
        sm_scost = gm_ads->s.solCost;
        sm_tour  = gm_ads->s.tour;
        sm_round = gm_ads->s.round * 0.5F;
    }
    __syncthreads();

    // Number of chunks
    cmax  = GPU_DIVCEIL(size,blockDim.x);

    /*
     * Copy clients coordinates
     */
    gm_data = ADS_COORD_PTR(gm_ads);
    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < size);c++) {
        sm_coordx[ctx] = GPU_HI_USHORT(gm_data[ctx]);
        sm_coordy[ctx] = GPU_LO_USHORT(gm_data[ctx]);
    }
    /*!
     * Copy ADS.T
     */
    // Points to ADS.T
    gm_data = ADS_TIME_PTR(gm_ads,sm_rsize);
    // Copy ADS.T data
    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < size);c++)
        sm_time[ctx] = gm_data[ctx];
    /*!
     * Copy ADS.C
     */
    // Points to ADS.C
    gm_data = ADS_COST_PTR(gm_ads,sm_rsize);
    // Row C[0] from ADS
    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < size);c++)
        sm_cost[ctx] = gm_data[ctx];         // C[0]

    // Wait all threads synchronize
    __syncthreads();


    // Best move/cost of chunk
    bmove = 0;
    bcost = COST_INFTY;

     __syncthreads();

     for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < size);c++) {

         if(ctx < size - sm_tour - 2) {

             n = (ctx >= by);

             // Movement indexes
             i = n*(ctx - by  + 1) + (!n)*(by - ctx);
             j = n*(ctx + 2)       + (!n)*(size + !sm_tour - ctx - 2);

             // Last solution index
             n  = size - 1;

             // Row C[j] from ADS
             gm_data = ADS_COST_PTR(gm_ads,sm_rsize) +
                       sm_rsize*j;

             /*
              * [0,i-1] + [j,i]
              */
             dist = GPU_DIST_COORD(i - 1,j);        // D[i-1,j]

             wait  = j - i + 1;                     // W[j,i] = j - i + (j > 0)
                                                    //        = j - i + 1
             cost  = sm_cost[i - 1] +               // C[0,i-1]
                     gm_data[i] +                   // C[j,i]
                     wait * (                       // W[j,i]
                     sm_time[i - 1] +               // T[0,i-1]
                     dist );                        // D[i-1,j] = D[j,i-1]

             time  = sm_time[i - 1] +               // T[0,i-1]
                     GPU_ADS_TIME(i,j) +            // T[j,i] = T[i,j]
                     dist;                          // D[i-1,j] = D[j,i-1]

             /*
              * [0,i-1] + [j,i] + [j+1,n]
              */
             if(j + 1 <= n) {

                 // Line 'j + 1' from eval matrix
                 gm_data = ADS_COST_PTR(gm_ads,sm_rsize) +
                           sm_rsize*(j + 1);

                 dist = GPU_DIST_COORD(i,j + 1);    // D[i,j+1]

                 wait  = n - j;                     // W[j+1,n] = n - j - 1 + (j+1 > 0)
                                                    //          = n - j - 1 + 1 = n - j
                 cost += gm_data[n] +               // C[j+1,n]
                         wait * (                   // W[j+1,n]
                         time +                     // T([0,i-1] + [j,i])
                         dist );                    // D[i,j+1] = D[j+1,i]
             }

             cost = cost - sm_scost;

             if(cost < bcost) {
                 bcost = cost;
                 bmove = GPU_MOVE_PACKID(i,j,MLMI_2OPT);
             }

             k4printf("GPU_2OPT(%d,%d) = %d\n",i,j,cost);
         }
     }

     __syncthreads();

     gm_move[by*blockDim.x+tx].w = GPU_MOVE_PACK64(bcost,bmove);

}





#else

/*
 * GPU On Demand Calculation (ODC) code
 *
 */
__global__
void
kernel2Opt(const MLADSData *gm_ads, MLMovePack *gm_move, const int size)
{
    /*!
     * Shared memory variables
     */
    extern  __shared__
    int     sm_buffer[];            // Dynamic shared memory buffer
    /*!
     * Local variables
     */
    __shared__
    int    *sm_soldist,             // Client distances
           *sm_coordx,              // Clients x-coordinates
           *sm_coordy,              // Clients y-coordinates
           *sm_move,                // Thread movement cost
            sm_rsize,               // ADS row size
            sm_tour;                // Tour/path

    uint   *gm_data;                // Points to some ADS data
    int     c,                      // Chunk no
            ctx,                    // Tx index for chunk
            cmax,                   // Number of chunks
            csize;                  // Chunk size
    int     cost,                   // Cost improvement
            i,j,                    // Movement indexes
            k;                      // Last solution index

    if(tx >= size)
        return;

    /*
     * Dynamic shared memory buffer usage
     *
     * buffer
     * |
     * v
     * +---------+--------+--------+------------------+
     * | soldist | coordx | coordy | movcost | moveid |
     * +---------+--------+--------+------------------+
     */

    /*
     * Set pointers to proper location in 'sm_buffer'
     * Only thread 0 initializes shared variables
     */
    if(tx == 0) {
        sm_soldist = sm_buffer;
        sm_coordx  = sm_soldist + size;
        sm_coordy  = sm_coordx  + size;
        sm_move    = sm_coordy  + size;

        sm_rsize = gm_ads->s.rowElems;
        sm_tour  = gm_ads->s.tour;
    }
    __syncthreads();

    // Number of chunks
    cmax  = GPU_DIVCEIL(size,blockDim.x);
    // Chunk size
    csize = GPU_MIN(size,int(blockDim.x));

    // Get clients coordinates
    gm_data = ADS_COORD_PTR(gm_ads);
    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < size);c++) {
        sm_coordx[ctx] = GPU_HI_USHORT(gm_data[ctx]);
        sm_coordy[ctx] = GPU_LO_USHORT(gm_data[ctx]);
    }

    // Get solution distances
    gm_data = ADS_SOLUTION_PTR(gm_ads,sm_rsize);
    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < size);c++)
        sm_soldist[ctx] = GPU_LO_USHORT(gm_data[ctx]);

    // Initialize movement/cost
    sm_move[tx] = COST_INFTY;

    // Wait all threads arrange data
    __syncthreads();

/*
    if(tx == 0 && by == 0) {
        kprintf("COORDS\n");
        for(i=0;i < size;i++)
            kprintf("%d\t(%d,%d)\n",i,sm_coordx[i],sm_coordy[i]);
        kprintf("\n");

        kprintf("DIST\n");
        for(i=0;i < size;i++)
            kprintf(" %d",sm_soldist[i]);
        kprintf("\n");
    }
*/

    // Get solution distances
    for(c=0;c < cmax;c++) {

        ctx = c*blockDim.x + tx;

        if(ctx < size - sm_tour - 2) {
            // Movement indexes
            k = ctx >= by;
            i = k * (ctx - by + 1) + (!k) * (by   - ctx);
            j = k * (ctx + 2)      + (!k) * (size - ctx - sm_tour - 1);

            cost = (size - i) * ( int(GPU_DIST_COORD(i - 1,j)) - sm_soldist[i] );
            // When computing PATHs and j = size-1, there's no j+1 element
            if(j + 1 < size)
                cost += (size - j - 1) * ( int(GPU_DIST_COORD(i,j + 1)) - sm_soldist[j + 1] );

            for(k=1;k <= j - i;k++)
                cost += (size - i - k) * ( int(sm_soldist[j - k + 1]) - sm_soldist[i + k] );

            k4printf("GPU_2OPT(%d,%d) = %d\n",i,j,cost);

            if(cost < sm_move[tx]) {
                sm_move[tx] = cost;
                sm_move[tx + csize] = GPU_MOVE_PACKID(i,j,MLMI_2OPT);
            }
        }

    }

    __syncthreads();

    /*
     * Minimum cost reduction
     */
    for(i=GPU_DIVCEIL(csize,2);i > 1;i=GPU_DIVCEIL(i,2)) {
        if(tx < i) {
            if((tx + i < csize) && (sm_move[tx] > sm_move[tx + i])) {
                sm_move[tx] = sm_move[tx + i];
                sm_move[tx + csize] = sm_move[tx + csize + i];
            }
        }
        __syncthreads();
    }

    if(tx == 0) {
        // The first 2 elements was not compared
        if(sm_move[0] > sm_move[1]) {
            sm_move[0] = sm_move[1];
            sm_move[csize] = sm_move[csize + 1];
        }

        gm_move[by].w = GPU_MOVE_PACK64(sm_move[0],sm_move[csize]);
        k4printf("Block %d: GPU_2OPT(%u,%u) = %d\n",
                        by,
                        gm_move[by].s.i,
                        gm_move[by].s.j,
                        gm_move[by].s.cost);
    }

/*
    // MIN cost reduction
    for(i=GPU_DIVCEIL(csize,2);i > 0;i >>= 1) {
        if(tx < i) {
            if(tx + i < csize) {
                if(sm_move[tx] > sm_move[tx + i]) {
                    sm_move[tx] = sm_move[tx + i];
                    sm_move[csize + tx] = sm_move[csize + tx + i];
                }
            }
        }
        __syncthreads();
    }

    if(tx == 0) {
        gm_move[by].w = GPU_MOVE_PACK64(sm_move[0],sm_move[csize]);
        k4printf("Block %d: MIN 2OPT(%u,%u) = %d\n",by,
                        gm_move[by].s.i,
                        gm_move[by].s.j,
                        gm_move[by].s.cost);
    }
*/
}

#endif


// ################################################################################# //
// ##                                                                             ## //
// ##                               KERNEL LAUNCHERS                              ## //
// ##                                                                             ## //
// ################################################################################# //

void
MLKernel2Opt::defineKernelGrid()
{
    int gsize,
        bsize;

#ifdef MLP_GPU_ADS
    /*
     *  Compute dynamic shared memory size
     */
    shared = 4 * GPU_BLOCK_CEIL(int,solSize);
    /*
     * Compute grid
     */
    grid.x = 1;
    grid.y = GPU_DIVCEIL(solSize,2);
    grid.z = 1;

#else
    /*
     *  Compute dynamic shared memory size
     */
    shared = 5 * GPU_BLOCK_CEIL(int,solSize);
    /*
     * Compute grid
     */
    grid.x = 1;
    grid.y = GPU_DIVCEIL(solSize,2);
    grid.z = 1;

#endif

    if(gpuOccupancyMaxPotentialBlockSizeVariableSMem(&gsize,&bsize,kernel2Opt,
                        MLKernelSharedSize(this),solSize) == cudaSuccess)
        block.x = bsize;
    //else
    //    block.x = problem.params.blockSize ? problem.params.blockSize : solSize;
    block.y = 1;
    block.z = 1;
    /*
     * Number of movements returned
     */
    moveElems = grid.y;

    if(isTotal)
    	moveElems = grid.y*block.x;

    //if(problem.params.maxMerge)
    //    maxMerge = problem.params.maxMerge;
    //else
        maxMerge = moveElems;

    l4printf("Kernel %s\tgrid(%d,%d,%d)\tblck(%d,%d,%d)\tshared=%u (%u KB)\n",
                    name,
                    grid.x,grid.y,grid.z,
                    block.x,block.y,block.z,
                    shared,shared / 1024);
}

void
MLKernel2Opt::launchKernel()
{
    l4printf("Calling kernel %s\n",name);

    // Update kernel calls counter
    callCount++;
    // Kernel in execution
    flagExec = true;

    // Calls kernel

//    if(!isTotal)
    	kernel2Opt<<<grid,block,shared,stream>>>(adsData,moveData,solSize);
//    else
//    	kernel2OptTotal<<<grid,block,shared,stream>>>(adsData,moveData,solSize);
}

void
MLKernel2Opt::applyMove(MLMove &move)
{
    ushort  t;
    int     i,j;

    //solution->showCostCalc("BASE    : ");

    solution->cost += move.cost;
    solution->time  = sysTimer();
    solution->move  = move;

    i = move.i;
    j = move.j;

    if(i < j) {
        while(i < j) {
            t = solution->clients[i];
            solution->clients[i] = solution->clients[j];
            solution->clients[j] = t;
            i++;
            j--;
        }

        j = move.j;
        if(j + 1 < solution->clientCount)
            j++;

        for(i=move.i;i <= j;i++)
            solution->weights[i] = problem.clients[ solution->clients[i - 1] ].weight[ solution->clients[i] ];
    }

#ifdef MLP_COST_CHECK
    if(problem.params.checkCost) {
        uint ccost = solution->costCalc();
        l4printf("CHECK COST: %s,\tcost=%u, check=%u\n",name,solution->cost,ccost);
        if(solution->cost != ccost) {
            lprintf("%s(%u,%u): wrong=%u, right=%u\n",name,move.i,move.j,solution->cost,ccost);
            solution->showCostCalc("SOLUTION: ");
            EXCEPTION("INVALID COST: %s",problem.name);
        }
    }
#endif
}
