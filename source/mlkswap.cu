/**
 * @file   mlkswap.cu
 *
 * @brief  MLP swap search in GPU.
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
#include "mlkswap.h"


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

#ifdef  MLP_GPU_ADS
//#if 0

/*
 * GPU Auxiliary Data Structures (ADS) code
 */
__global__
void
kernelSwap(const MLADSData *gm_ads, MLMovePack *gm_move, int size)
{
    /*!
     * Shared memory variables
     */
    extern
    __shared__
    int     sm_buffer[];        // Dynamic shared memory buffer

    /*!
     * Shared variables
     */
    __shared__
    int    *sm_coordx,          // Clients x-coordinates
           *sm_coordy,          // Clients y-coordinates
           *sm_move,            // Thread movement id/cost
           *sm_time,            // ADS time
           *sm_cost,            // ADS cost
            sm_rsize,           // ADS row size
            sm_scost,           // Solution cost
            sm_tour,            // Tour/path
            sm_cost0;           // ADS cost element
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
     *                   etime   ecost
     */

    // Only thread 0 initializes shared variables
    if(tx == 0) {
        sm_rsize = gm_ads->s.rowElems;
        sm_scost = gm_ads->s.solCost;
        sm_tour  = gm_ads->s.tour;
        sm_round = gm_ads->s.round * 0.5F;

        //sm_coordx = sm_buffer + 2; -- was like this in 2016-05-08
        sm_coordx = sm_buffer;
        sm_coordy = sm_coordx + size;
        sm_move   = sm_coordy + size;

        sm_time = sm_move;
        sm_cost = sm_move + size;

//        if(by == 0) {
//            printf("size\t\t%d\n",size);
//            printf("sm_buffer\t%p\n",sm_buffer);
//            printf("sm_coordx\t%p\n",sm_coordx);
//            printf("sm_coordy\t%p\n",sm_coordy);
//            printf("sm_move\t\t%p\n",sm_move);
//            printf("sm_time\t\t%p\n",sm_time);
//            printf("sm_cost\t\t%p\n",sm_cost);
//        }

        // Row 0 from ADS: needs only column i - 1
        gm_data = ADS_COST_PTR(gm_ads,sm_rsize);
        sm_cost0 = gm_data[by];
    }
    __syncthreads();

    // Number of chunks
    cmax  = GPU_DIVCEIL(size,blockDim.x);

    /*
     * Copy clients coordinates
     */
    gm_data = ADS_COORD_PTR(gm_ads);
    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < size);c++) {
        // Split coordinates
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

    // Row i + 1 from ADS
    n = (by + 2) * sm_rsize;
    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < size);c++)
        sm_cost[ctx] = gm_data[n + ctx];             // C[i+1]

    // Row 0 from ADS: needs only column i - 1
    if(tx == 0)
        sm_cost0 = gm_data[by];                      // C[0,i-1]

    // Wait all threads synchronize
    __syncthreads();

    bmove = 0;
    bcost = COST_INFTY;

/*
    if(tx == 0 && by == 2) {
        for(i=0;i < size;i++)
            kprintf("%d\t(%d,%d)\n",i,sm_coordx[i],sm_coordy[i]);
        kprintf("\n");

        for(i=1;i < size;i++)
            kprintf("%d-%d: %d\n",i-1,i,sm_cdist[i]);
        kprintf("\n");

        kprintf("T\t: ");
        for(i=0;i < size;i++)
            kprintf(" %d",sm_time[i]);
        kprintf("\n");

        kprintf("C[%d]\t: ",by + 2);
        for(i=0;i < size;i++)
            kprintf(" %d",sm_cost[i]);
        kprintf("\n");

        kprintf("C[0,%d]\t: %d\n",by,sm_cost0);
        kprintf("C[%d,%d]\t: %d\n",by+1,by+1,sm_cost1);
    }
*/

    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < size);c++) {

        // Movement indexes
        i = by + 1;
        j = ctx;

        cost = COST_INFTY;

        if((j > i + 1) && (j < size - sm_tour)) {
            // Last solution index
            n = size - 1;

            /*
             * [0,i-1] + [j,j]
             */
            dist = GPU_DIST_COORD(i - 1,j);     // D[i-1,j]

            // wait = 1;                        // W[j,j]   = j - j + (j > 0)
                                                //          = 1
            cost = sm_cost0 +                   // C[0,i-1]
                   0 +                          // C[j,j] = 0
                   1 * (                        // W[j,j] = 1
                   sm_time[i - 1] +             // T[0,i-1]
                   dist );                      // D[i-1,j]

            time = sm_time[i - 1] +             // T[0,i-1]
                   0 +                          // T[j,j] = 0
                   dist;                        // D[i-1,j]

            /*
             * [0,i-1] + [j,j] + [i+1,j-1]
             */
            dist = GPU_DIST_COORD(j,i + 1);     // D[j,i+1]

            wait  = j - i - 1;                  // W[i+1,j-1] = j - 1 - i - 1 + (i+1 > 0)
                                                //            = j - i - 2 + 1 = j - i - 1
            cost += sm_cost[j - 1] +            // C[i+1,j-1]
                    wait * (                    // W[i+1,j-1]
                    time +                      // T([0,i-1] + [j,j])
                    dist );                     // D[j,i+1]

            time += GPU_ADS_TIME(i + 1,j - 1) + // T[i+1,j-1]
                    dist;                       // D[j,i+1]

            /*
             * [0,i-1] + [j,j] + [i+1,j-1] + [i,i]
             */
            dist = GPU_DIST_COORD(j - 1,i);     // D[j-1,i]

            // wait  = 1;                       // W[i,i] = i - i + (i > 0)
                                                //        = 0 + 1 = 1
            cost += 0 +                         // C[i,i] = 0
                    1 * (                       // W[i,i] = 1
                    time +                      // T([0,i-1] + [j,j] + [i+1,j-1])
                    dist );                     // D[j-1,i]

            time += 0 +                         // T[i,i] = 0
                    dist;                       // D[j-1,i]

            /*
             * [0,i-1] + [j,j] + [i+1,j-1] + [i,i] + [j+1,n]
             */
            if(j + 1 <= n) {
                // Row j + 1 from ADS: needs only column n
                gm_data = ADS_COST_PTR(gm_ads,sm_rsize) +
                       sm_rsize*(j + 1);

                dist = GPU_DIST_COORD(i,j + 1);    // D[i,j+1]

                wait  = n - j;                     // W[j+1,n] = n - j - 1 + (j+1 > 0)
                                                   //          = n - j - 1 + 1 = n - j
                cost += gm_data[n] +               // C[j+1,n]
                        wait * (                   // W[j+1,n]
                        time +                     // T([0,i-1] + [j,j] + [i+1,j-1] + [i,i])
                        dist );                    // D[i,j+1]
            }

            cost = cost - sm_scost;

            k4printf("GPU_SWAP(%d,%d) = %d\n",i,j,cost);
        }

        if(cost < bcost) {
            bcost = cost;
            bmove = GPU_MOVE_PACKID(i,j,MLMI_SWAP);
        }

        /*
        if((tx==283) && (by == 171))
            kprintf("WWWW Block %d tx:%d bdim:%d ID %d: GPU_SWAP(%u,%u) = %d bcost=%d bmove=%d\n",
            		by, tx, blockDim.x, by*blockDim.x+tx,
                            i,
                            j,
                            cost, bcost, bmove);
        */
    }

//#undef  sm_ecost0

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
        k4printf("Block %d: GPU_SWAP(%u,%u) = %d\n",
                        by,
                        gm_move[by].s.i,
                        gm_move[by].s.j,
                        gm_move[by].s.cost);
    }
}


__global__
void
kernelSwapTotal(const MLADSData *gm_ads, MLMovePack *gm_move, int size)
{
    /*!
     * Shared memory variables
     */
    extern
    __shared__
    int     sm_buffer[];        // Dynamic shared memory buffer

    /*!
     * Shared variables
     */
    __shared__
    int    *sm_coordx,          // Clients x-coordinates
           *sm_coordy,          // Clients y-coordinates
           *sm_move,            // Thread movement id/cost
           *sm_time,            // ADS time
           *sm_cost,            // ADS cost
            sm_rsize,           // ADS row size
            sm_scost,           // Solution cost
            sm_tour,            // Tour/path
            sm_cost0;           // ADS cost element
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


    // Only thread 0 initializes shared variables
    if(tx == 0) {
        sm_rsize = gm_ads->s.rowElems;
        sm_scost = gm_ads->s.solCost;
        sm_tour  = gm_ads->s.tour;
        sm_round = gm_ads->s.round * 0.5F;

        //sm_coordx = sm_buffer + 2; -- was like this in 2016-05-08
        sm_coordx = sm_buffer;
        sm_coordy = sm_coordx + size;
        sm_move   = sm_coordy + size;

        sm_time = sm_move;
        sm_cost = sm_move + size;

        // Row 0 from ADS: needs only column i - 1
        gm_data = ADS_COST_PTR(gm_ads,sm_rsize);
        sm_cost0 = gm_data[by];
    }
    __syncthreads();

    // Number of chunks
    cmax  = GPU_DIVCEIL(size,blockDim.x);

    /*
     * Copy clients coordinates
     */
    gm_data = ADS_COORD_PTR(gm_ads);
    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < size);c++) {
        // Split coordinates
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

    // Row i + 1 from ADS
    n = (by + 2) * sm_rsize;
    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < size);c++)
        sm_cost[ctx] = gm_data[n + ctx];             // C[i+1]

    // Row 0 from ADS: needs only column i - 1
    if(tx == 0)
        sm_cost0 = gm_data[by];                      // C[0,i-1]

    // Wait all threads synchronize
    __syncthreads();

    bmove = 0;
    bcost = COST_INFTY;


    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < size);c++) {

        // Movement indexes
        i = by + 1;
        j = ctx;

        cost = COST_INFTY;

        if((j > i + 1) && (j < size - sm_tour)) {
            // Last solution index
            n = size - 1;

            /*
             * [0,i-1] + [j,j]
             */
            dist = GPU_DIST_COORD(i - 1,j);     // D[i-1,j]

            // wait = 1;                        // W[j,j]   = j - j + (j > 0)
                                                //          = 1
            cost = sm_cost0 +                   // C[0,i-1]
                   0 +                          // C[j,j] = 0
                   1 * (                        // W[j,j] = 1
                   sm_time[i - 1] +             // T[0,i-1]
                   dist );                      // D[i-1,j]

            time = sm_time[i - 1] +             // T[0,i-1]
                   0 +                          // T[j,j] = 0
                   dist;                        // D[i-1,j]

            /*
             * [0,i-1] + [j,j] + [i+1,j-1]
             */
            dist = GPU_DIST_COORD(j,i + 1);     // D[j,i+1]

            wait  = j - i - 1;                  // W[i+1,j-1] = j - 1 - i - 1 + (i+1 > 0)
                                                //            = j - i - 2 + 1 = j - i - 1
            cost += sm_cost[j - 1] +            // C[i+1,j-1]
                    wait * (                    // W[i+1,j-1]
                    time +                      // T([0,i-1] + [j,j])
                    dist );                     // D[j,i+1]

            time += GPU_ADS_TIME(i + 1,j - 1) + // T[i+1,j-1]
                    dist;                       // D[j,i+1]

            /*
             * [0,i-1] + [j,j] + [i+1,j-1] + [i,i]
             */
            dist = GPU_DIST_COORD(j - 1,i);     // D[j-1,i]

            // wait  = 1;                       // W[i,i] = i - i + (i > 0)
                                                //        = 0 + 1 = 1
            cost += 0 +                         // C[i,i] = 0
                    1 * (                       // W[i,i] = 1
                    time +                      // T([0,i-1] + [j,j] + [i+1,j-1])
                    dist );                     // D[j-1,i]

            time += 0 +                         // T[i,i] = 0
                    dist;                       // D[j-1,i]

            /*
             * [0,i-1] + [j,j] + [i+1,j-1] + [i,i] + [j+1,n]
             */
            if(j + 1 <= n) {
                // Row j + 1 from ADS: needs only column n
                gm_data = ADS_COST_PTR(gm_ads,sm_rsize) +
                       sm_rsize*(j + 1);

                dist = GPU_DIST_COORD(i,j + 1);    // D[i,j+1]

                wait  = n - j;                     // W[j+1,n] = n - j - 1 + (j+1 > 0)
                                                   //          = n - j - 1 + 1 = n - j
                cost += gm_data[n] +               // C[j+1,n]
                        wait * (                   // W[j+1,n]
                        time +                     // T([0,i-1] + [j,j] + [i+1,j-1] + [i,i])
                        dist );                    // D[i,j+1]
            }

            cost = cost - sm_scost;

            k4printf("GPU_SWAP(%d,%d) = %d\n",i,j,cost);
        } // if (j > i + 1)

        if(cost < bcost) {
            bcost = cost;
            bmove = GPU_MOVE_PACKID(i,j,MLMI_SWAP);
        }

    } // for c=0

//#undef  sm_ecost0

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
kernelSwap(const MLADSData *gm_ads, MLMovePack *gm_move, int size)
{
    /*!
     * Shared memory variables
     */
    extern  __shared__
    int     sm_buffer[];            // Dynamic shared memory buffer
    /*!
     * Shared variables
     */
    __shared__
    int    *sm_soldist,             // Solution distance (adjacent clients)
           *sm_coordx,              // Clients x-coordinates
           *sm_coordy,              // Clients y-coordinates
           *sm_move,                // Thread movement id/cost
            sm_rsize,               // ADS row size
            sm_tour;                // Tour/path
    /*!
     * Local variables
     */
    uint   *gm_data;                // Points to some data
    int     dcost;                  // Cost improvement
    int     c,                      // Chunk no
            ctx,                    // Tx index for chunk
            cmax,                   // Number of chunks
            csize;                  // Chunk size
    int     i,j,k;                  // Auxiliary

    if(tx >= size)
        return;
    /*
     * Dynamic shared memory buffer usage
     *
     * buffer
     * |
     * v
     * +---------+--------+--------+-----------------+
     * | soldist | coordx | coordy | movid | movcost |
     * +---------+--------+--------+-----------------+
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

    cmax  = GPU_DIVCEIL(size,blockDim.x);
    csize = GPU_MIN(size,int(blockDim.x));

    // Get coordinates
    gm_data = ADS_COORD_PTR(gm_ads);
    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < size);c++) {
        // Split coordinates
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

    for(c=0;c < cmax;c++) {

        ctx = c*blockDim.x + tx;

        i = j = 0;
        dcost = COST_INFTY;

        if(ctx < size - sm_tour - 3) {
            // Compute movement indexes performed by thread
            k = (ctx >= by);
            i = k*(by  + 1) + (!k)*(size - sm_tour - by  - 2);
            j = k*(ctx + 3) + (!k)*(size - sm_tour - ctx - 1);

            dcost  = (size - i)     * ( int(GPU_DIST_COORD(i - 1,j)) - sm_soldist[i] );
            dcost += (size - i - 1) * ( int(GPU_DIST_COORD(j,i + 1)) - sm_soldist[i + 1] );
            dcost += (size - j)     * ( int(GPU_DIST_COORD(j - 1,i)) - sm_soldist[j] );

            // When computing PATHs and j = size - 1, there's no j+1 element,
            // so threre's no related cost
            if(j + 1 < size)
                dcost += (size - j - 1) * ( int(GPU_DIST_COORD(i,j + 1)) - sm_soldist[j+1] );

            k4printf("GPU_SWAP(%d,%d) = %d\n",i,j,dcost);
        }

        if(dcost < sm_move[tx]) {
            sm_move[tx] = dcost;
            sm_move[tx + csize] = GPU_MOVE_PACKID(i,j,MLMI_SWAP);
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
        k4printf("Block %d: GPU_SWAP(%u,%u) = %d\n",
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
                    sm_move[tx + csize] = sm_move[tx + i + csize];
                }
            }
        }
        __syncthreads();
    }

    if(tx == 0) {
        gm_move[by].w = GPU_MOVE_PACK64(sm_move[0],sm_move[csize]);
        k4printf("Block %d: MIN SWAP(%u,%u) = %d\n",by,
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
MLKernelSwap::defineKernelGrid()
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
    grid.x  = 1;
    grid.y  = solSize - problem.costTour - 3;
    grid.z  = 1;

#else
    /*
     *  Compute dynamic shared memory size
     */
    shared = 5 * GPU_BLOCK_CEIL(int,solSize);
    /*
     * Compute grid
     */
    grid.x = 1;
    grid.y = ((solSize + 1) / 2) - 1;
    grid.z = 1;

#endif

    if(gpuOccupancyMaxPotentialBlockSizeVariableSMem(&gsize,&bsize,kernelSwap,
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

    /*
    l4printf("Kernel %s\tgrid(%d,%d,%d)\tblck(%d,%d,%d)\tshared=%u (%u KB)\n",
                    name,
                    grid.x,grid.y,grid.z,
                    block.x,block.y,block.z,
                    shared,shared / 1024);
     */
}

void
MLKernelSwap::launchKernel()
{
    ////lprintf("Calling kernel %s\n",name);

    // Update kernel calls counter
    callCount++;
    // Kernel in execution
    flagExec = true;

    /*
    lprintf("Kernel %s\tgrid(%d,%d,%d)\tblck(%d,%d,%d)\tshared=%u (%u KB)\tsize=%u\n",
                    name,
                    grid.x,grid.y,grid.z,
                    block.x,block.y,block.z,
                    shared,shared / 1024,
                    solSize);
    lprintf("adsData=%p\n",adsData);
    */

    // Calls kernel
//    if(!isTotal)
	//std::cout << "MLKernelSwap::launchKernel - grid(x: " << grid.x << ", y: " << grid.y << ", z: " << grid.z
    	//	<< ") block(x: " << block.x << ", y: " << block.y << ", z: " << block.z << ") shared: " << shared << std::endl;
    	kernelSwap<<<grid,block,shared,stream>>>(adsData,moveData,solSize);
//    else
//    	kernelSwapTotal<<<grid,block,shared,stream>>>(adsData,moveData,solSize);
    gpuDeviceSynchronize();
}

void
MLKernelSwap::applyMove(MLMove &move)
{
    int    i,j;
    ushort t;

    i = move.i;
    j = move.j;

    //solution->showCostCalc("BASE    : ");

    solution->cost += move.cost;
    solution->time  = sysTimer();
    solution->move  = move;

    t = solution->clients[i];
    solution->clients[i] = solution->clients[j];
    solution->clients[j] = t;

    solution->weights[i] = solution->dist(i - 1,i);
    solution->weights[i + 1] = solution->dist(i,i + 1);

    solution->weights[j] = solution->dist(j - 1,j);
    if(j + 1 < solution->clientCount)
        solution->weights[j + 1] = solution->dist(j,j + 1);

    // DEFINIR -DMLP_COST_CHECK

#ifdef MLP_COST_CHECK
    if(problem.params.checkCost) {
        uint ccost = solution->costCalc();
        l4printf("CHECK COST: %s,\tcost=%u, check=%u\n",name,solution->cost,ccost);
        if(solution->cost != ccost) {
            lprintf("%s(%u,%u): wrong=%u, right=%u\n",name ,move.i ,move.j ,solution->cost ,ccost );
            solution->showCostCalc("SOLUTION: ");
            EXCEPTION("INVALID COST: %s",problem.name);
        }
    }
#endif
}
