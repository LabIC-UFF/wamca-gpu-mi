/**
 * @file   mlkoropt.cu
 *
 * @brief  MLP OrOpt-k search in GPU.
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
#include "mlkoropt.h"


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

/*
 * GPU Auxiliary Data Structures (ADS) code
 */
__global__
void
kernelOrOpt(const MLADSData *gm_ads, MLMovePack *gm_move, const int k, const int size)
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
           *sm_move,            // Moves id/cost
           *sm_time,            // ADS time
           *sm_cost,            // ADS cost
            sm_rsize,           // ADS row size
            sm_scost,           // Solution cost
            sm_tour,            // Tour/path
            sm_lsize;           // Loop size
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
     * +--------+--------+------+------+
     * | coordx | coordy | time | cost |
     * +--------+--------+------+------+
     *                   ^
     *                   |
     *                   move
     */
    // Only thread 0 initializes shared variables
    if(tx == 0) {
        sm_coordx = sm_buffer;
        sm_coordy = sm_coordx + size;
        sm_time   = sm_coordy + size;
        sm_cost   = sm_time   + size;
        sm_move   = sm_time;

        sm_rsize = gm_ads->s.rowElems;
        sm_scost = gm_ads->s.solCost;
        sm_tour  = gm_ads->s.tour;
        sm_round = gm_ads->s.round * 0.5F;

        sm_lsize = size - k + !sm_tour - 1;
    }
    __syncthreads();

    if((k < 1) || (k > 3))
        k4printf("ERROR k=%d\tsm_tour=%d\tsm_rsize=%d\n",k,sm_tour,sm_rsize);

    // Number of chunks
    cmax  = GPU_DIVCEIL(size,blockDim.x);

    // Get clients coordinates
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
    // Row C[0] from ADS: columns 0 to by
    gm_data = ADS_COST_PTR(gm_ads,sm_rsize);
    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) <= by);c++)
        sm_cost[ctx] = gm_data[ctx];         // C[0]

    // Row C[by + k + 1] from ADS: columns by + k + 1 to N-T-1
    n = size - by - k - sm_tour - 1;
    i = by + k + 1;
    gm_data = ADS_COST_PTR(gm_ads,sm_rsize) +
             sm_rsize*i;
    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) <= n);c++)
        sm_cost[i + ctx] = gm_data[i + ctx]; // C[by + k + 1]

    // Wait all threads synchronize
    __syncthreads();

    /*
    if(tx == 0 && by == 2) {
        for(i=0;i < size;i++)
            kprintf("%d\t(%d,%d)\n",i,sm_coordx[i],sm_coordy[i]);
        kprintf("\n");

        kprintf("T\t: ");
        for(i=0;i < size;i++)
            kprintf(" %d",sm_time[i]);
        kprintf("\n");

        printf("k=%d\tby=%d\n\n",k,by);

        kprintf("C[%d]\t: ",0);
        for(i=0;i <= by;i++)
            kprintf(" %d",sm_cost[i]);
        kprintf("\n");

        kprintf("C[%d]\t: ",by + k + 1);
        for(i=by + k + 1;i < size;i++)
            kprintf(" %d",sm_cost[i]);
        kprintf("\n");
    }*/

    // Best move/cost of chunk
    bmove = 0;
    bcost = COST_INFTY;

    n = size - 1;

    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < sm_lsize);c++) {

        // ###################### //
        // ##      i < j       ## //
        // ###################### //
        if(ctx > by) {
            i = by + 1;
            j = ctx + 1;

            /*
             * [0,i-1] + [i+k,j+k-1]
             */
            dist = GPU_DIST_COORD(i - 1,i + k);     // D[i-1,i+k]

            wait  = j - i;                          // W[i+k,j+k-1] = j + k - 1 - i - k + (i + k > 0)
                                                    //              = j - i - 1 + 1 = j - i
            cost  = sm_cost[i - 1] +                // C[0,i-1]
                    sm_cost[j + k - 1] +            // C[i+k,j+k-1]
                    wait * (                        // W[i+k,j+k-1]
                    sm_time[i - 1] +                // T[0,i-1]
                    dist );                         // D[i-1,i+k] = D[i+k,i-1]

            time  = sm_time[i - 1] +                // T[0,i-1]
                    GPU_ADS_TIME(i + k,j + k - 1) + // T[i+k,j+k-1]
                    dist;                           // D[i-1,i+k] = D[i+k,i-1]
            /*
             * [0,i-1] + [i+k,j+k-1] + [i,i+k-1]
             */
            gm_data = ADS_COST_PTR(gm_ads,sm_rsize) +   // C[i]
                      sm_rsize*i;

            dist = GPU_DIST_COORD(j + k - 1,i);     // D[j+k-1,i]

            wait = k;                               //  W[i,i+k-1] = i + k - 1 - i + (i > 0)
                                                    //             = k - 1 + 1 = k
            cost += gm_data[i + k - 1] +            // C[i,i+k-1]
                    wait * (                        // W[i,i+k-1]
                    time +                          // T([0,i-1] + [i+k,j+k-1])
                    dist );                         // D[j+k-1,i] = D[i,j+k-1]

            time += GPU_ADS_TIME(i,i + k - 1) +     // T[i,i+k-1]
                    dist;                           // D[j+k-1,i] = D[i,j+k-1]

            /*
             * [0,i-1] + [i+k,j+k-1] + [i,i+k-1] + [j+k,n]
             */
            if(j + k <= n) {
                gm_data = ADS_COST_PTR(gm_ads,sm_rsize) +  // C[j+k]
                          sm_rsize*(j + k);

                dist = GPU_DIST_COORD(i + k - 1,j + k);    // D[i+k-1,j+k]

                wait  = n - j - k + 1;              // W[j+k,n] = n - j - k + (j+k > 0)
                                                    //          = n - j - k + 1
                cost += gm_data[n] +                // C[j+k,n]
                        wait * (                    // W[j+k,n]
                        time +                      // T([0,i-1] + [i+k,j+k-1] + [i,i+k-1])
                        dist );                     // D[i+k-1,j+k] = D[j+k,i+k-1]
            }
        }
        // ###################### //
        // ##      i > j      ## //
        // ###################### //
        else {
            i = by  + 2;
            j = ctx + 1;

            /*
             * [0,j-1] + [i,i+k-1]
             */
            gm_data = ADS_COST_PTR(gm_ads,sm_rsize) +   // C[i]
                      sm_rsize*i;

            dist  = GPU_DIST_COORD(j - 1,i);        // D[j-1,i]

            wait  = k;                              // W[i,i+k-1] = i + k - 1 - i + (i > 0)
                                                    //            = k - 1 + 1 = k
            cost  = sm_cost[j - 1] +                // C[0,j-1]
                    gm_data[i + k - 1] +            // C[i,i+k-1]
                    wait * (                        // W[i,i+k-1]
                    sm_time[j - 1] +                // T[0,j-1]
                    dist);                          // D[j-1,i] = D[i,j-1]

            time  = sm_time[j - 1] +                // T[0,j-1]
                    GPU_ADS_TIME(i,i + k - 1) +     // T[i,i + k - 1]
                    dist;                           // D[j-1,i] = D[i,j-1]
            /*
             * [0,j-1] + [i,i+k-1] + [j,i-1]
             */
            gm_data = ADS_COST_PTR(gm_ads,sm_rsize) +   // C[j]
                      sm_rsize*j;

            dist = GPU_DIST_COORD(i + k - 1,j);     // D[i+k-1,j]

            wait  = i - j;                          // W[j,i-1] = i - 1 - j + (j > 0)
                                                    //          = i - j - 1 + 1 = i - j
            cost += gm_data[i - 1] +                // C[j,i-1]
                    wait * (                        // W[j,i-1]
                    time +                          // T([0,j-1] + [i,i+k-1])
                    dist );                         // D[i+k-1,j] = D[j,i+k-1]

            time += GPU_ADS_TIME(j,i - 1) +         // T[j,i-1]
                    dist;                           // D[i+k-1,j] = D[j,i+k-1]
            /*
             * [0,j-1] + [i,i+k-1] + [j,i-1] + [i+k,n]
             */
            if(i + k <= n) {
                gm_data = ADS_COST_PTR(gm_ads,sm_rsize) +   // C[i+k]
                          sm_rsize*(i + k);

                dist = GPU_DIST_COORD(i - 1,i + k);         // D[i-1,i+k]

                wait  = n - i - k + 1;             // W[i+k,n] = n - i - k + (i+k > 0)

                cost += gm_data[n] +               // C[i+k,n]
                        wait * (                   // W[i+k,n]
                        time +                     // T([0,i-1] + [i+k,j+k-1] + [i,i+k-1])
                        dist);                     // D[i-1,i+k] = D[i+k,i-1]
            }
        }

        cost = cost - sm_scost;

        if(cost < bcost) {
            bcost = cost;
            bmove = GPU_MOVE_PACKID(i,j,MLMI_OROPT(k));
        }

        k4printf("GPU_OROPT%u(%d,%d) = %d\n",k,i,j,cost);
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
        k4printf("Block %d: GPU_OROPT%d(%u,%u) = %d\n",
                        by,k,
                        gm_move[by].s.i,
                        gm_move[by].s.j,
                        gm_move[by].s.cost);
    }

/*
    // MIN cost reduction
    for(i=GPU_DIVCEIL(n,2);i > 0;i >>= 1) {
        if(tx < i) {
            if((tx + i < n) && (sm_move[tx] > sm_move[tx + i])) {
                sm_move[tx] = sm_move[tx + i];
                sm_move[n + tx] = sm_move[n + tx + i];
            }
        }
        __syncthreads();
    }

    if(tx == 0) {
        gm_move[by].w = GPU_MOVE_PACK64(sm_move[0],sm_move[n]);
        k4printf("Block %d: GPU_2OPT(%u,%u) = %d\n",
                        by,gm_move[by].s.i,gm_move[by].s.j,gm_move[by].s.cost);
    }
*/
}


__global__
void
kernelOrOptTotal(const MLADSData *gm_ads, MLMovePack *gm_move, const int k, const int size)
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
           *sm_move,            // Moves id/cost
           *sm_time,            // ADS time
           *sm_cost,            // ADS cost
            sm_rsize,           // ADS row size
            sm_scost,           // Solution cost
            sm_tour,            // Tour/path
            sm_lsize;           // Loop size
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
     * +--------+--------+------+------+
     * | coordx | coordy | time | cost |
     * +--------+--------+------+------+
     *                   ^
     *                   |
     *                   move
     */
    // Only thread 0 initializes shared variables
    if(tx == 0) {
        sm_coordx = sm_buffer;
        sm_coordy = sm_coordx + size;
        sm_time   = sm_coordy + size;
        sm_cost   = sm_time   + size;
        sm_move   = sm_time;

        sm_rsize = gm_ads->s.rowElems;
        sm_scost = gm_ads->s.solCost;
        sm_tour  = gm_ads->s.tour;
        sm_round = gm_ads->s.round * 0.5F;

        sm_lsize = size - k + !sm_tour - 1;
    }
    __syncthreads();

    if((k < 1) || (k > 3))
        k4printf("ERROR k=%d\tsm_tour=%d\tsm_rsize=%d\n",k,sm_tour,sm_rsize);

    // Number of chunks
    cmax  = GPU_DIVCEIL(size,blockDim.x);

    // Get clients coordinates
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
    // Row C[0] from ADS: columns 0 to by
    gm_data = ADS_COST_PTR(gm_ads,sm_rsize);
    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) <= by);c++)
        sm_cost[ctx] = gm_data[ctx];         // C[0]

    // Row C[by + k + 1] from ADS: columns by + k + 1 to N-T-1
    n = size - by - k - sm_tour - 1;
    i = by + k + 1;
    gm_data = ADS_COST_PTR(gm_ads,sm_rsize) +
             sm_rsize*i;
    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) <= n);c++)
        sm_cost[i + ctx] = gm_data[i + ctx]; // C[by + k + 1]

    // Wait all threads synchronize
    __syncthreads();

    // Best move/cost of chunk
    bmove = 0;
    bcost = COST_INFTY;

    n = size - 1;

    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < sm_lsize);c++) {

        // ###################### //
        // ##      i < j       ## //
        // ###################### //
        if(ctx > by) {
            i = by + 1;
            j = ctx + 1;

            /*
             * [0,i-1] + [i+k,j+k-1]
             */
            dist = GPU_DIST_COORD(i - 1,i + k);     // D[i-1,i+k]

            wait  = j - i;                          // W[i+k,j+k-1] = j + k - 1 - i - k + (i + k > 0)
                                                    //              = j - i - 1 + 1 = j - i
            cost  = sm_cost[i - 1] +                // C[0,i-1]
                    sm_cost[j + k - 1] +            // C[i+k,j+k-1]
                    wait * (                        // W[i+k,j+k-1]
                    sm_time[i - 1] +                // T[0,i-1]
                    dist );                         // D[i-1,i+k] = D[i+k,i-1]

            time  = sm_time[i - 1] +                // T[0,i-1]
                    GPU_ADS_TIME(i + k,j + k - 1) + // T[i+k,j+k-1]
                    dist;                           // D[i-1,i+k] = D[i+k,i-1]
            /*
             * [0,i-1] + [i+k,j+k-1] + [i,i+k-1]
             */
            gm_data = ADS_COST_PTR(gm_ads,sm_rsize) +   // C[i]
                      sm_rsize*i;

            dist = GPU_DIST_COORD(j + k - 1,i);     // D[j+k-1,i]

            wait = k;                               //  W[i,i+k-1] = i + k - 1 - i + (i > 0)
                                                    //             = k - 1 + 1 = k
            cost += gm_data[i + k - 1] +            // C[i,i+k-1]
                    wait * (                        // W[i,i+k-1]
                    time +                          // T([0,i-1] + [i+k,j+k-1])
                    dist );                         // D[j+k-1,i] = D[i,j+k-1]

            time += GPU_ADS_TIME(i,i + k - 1) +     // T[i,i+k-1]
                    dist;                           // D[j+k-1,i] = D[i,j+k-1]

            /*
             * [0,i-1] + [i+k,j+k-1] + [i,i+k-1] + [j+k,n]
             */
            if(j + k <= n) {
                gm_data = ADS_COST_PTR(gm_ads,sm_rsize) +  // C[j+k]
                          sm_rsize*(j + k);

                dist = GPU_DIST_COORD(i + k - 1,j + k);    // D[i+k-1,j+k]

                wait  = n - j - k + 1;              // W[j+k,n] = n - j - k + (j+k > 0)
                                                    //          = n - j - k + 1
                cost += gm_data[n] +                // C[j+k,n]
                        wait * (                    // W[j+k,n]
                        time +                      // T([0,i-1] + [i+k,j+k-1] + [i,i+k-1])
                        dist );                     // D[i+k-1,j+k] = D[j+k,i+k-1]
            }
        }
        // ###################### //
        // ##      i > j      ## //
        // ###################### //
        else {
            i = by  + 2;
            j = ctx + 1;

            /*
             * [0,j-1] + [i,i+k-1]
             */
            gm_data = ADS_COST_PTR(gm_ads,sm_rsize) +   // C[i]
                      sm_rsize*i;

            dist  = GPU_DIST_COORD(j - 1,i);        // D[j-1,i]

            wait  = k;                              // W[i,i+k-1] = i + k - 1 - i + (i > 0)
                                                    //            = k - 1 + 1 = k
            cost  = sm_cost[j - 1] +                // C[0,j-1]
                    gm_data[i + k - 1] +            // C[i,i+k-1]
                    wait * (                        // W[i,i+k-1]
                    sm_time[j - 1] +                // T[0,j-1]
                    dist);                          // D[j-1,i] = D[i,j-1]

            time  = sm_time[j - 1] +                // T[0,j-1]
                    GPU_ADS_TIME(i,i + k - 1) +     // T[i,i + k - 1]
                    dist;                           // D[j-1,i] = D[i,j-1]
            /*
             * [0,j-1] + [i,i+k-1] + [j,i-1]
             */
            gm_data = ADS_COST_PTR(gm_ads,sm_rsize) +   // C[j]
                      sm_rsize*j;

            dist = GPU_DIST_COORD(i + k - 1,j);     // D[i+k-1,j]

            wait  = i - j;                          // W[j,i-1] = i - 1 - j + (j > 0)
                                                    //          = i - j - 1 + 1 = i - j
            cost += gm_data[i - 1] +                // C[j,i-1]
                    wait * (                        // W[j,i-1]
                    time +                          // T([0,j-1] + [i,i+k-1])
                    dist );                         // D[i+k-1,j] = D[j,i+k-1]

            time += GPU_ADS_TIME(j,i - 1) +         // T[j,i-1]
                    dist;                           // D[i+k-1,j] = D[j,i+k-1]
            /*
             * [0,j-1] + [i,i+k-1] + [j,i-1] + [i+k,n]
             */
            if(i + k <= n) {
                gm_data = ADS_COST_PTR(gm_ads,sm_rsize) +   // C[i+k]
                          sm_rsize*(i + k);

                dist = GPU_DIST_COORD(i - 1,i + k);         // D[i-1,i+k]

                wait  = n - i - k + 1;             // W[i+k,n] = n - i - k + (i+k > 0)

                cost += gm_data[n] +               // C[i+k,n]
                        wait * (                   // W[i+k,n]
                        time +                     // T([0,i-1] + [i+k,j+k-1] + [i,i+k-1])
                        dist);                     // D[i-1,i+k] = D[i+k,i-1]
            }
        }

        cost = cost - sm_scost;

        if(cost < bcost) {
            bcost = cost;
            bmove = GPU_MOVE_PACKID(i,j,MLMI_OROPT(k));
        }

        k4printf("GPU_OROPT%u(%d,%d) = %d\n",k,i,j,cost);
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
kernelOrOpt(const MLADSData *gm_ads, MLMovePack *gm_move, const int k, const int size)
{
    /*!
     * Shared memory variables
     */
    extern  __shared__
    int     sm_buffer[];        // Dynamic shared memory buffer

    __shared__
    int    *sm_soldist,         // Solution distances
           *sm_coordx,          // Clients x-coordinates
           *sm_coordy,          // Clients y-coordinates
           *sm_move,            // Moves id/cost
            sm_rsize,           // ADS row size
            sm_tour;            // Tour/path
    /*!
     * Local variables
     */
    uint   *gm_data;            // Points to some ADS data
    int     c,                  // Chunk no
            ctx,                // Tx index for chunk
            cmax,               // Number of chunks
            csize;              // Chunk size
    int     cost,               // Cost improvement
            i,j,                // Movement indexes
            l,s;                // Auxiliary

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
    // Only thread 0 initializes shared variables
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
    // Chunck size
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

    for(c=0;(c < cmax) && ((ctx = c*blockDim.x + tx) < size);c++) {

        cost = COST_INFTY;
        i = j = 0;

        if(ctx < size - sm_tour - k - 1) {

            if(ctx >= by) {
                //
                // i <= j
                //
                i = ctx - by + 1;
                j = ctx + 2;

                cost  = (size - i)     * ( int(GPU_DIST_COORD(i - 1,i + k))     - sm_soldist[i] );
                cost += (size - j - k) * ( int(GPU_DIST_COORD(i + k - 1,j + k)) - sm_soldist[j + k] );
                cost += (size - j)     * ( int(GPU_DIST_COORD(j + k - 1,i)) ) -
                        (size - i - k) * sm_soldist[i + k];

                s = 0;
                for(l=i + 1;l < i + k;l++)
                    s += sm_soldist[l];
                cost += (i - j) * s;

                s = 0;
                for(l=i + k + 1;l < j + k;l++)
                    s += sm_soldist[l];
                cost += k * s;
            }
            else {
                //
                // i > j
                //
                i = size - ctx - k - sm_tour;
                j = by - ctx;

                cost  = (size - j)     * ( int(GPU_DIST_COORD(j - 1,i))     - sm_soldist[j] );
                cost += (size - i - k) * ( int(GPU_DIST_COORD(i - 1,i + k)) - sm_soldist[i + k] );
                cost += (size - j - k) * ( int(GPU_DIST_COORD(i + k - 1,j)) ) -
                        (size - i)     * sm_soldist[i];

                s = 0;
                for(l=i + 1;l < i + k;l++)
                    s += sm_soldist[l];
                cost += (i - j) * s;

                s = 0;
                for(l=j + 1;l < i;l++)
                    s += sm_soldist[l];
                cost -= k * s;
            }

            if(cost < sm_move[tx]) {
                sm_move[tx] = cost;
                sm_move[tx + csize] = GPU_MOVE_PACKID(i,j,MLMI_OROPT(k));
            }
            k4printf("GPU_OROPT%d(%d,%d) = %d\n",k,i,j,cost);
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
        k4printf("Block %d: GPU_OROPT%d(%u,%u) = %d\n",
                        by,k,
                        gm_move[by].s.i,
                        gm_move[by].s.j,
                        gm_move[by].s.cost);
    }

/*
    // MIN cost reduction
    for(i=GPU_DIVCEIL(csize,2);i > 0;i >>= 1) {
        if(tx < i) {
            if((tx + i < csize) && (sm_move[tx] > sm_move[tx + i])) {
                sm_move[tx] = sm_move[tx + i];
                sm_move[tx + csize] = sm_move[tx + i + csize];
            }
        }
        __syncthreads();
    }

    if(tx == 0) {
        gm_move[by].w = GPU_MOVE_PACK64(sm_move[0],sm_move[csize]);
        k4printf("Block %d: MIN OROPT%d(%u,%u) = %4d\n",by,k,
                        uint(gm_move[by].s.i),
                        uint(gm_move[by].s.j),
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
MLKernelOrOpt::defineKernelGrid()
{
    int gsize,
        bsize;

#ifdef MLP_GPU_ADS

    /*
     *  Compute dynamic shared memory size
     */
    shared = 5 * GPU_BLOCK_CEIL(int,solSize);
    /*
     * Compute grid
     */
    grid.x = 1;
    grid.y = solSize + !problem.costTour - tag - 2;
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
    grid.y = solSize + !problem.costTour - tag - 2;    ////solSize - problem.params.costTour - tag; // TODO: FIX!!
    grid.z = 1;

#endif

    if(gpuOccupancyMaxPotentialBlockSizeVariableSMem(&gsize,&bsize,kernelOrOpt,
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
MLKernelOrOpt::launchKernel()
{
    l4printf("Calling kernel %s\n",name);

    if(tag < 1 || tag > 3)
        EXCEPTION("Invalid K value (k=%d)",tag);

    // Update kernel calls counter
    callCount++;
    // Kernel in execution
    flagExec = true;

    // Calls kernel
//    if(!isTotal)
//    	std::cout << "MLKernelOrOpt" << tag << "::launchKernel - grid(x: " << grid.x << ", y: " << grid.y << ", z: " << grid.z
//			<< ") block(x: " << block.x << ", y: " << block.y << ", z: " << block.z << ") shared: " << shared << std::endl;
    	kernelOrOpt<<<grid,block,shared,stream>>>(adsData,moveData,tag,solSize);
//    else
//    	kernelOrOptTotal<<<grid,block,shared,stream>>>(adsData,moveData,tag,solSize);
}

void
MLKernelOrOpt::applyMove(MLMove &move)
{
    uint    i,j,l;
    ushort  temp[tag];

    //solution->showCostCalc("BASE    : ");

    solution->cost += move.cost;
    solution->time = sysTimer();
    solution->move = move;

    i = move.i;
    j = move.j;

    for(l=0;l < tag;l++)
        temp[l] = solution->clients[l + i];

    if(i < j) {
        for(l=i + tag;l < j + tag;l++)
            solution->clients[i++] = solution->clients[l];

        for(l=0;l < tag;l++)
            solution->clients[i++] = temp[l];

        i = move.i;
        j = move.j + tag;
        if(j >= solution->clientCount)
            j--;
    }
    else
    if(i > j) {
        for(l=i - 1;l >= j;l--)
            solution->clients[l + tag] = solution->clients[l];

        for(l=0;l < tag;l++)
            solution->clients[l + j] = temp[l];

        i = move.j;
        j = move.i + tag;
        if(j >= solution->clientCount)
            j--;
    }
    for(;i <= j;i++)
        solution->weights[i] = problem.clients[ solution->clients[i - 1] ].weight[ solution->clients[i] ];

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
