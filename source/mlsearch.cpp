/**
 * @file   mlsearch.cpp
 *
 * @brief  MLP CPU local searches.
 *
 * @author Eyder Rios
 * @date   2014-06-01
 */

#include "log.h"
#include "utils.h"
#include "mlsearch.h"

// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //

#define CPU_ADS_TIME(i,j)       (adsTime[j] - adsTime[i])


// ################################################################################ //
// ##                                                                            ## //
// ##                                CLASS MLSearch                              ## //
// ##                                                                            ## //
// ################################################################################ //

// ################################################################################ //
// ##                                                                            ## //
// ##                           CLASS MLSearchSwap                               ## //
// ##                                                                            ## //
// ################################################################################ //

void
MLSearchSwap::launchSearch(MLSolution *solBase, MLMove &move)
{
    int     i,j,p,n,
            in,jn,
            size,
            cost;

#ifdef MLP_CPU_ADS
    uint   *adsTime,
           *adsCost,
           *adsCostRow;
    int     dist,
            wait,
            time;

    adsTime = ADS_TIME_PTR(solBase->adsData,solBase->adsRowElems);
    adsCost = ADS_COST_PTR(solBase->adsData,solBase->adsRowElems);
#endif

    callCount++;

    size = solBase->size;

    p  = !solBase->problem.params.costTour;
    in = size - 3 + p;
    jn = size - 1 + p;
    n  = size - 1;

    // Initialize movement structure
    move = MOVE_INFTY;
    move.id = MLMI_SWAP;

    for(i=1;i < in;i++) {

        for(j=i + 2;j < jn;j++) {

#ifdef MLP_CPU_ADS
            /*
             * [0,i-1] + [j,j]
             */
            dist = solBase->dist(i - 1,j);      // D[i-1,j]]

            // wait = 1;                        // W[j,j]   = j - j + (j > 0)
                                                //          = 1
            cost = adsCost[i - 1] +             // C[0,i-1]
                   0 +                          // C[j,j] = 0
                   1 * (                        // W[j,j] = 1
                   adsTime[i - 1] +             // T[0,i-1]
                   dist );                      // D[i-1,j]

            time = adsTime[i - 1] +             // T[0,i-1]
                   0 +                          // T[j,j] = 0
                   dist;                        // D[i-1,j]

            /*
             * [0,i-1] + [j,j] + [i+1,j-1]
             */
            // row C[i + 1]
            adsCostRow = adsCost + (i + 1)*solBase->adsRowElems;

            dist = solBase->dist(j,i + 1);      // D[j,i+1]

            wait  = j - i - 1;                  // W[i+1,j-1] = j - 1 - i - 1 + (i+1 > 0)
                                                //            = j - i - 2 + 1 = j - i - 1
            cost += adsCostRow[j - 1] +         // C[i+1,j-1]
                    wait * (                    // W[i+1,j-1]
                    time +                      // T([0,i-1] + [j,j])
                    dist );                     // D[j,i+1]

            time += CPU_ADS_TIME(i + 1,j - 1) + // T[i+1,j-1]
                    dist;                       // D[j,i+1]

            /*
             * [0,i-1] + [j,j] + [i+1,j-1] + [i,i]
             */
            // row C[i]
            adsCostRow = adsCost + i*solBase->adsRowElems;

            dist = solBase->dist(j - 1,i);      // D[j-1,i]

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
                // row C[j + 1]
                adsCostRow = adsCost + (j + 1)*solBase->adsRowElems;

                dist = solBase->dist(i,j + 1);     // D[i,j+1]

                wait  = n - j;                     // W[j+1,n] = n - j - 1 + (j+1 > 0)
                                                   //          = n - j - 1 + 1 = n - j
                cost += adsCostRow[n] +            // C[j+1,n]
                        wait * (                   // W[j+1,n]
                        time +                     // T([0,i-1] + [j,j] + [i+1,j-1] + [i,i])
                        dist );                    // D[i,j+1]
            }

            cost = cost - solBase->cost;

#else

#ifdef MLP_CALC_DIST
            cost  = (size - i) * ( solBase->calcDist(i - 1,j) - solBase->calcDist(i - 1,i) );
            cost += (size - i - 1) * ( solBase->calcDist(j,i + 1) - solBase->calcDist(i,i + 1) );
            cost += (size - j) * ( solBase->calcDist(j - 1,i) - solBase->calcDist(j - 1,j) );
            if(j + 1 < size)
                cost += (size - j - 1) * ( solBase->calcDist(i,j + 1) - solBase->calcDist(j,j + 1) );
#else
            cost  = (size - i) * (solBase->dist(i - 1, j) - solBase->dist(i - 1, i));
            cost += (size - i - 1) * (solBase->dist(j, i + 1) - solBase->dist(i, i + 1));
            cost += (size - j) * (solBase->dist(j - 1, i) - solBase->dist(j - 1, j));
            if(j + 1 < size)
                cost += (size - j - 1) * (solBase->dist(i, j + 1) - solBase->dist(j, j + 1));
#endif

#endif

            if(cost < move.cost) {
                move.i = i;
                move.j = j;
                move.cost = cost;
            }

            l4printf("CPU_SWAP(%u,%u) = %d\n",i,j,cost);
        }
    }
}

void
MLSearchSwap::applyMove(MLSolution *sol, MLMove &move)
{
    int    i,j;
    ushort t;

    i = move.i;
    j = move.j;

    sol->cost += move.cost;
    sol->time  = sysTimer();
    sol->move  = move;

    t = sol->clients[i];
    sol->clients[i] = sol->clients[j];
    sol->clients[j] = t;

    sol->weights[i] = sol->dist(i - 1,i);
    sol->weights[i + 1] = sol->dist(i,i + 1);

    sol->weights[j] = sol->dist(j - 1,j);
    if(j + 1 < sol->clientCount)
        sol->weights[j + 1] = sol->dist(j,j + 1);

#ifdef MLP_COST_CHECK
    if(sol->problem.params.checkCost) {
        uint ccost = sol->costCalc();
        l4printf("CHECK COST: 2OPT,\tcost=%u, check=%u\n",sol->cost,ccost);
        if(sol->cost != ccost) {
            lprintf("%s(%u,%u): wrong=%u, right=%u\n",name,move.i,move.j,sol->cost,ccost);
            EXCEPTION("INVALID COST!");
        }
    }
#endif
}

// ################################################################################ //
// ##                                                                            ## //
// ##                           CLASS MLSearch2Opt                               ## //
// ##                                                                            ## //
// ################################################################################ //

void
MLSearch2Opt::launchSearch(MLSolution *solBase, MLMove &move)
{
    int     i,j,l,p,
            n,nn,
            cost;

#ifdef MLP_CPU_ADS
    uint   *adsTime,
           *adsCost,
           *adsCostRow;
    int     dist,
            wait,
            time;

    adsTime = ADS_TIME_PTR(solBase->adsData,solBase->adsRowElems);
    adsCost = ADS_COST_PTR(solBase->adsData,solBase->adsRowElems);
#endif

    callCount++;

    p  = !solBase->problem.params.costTour;
    // Last solution index
    n  = solBase->size - 1;
    // Movement index limit
    nn = n + p;

    // Initialize movement structure
    move = MOVE_INFTY;
    move.id = MLMI_2OPT;

    for(i=1;i < nn;i++) {

        for(j=i+1;j < nn;j++) {

#ifdef MLP_CPU_ADS
            /*
             * [0,i-1] + [j,i]
             */
            // row C[j]
            adsCostRow = adsCost + j*solBase->adsRowElems;

            dist = solBase->dist(i - 1,j);         // D[i-1,j]

            wait  = j - i + 1;                     // W[j,i] = j - i + (j > 0)
                                                   //        = j - i + 1
            cost  = adsCost[i - 1] +               // C[0,i-1]
                    adsCostRow[i] +                // C[j,i]
                    wait * (                       // W[j,i]
                    adsTime[i - 1] +               // T[0,i-1]
                    dist );                        // D[i-1,j] = D[j,i-1]

            time  = adsTime[i - 1] +               // T[0,i-1]
                    CPU_ADS_TIME(i,j) +            // T[j,i] = T[i,j]
                    dist;                          // D[i-1,j] = D[j,i-1]

            /*
             * [0,i-1] + [j,i] + [j+1,n]
             */
            if(j + 1 <= n) {
                // row C[j]
                adsCostRow = adsCost + (j + 1)*solBase->adsRowElems;

                dist = solBase->dist(i,j + 1);     // D[i,j+1]

                wait  = n - j;                     // W[j+1,n] = n - j - 1 + (j+1 > 0)
                                                   //          = n - j - 1 + 1 = n - j
                cost += adsCostRow[n] +            // C[j+1,n]
                        wait * (                   // W[j+1,n]
                        time +                     // T([0,i-1] + [j,i])
                        dist );                    // D[i,j+1] = D[j+1,i]
            }

            cost = cost - solBase->cost;
#else

#ifdef MLP_CALC_DIST
            cost  = (solBase->size - i) * ( solBase->calcDist(i - 1,j) - solBase->calcDist(i - 1,i) );
            if(j + 1 < solBase->size)
                cost += (solBase->size - j - 1) * ( solBase->calcDist(i,j + 1) - solBase->calcDist(j,j + 1) );
            for(l=0;l < j - i;l++)
                cost += (solBase->size - i - l - 1) * ( solBase->calcDist(j - l,j - l - 1) - solBase->calcDist(i + l,i + l + 1) );
#else
            cost  = (solBase->size - i) * ( solBase->dist(i - 1,j) - solBase->dist(i - 1,i) );
            if(j + 1 < solBase->size)
                cost += (solBase->size - j - 1) * ( solBase->dist(i,j + 1) - solBase->dist(j,j + 1) );
            for(l=0;l < j - i;l++)
                cost += (solBase->size - i - l - 1) * ( solBase->dist(j - l,j - l - 1) - solBase->dist(i + l,i + l + 1) );
#endif

#endif

            if(cost < move.cost) {
                move.i = i;
                move.j = j;
                move.cost = cost;
            }

            l4printf("CPU_2OPT(%u,%u) = %d\n",i,j,cost);
        }
    }
}

void
MLSearch2Opt::applyMove(MLSolution *sol, MLMove &move)
{
    ushort  t;
    int     i,j;

    sol->cost += move.cost;
    sol->time  = sysTimer();
    sol->move  = move;

    i = move.i;
    j = move.j;

    if(i < j) {
        while(i < j) {
            t = sol->clients[i];
            sol->clients[i] = sol->clients[j];
            sol->clients[j] = t;
            i++;
            j--;
        }

        j = move.j;
        if(j + 1 < sol->clientCount)
            j++;

        for(i=move.i;i <= j;i++)
            sol->weights[i] = sol->dist(i - 1,i);
    }

#ifdef MLP_COST_CHECK
    if(sol->problem.params.checkCost) {
        uint ccost = sol->costCalc();
        l4printf("CHECK COST: 2OPT,\tcost=%u, check=%u\n",sol->cost,ccost);
        if(sol->cost != ccost) {
            lprintf("%s(%u,%u): wrong=%u, right=%u\n",name,move.i,move.j,sol->cost,ccost);
            EXCEPTION("INVALID COST!");
        }
    }
#endif
}

// ################################################################################ //
// ##                                                                            ## //
// ##                           CLASS MLSearchOrOpt                              ## //
// ##                                                                            ## //
// ################################################################################ //

void
MLSearchOrOpt::launchSearch(MLSolution *solBase, MLMove &move)
{
    int     i,j,l,
            n,nn,p;
    int     cost;

#ifdef MLP_CPU_ADS
    uint   *adsTime,
           *adsCost,
           *adsCostRow;
    int     dist,
            wait,
            time;

    adsTime = ADS_TIME_PTR(solBase->adsData,solBase->adsRowElems);
    adsCost = ADS_COST_PTR(solBase->adsData,solBase->adsRowElems);
#endif

    callCount++;

    p  = !solBase->problem.params.costTour;
    // Last solution index
    n  = solBase->size - 1;
    // Index limit
    nn = solBase->size - k - solBase->problem.params.costTour + 1;

    // Initialize movement structure
    move = MOVE_INFTY;
    move.id = MLMoveId(MLMI_OROPT(k));

    /*********
     * i < j *
     *********/
    for(i=1;i < nn;i++) {

        for(j=i+1;j < nn;j++) {

#ifdef MLP_CPU_ADS
            /*
             * [0,i-1] + [i+k,j+k-1]
             */
            // row C[j]
            adsCostRow = adsCost + (i + k)*solBase->adsRowElems;

            dist = solBase->dist(i - 1,i + k);      // D[i-1,i+k]

            wait  = j - i;                          // W[i+k,j+k-1] = j + k - 1 - i - k + (i + k > 0)
                                                    //              = j - i - 1 + 1 = j - i
            cost  = adsCost[i - 1] +                // C[0,i-1]
                    adsCostRow[j + k - 1] +         // C[i+k,j+k-1]
                    wait * (                        // W[i+k,j+k-1]
                    adsTime[i - 1] +                // T[0,i-1]
                    dist );                         // D[i-1,i+k] = D[i+k,i-1]

            time  = adsTime[i - 1] +                // T[0,i-1]
                    CPU_ADS_TIME(i + k,j + k - 1) + // T[i+k,j+k-1]
                    dist;                           // D[i-1,i+k] = D[i+k,i-1]
            /*
             * [0,i-1] + [i+k,j+k-1] + [i,i+k-1]
             */
            // row C[i]
            adsCostRow = adsCost + i*solBase->adsRowElems;

            dist = solBase->dist(j + k - 1,i);      // D[j+k-1,i]

            wait = k;                               //  W[i,i+k-1] = i + k - 1 - i + (i > 0)
                                                    //             = k - 1 + 1 = k
            cost += adsCostRow[i + k - 1] +         // C[i,i+k-1]
                    wait * (                        // W[i,i+k-1]
                    time +                          // T([0,i-1] + [i+k,j+k-1])
                    dist );                         // D[j+k-1,i] = D[i,j+k-1]

            time += CPU_ADS_TIME(i,i + k - 1) +     // T[i,i+k-1]
                    dist;                           // D[j+k-1,i] = D[i,j+k-1]

            /*
             * [0,i-1] + [i+k,j+k-1] + [i,i+k-1] + [j+k,n]
             */
            if(j + k <= n) {
                // row C[j+k]
                adsCostRow = adsCost + (j + k)*solBase->adsRowElems;

                dist = solBase->dist(i + k - 1,j + k);  // D[i+k-1,j+k]

                wait  = n - j - k + 1;              // W[j+k,n] = n - j - k + (j+k > 0)
                                                    //          = n - j - k + 1
                cost += adsCostRow[n] +             // C[j+k,n]
                        wait * (                    // W[j+k,n]
                        time +                      // T([0,i-1] + [i+k,j+k-1] + [i,i+k-1])
                        dist );                     // D[i+k-1,j+k] = D[j+k,i+k-1]
            }

            cost = cost - solBase->cost;
#else

#ifdef MLP_CALC_DIST
            cost  = (solBase->size - i) * ( solBase->calcDist(i - 1,i + k) - solBase->calcDist(i - 1,i) );
            if(j + k < solBase->size)
                cost += (solBase->size - j - k) * ( solBase->calcDist(i + k - 1,j + k) - solBase->calcDist(j + k - 1,j + k) );
            cost += (solBase->size - j)     *   solBase->calcDist(j + k - 1,i);
            cost -= (solBase->size - i - k) *   solBase->calcDist(i + k - 1,i + k);

            p = 0;
            for(l=i + 1;l < i + k;l++)
                p += solBase->calcDist(l - 1,l);
            cost += (i - j) * p;

            p = 0;
            for(l=i + k + 1;l < j + k;l++)
                p += solBase->calcDist(l - 1,l);
            cost += k * p;
#else
            cost  = (solBase->size - i) * ( solBase->dist(i - 1,i + k) - solBase->dist(i - 1,i) );
            if(j + k < solBase->size)
                cost += (solBase->size - j - k) * ( solBase->dist(i + k - 1,j + k) - solBase->dist(j + k - 1,j + k) );
            cost += (solBase->size - j)     *   solBase->dist(j + k - 1,i);
            cost -= (solBase->size - i - k) *   solBase->dist(i + k - 1,i + k);

            p = 0;
            for(l=i + 1;l < i + k;l++)
                p += int(solBase->dist(l-1,l));
            cost += (i - j) * p;

            p = 0;
            for(l=i + k + 1;l < j + k;l++)
                p += int(solBase->dist(l-1,l));
            cost += k * p;
#endif

#endif

            if(cost < move.cost) {
                move.i = i;
                move.j = j;
                move.cost = cost;
            }

            l4printf("CPU_OROPT%u(%u,%u) = %d\n",k,i,j,cost);
        }
    }

    /*********
     * i > j *
     *********/
    for(i=1;i < nn;i++) {

        for(j=1;j < i;j++) {

#ifdef MLP_CPU_ADS
            /*
             * [0,j-1] + [i,i+k-1]
             */
            // row C[i]
            adsCostRow = adsCost + i*solBase->adsRowElems;

            dist  = solBase->dist(j - 1,i);         // D[j-1,i]

            wait  = k;                              // W[i,i+k-1] = i + k - 1 - i + (i > 0)
                                                    //            = k - 1 + 1 = k
            cost  = adsCost[j - 1] +                // C[0,j-1]
                    adsCostRow[i + k - 1] +         // C[i,i+k-1]
                    wait * (                        // W[i,i+k-1]
                    adsTime[j - 1] +                // T[0,j-1]
                    dist );                         // D[j-1,i] = D[i,j-1]

            time  = adsTime[j - 1] +                // T[0,j-1]
                    CPU_ADS_TIME(i,i + k - 1) +     // T[i,i + k - 1]
                    dist;                           // D[j-1,i] = D[i,j-1]
            /*
             * [0,j-1] + [i,i+k-1] + [j,i-1]
             */
            // row C[j]
            adsCostRow = adsCost + j*solBase->adsRowElems;

            dist = solBase->dist(i + k - 1,j);      // D[i+k-1,j]

            wait  = i - j;                          // W[j,i-1] = i - 1 - j + (j > 0)
                                                    //          = i - j - 1 + 1 = i - j
            cost += adsCostRow[i - 1] +             // C[j,i-1]
                    wait * (                        // W[j,i-1]
                    time +                          // T([0,j-1] + [i,i+k-1])
                    dist );                         // D[i+k-1,j] = D[j,i+k-1]

            time += CPU_ADS_TIME(j,i - 1) +         // T[j,i-1]
                    dist;                           // D[i+k-1,j] = D[j,i+k-1]
            /*
             * [0,j-1] + [i,i+k-1] + [j,i-1] + [i+k,n]
             */
            if(i + k <= n) {
                // row C[i+k]
                adsCostRow = adsCost + (i + k)*solBase->adsRowElems;

                dist = solBase->dist(i - 1,i + k); // D[i-1,i+k]

                wait  = n - i - k + 1;             // W[i+k,n] = n - i - k + (i+k > 0)

                cost += adsCostRow[n] +            // C[i+k,n]
                        wait * (                   // W[i+k,n]
                        time +                     // T([0,i-1] + [i+k,j+k-1] + [i,i+k-1])
                        dist);                     // D[i-1,i+k] = D[i+k,i-1]
            }

            cost = cost - solBase->cost;

#else

#ifdef MLP_CALC_DIST
            cost  = (solBase->size - j) * ( solBase->calcDist(j - 1,i) - solBase->calcDist(j - 1,j) );
            if(i + k < solBase->size)
                cost += (solBase->size - i - k) * ( solBase->calcDist(i - 1,i + k) - solBase->calcDist(i + k - 1,i + k) );
            cost += (solBase->size - j - k) *   solBase->calcDist(i + k - 1,j);
            cost -= (solBase->size - i)     *   solBase->calcDist(i - 1,i);

            p = 0;
            for(l=i + 1;l < i + k;l++)
                p += solBase->calcDist(l - 1,l);
            cost += (i - j) * p;

            p = 0;
            for(l=j + 1;l < i;l++)
                p += solBase->calcDist(l - 1,l);
            cost -= k * p;
#else
            cost  = (solBase->size - j) * ( solBase->dist(j - 1,i) - solBase->dist(j - 1,j) );
            if(i + k < solBase->size)
                cost += (solBase->size - i - k) * ( solBase->dist(i - 1,i + k) - solBase->dist(i + k - 1,i + k) );
            cost += (solBase->size - j - k) * solBase->dist(i + k - 1,j);
            cost -= (solBase->size - i)     * solBase->dist(i - 1,i);

            p = 0;
            for(l=i + 1;l < i + k;l++)
                p += solBase->dist(l - 1,l);
            cost += (i - j) * p;

            p = 0;
            for(l=j + 1;l < i;l++)
                p += solBase->dist(l - 1,l);
            cost -= k * p;
#endif

#endif

            if(cost < move.cost) {
                move.i = i;
                move.j = j;
                move.cost = cost;
            }

            l4printf("CPU_OROPT%u(%u,%u) = %d\n",k,i,j,cost);
        }
    }
}

void
MLSearchOrOpt::applyMove(MLSolution *sol, MLMove &move)
{
    uint    i,j,l;
    ushort  temp[k];

    sol->cost += move.cost;
    sol->time = sysTimer();
    sol->move = move;

    i = move.i;
    j = move.j;

    for(l=0;l < k;l++)
        temp[l] = sol->clients[l + i];

    if(i < j) {
        for(l=i + k;l < j + k;l++)
            sol->clients[i++] = sol->clients[l];

        for(l=0;l < k;l++)
            sol->clients[i++] = temp[l];

        i = move.i;
        j = move.j + k;
        if(j >= sol->clientCount)
            j--;
    }
    else
    if(i > j) {
        for(l=i - 1;l >= j;l--)
            sol->clients[l + k] = sol->clients[l];

        for(l=0;l < k;l++)
            sol->clients[l + j] = temp[l];

        i = move.j;
        j = move.i + k;
        if(j >= sol->clientCount)
            j--;
    }
    for(;i <= j;i++)
        sol->weights[i] = sol->dist(i - 1,i);

#ifdef MLP_COST_CHECK
    if(sol->problem.params.checkCost) {
        uint ccost = sol->costCalc();
        l4printf("CHECK COST: 2OPT,\tcost=%u, check=%u\n",sol->cost,ccost);
        if(sol->cost != ccost) {
            lprintf("%s(%u,%u): wrong=%u, right=%u\n",name,move.i,move.j,sol->cost,ccost);
            EXCEPTION("INVALID COST!");
        }
    }
#endif
}
