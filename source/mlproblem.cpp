/**
 * @file	mlproblem.h
 *
 * @brief	Handle a general optimization problem
 *
 * @author	Eyder Rios
 * @date    2015-05-28
 */

#include <math.h>
#include "except.h"
#include "log.h"
#include "utils.h"
#include "tsplib.h"
#include "gpu.h"
#include "mlproblem.h"


using namespace std;


// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //


#define RLIB_FILE_SIGNATURE     "RLib: Traveling Salesman Problem (TSP)"


// ################################################################################ //
// ##                                                                            ## //
// ##                                GLOBAL VARIABLES                            ## //
// ##                                                                            ## //
// ################################################################################ //


// ################################################################################ //
// ##                                                                            ## //
// ##                               CLASS OFProblem                              ## //
// ##                                                                            ## //
// ################################################################################ //

void
MLProblem::free()
{
    if(clients) {
        for(uint i=0;i < size;i++)
            delete clients[i].weight;
        delete[] clients;
    }

    clients = NULL;
    size = 0;
}

void
MLProblem::allocData(uint siz)
{
    int     **rows,
             *base;
    uint      i,n;

    free();

    // If tour, one client (depot) added to end
    size = siz + int(costTour);
    clients = new MLClientData[size];

    for(i=0;i < size;i++) {
        clients[i].id = 0;
        clients[i].weight = new int[size];
    }
}

void
MLProblem::load(const char *fname)
{
    tsplib::TSPLibInstance   tsp;
    MLClientData            *client;
    uint                     i,j,ii,jj;
    float                    s,dr;
    int                      xmin,ymin;

    timeLoad = sysTimer();

    dr = distRound ? 0.5F : 0.0F;
    l4printf("Euclidean distance round: %0.2f\n",dr);

    // Load TSPLIB file
    if(tsp.load(fname)) {
        // Check if coordinates type is EUC_2D
        if(tsp.ewtype != tsplib::TWT_EUC_2D)
            EXCEPTION("Invalid instance: this application deal only with EUC_2D coordinates");

        l4printf("TSP name: %s\n",tsp.name.c_str());

        // Copy filename
        strncpy(filename,fname,sizeof(filename));
        filename[sizeof(filename) - 1] = '\0';
        // Copy instance name
        strncpy(name,tsp.name.c_str(),sizeof(name));
        name[sizeof(name) - 1] = '\0';

        allocData(tsp.dimension);

        xmin = INT_MAX;
        ymin = INT_MAX;
        for(i=0;i < size;i++) {
            client = clients + i;

            ii = i % tsp.dimension;

            client->id = tsp.nodeData[ii].id;
            client->x  = tsplib::TSPLibInstance::inear(tsp.nodeData[ii].x);
            client->y  = tsplib::TSPLibInstance::inear(tsp.nodeData[ii].y);

//            if((client->x != tsp.nodeData[ii].x) || (client->y != tsp.nodeData[ii].y))
//                lprintf("Coord diff: int(%d,%d) != float(%f,%f)\tin %s\n",
//                                client->x,client->y,
//                                tsp.nodeData[ii].x,tsp.nodeData[ii].y,
//                                filename);

            // Get min(x) and min(y)
            if(client->x < xmin)
                xmin = client->x;
            if(client->y < ymin)
                ymin = client->y;
        }

        /*
         * If any negative coordinate, shift coordinates to get
         * only non negative numbers.
         * This should not affect client distances.
         */
        if(coordShift) {
            shiftX = (xmin < 0) ? -xmin : 0;
            shiftY = (ymin < 0) ? -ymin : 0;

            // Is necessary any coordinate shifting?
            if(shiftX || shiftY) {
                l4printf("Coordinate shifted: %s\n",filename);
                for(i=0;i < size;i++) {
                    client = clients + i;

                    client->x += shiftX;
                    client->y += shiftY;
                }
            }
        }

        maxWeight = 0;
        for(i=0;i < size;i++) {
            ii = i % tsp.dimension;
            for(j=0;j < size;j++) {
                jj = j % tsp.dimension;

                s = (clients[ii].x - clients[jj].x)*(clients[ii].x - clients[jj].x) +
                    (clients[ii].y - clients[jj].y)*(clients[ii].y - clients[jj].y);

                clients[i].weight[j] = int(sqrtf(s) + dr);
                if(clients[i].weight[j] > maxWeight)
                    maxWeight = clients[i].weight[j];
            }
        }
    }
    else
        EXCEPTION("Error opening instance file: '%s'",fname);

#if 0

    FILE  *fo;

//#define TSP_STDOUT
#ifndef TSP_STDOUT
    char   name[128];

    if(params.outputDir)
        sprintf(name,"%s/%s.tsp",params.outputDir,tsp.name.c_str());
    else
        sprintf(name,"%s.tsp",tsp.name.c_str());

    if((fo = fopen(name,"wt")) == NULL)
        EXCEPTION("Error creating instance file: %s",fname);
    lprintf("Writing to %s\n",name);
#else
    fo = stdout;
#endif
    fprintf(fo,"RLib: Traveling Salesman Problem (TSP)\n");
    fprintf(fo,"%s\n",tsp.name.c_str());
    fprintf(fo,"%u %u\n",tsp.dimension,0);

    for(i=0;i < tsp.dimension;i++) {
        fprintf(fo,"%u",i);
        for(j=0;j < i;j++)
            fprintf(fo," %d",clients[i].weight[j]);
        fprintf(fo,"\n");
    }
#ifndef TSP_STDOUT
    fclose(fo);
#endif

#endif

    timeLoad = sysTimer() - timeLoad;
}

void
MLProblem::save(const char *fname)
{
    ofstream  fout;
    char      bname[128];
    uint      i,j;

    strcpy(bname,fname);
    stripext(bname);

    fout.open(fname);

    fout << RLIB_FILE_SIGNATURE << '\n';
    fout << bname << '\n';
    fout << size << ' ' << 0 << '\n';
    for(i=0;i < size;i++) {
        fout << i;
        for(j=0;j < i;j++)
            fout << ' ' << clients[i].weight[j];
        fout << '\n';
    }
    fout.close();
}

//MLSolution *
//MLProblem::createSolution()
//{
//    return NULL;
//}
