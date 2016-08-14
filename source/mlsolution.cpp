/**
 * @file   mlsolution.cpp
 *
 * @brief  Minimum Latency Problem solution.
 *
 * @author Eyder Rios
 * @date   2014-06-01
 */

#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstddef>
#include <math.h>
#include <string.h>
#include "except.h"
#include "log.h"
#include "utils.h"
#include "mlsolution.h"

#include "gpu.h"

using namespace std;

// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //

// ################################################################################ //
// ##                                                                            ## //
// ##                                  DATA TYPES                                ## //
// ##                                                                            ## //
// ################################################################################ //

/*!
 * ClientCompAsc
 *
 * Use by STL std::sort() function to sort clients.
 */
struct ClientCompAsc
{
    MLClientData    *clRef;

    ClientCompAsc(MLClientData  *ref) {
        clRef = ref;
    }
    bool operator()(MLClientData *c1, MLClientData *c2) {
        return clRef->weight[c1->id] < clRef->weight[c2->id];
    }
};

/*!
 * ClientCompDsc
 *
 * Use by STL std::sort() function to sort clients.
 */
struct ClientCompDsc
{
    MLClientData    *clRef;

    ClientCompDsc(MLClientData  *ref) {
        clRef = ref;
    }
    bool operator()(MLClientData *c1, MLClientData *c2) {
        return clRef->weight[c1->id] > clRef->weight[c2->id];
    }
};


// ################################################################################ //
// ##                                                                            ## //
// ##                               GLOBAL DATA                                  ## //
// ##                                                                            ## //
// ################################################################################ //

const char *
MLSolution::costTypeNames[] = {
		"PATH",
        "TOUR",
};

// ################################################################################ //
// ##                                                                            ## //
// ##                                  DATA TYPES                                ## //
// ##                                                                            ## //
// ################################################################################ //

/*!
 * Solution ID generator
 */
uint
MLSolution::idGen = 0;


// ################################################################################ //
// ##                                                                            ## //
// ##                              CLASS MLSolution                              ## //
// ##                                                                            ## //
// ################################################################################ //


MLSolution::MLSolution(MLProblem &prob, int flags) : problem(prob)
{
    if(problem.params.pinnedAlloc)
        adsFlags = flags;
    else
        adsFlags = cudaHostAllocPaged;

    init();
}

MLSolution::~MLSolution()
{
    free();
}

void
MLSolution::init()
{
    id = idGen++;

    size = problem.size;

    clients  = new ushort[2 * size];
    weights  = clients + size;
    dataSize = 2 * size * sizeof(ushort);
    clientCount = 0;

    adsData = NULL;
    adsDataTime = NULL;

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
     * info[0]  : size (ADS buffer size in bytes)
     * info[1]  : rowElems
     * info[2]  : solElems (sol size)
     * info[3]  : solCost
     * info[4]  : tour
     * info[5]  : k
     * info[6]  : reserved
     * ...
     * info[b-1]: reserved
     */
#ifdef GPU_EVAL_ALIGNED
    adsRowElems = GPU_BLKSIZE_GLOBAL(size * sizeof(uint)) / sizeof(uint);
#else
    adsRowElems = size;
#endif

    adsDataSize = ADS_INFO_SIZE + adsRowElems * (size + 3) * sizeof(uint);

//    if(problem.params.cpuAds)
//        adsDataSize = ADS_INFO_SIZE + (size + 3) * adsRowElems * sizeof(uint);
//    else
//        adsDataSize = ADS_INFO_SIZE +         3  * adsRowElems * sizeof(uint);

    move = MOVE_INFTY;
    time = 0;

    l4printf("probSize=%u\tdataSize=%u\tadsDataSize=%u\n",
                    size,dataSize,adsDataSize);
}

void
MLSolution::free()
{
	adsFree();

    if(clients)
        delete[] clients;
    clients = NULL;
	weights = NULL;
	dataSize = 0;

	clientCount = 0;
    cost = COST_INFTY;
}

const char *
MLSolution::costTypeName()
{
    return costTypeNames[problem.params.costTour];
}

ulong
hash(byte *data, uint size)
{
    int   c;
    ulong hash = 5381;

    while(size > 0) {
        hash = ((hash << 5) + hash) + *data;
        data++;
        size--;
    }

    return hash;
}

void
MLSolution::update()
{
    uint    k;

    cost = 0;
    k = clientCount - 1;

    weights[0] = 0;
    for(uint i=1;i < clientCount;i++,k--) {
        weights[i] = dist(i - 1,i);
        cost += k * weights[i];
    }
}

uint
MLSolution::costCalc()
{
    uint     ccost;
	uint     k;

    ccost = 0;
    k = clientCount - 1;

	for(uint i=1;i < clientCount;i++,k--)
        ccost += k * weights[i];

	return ccost;
}

uint
MLSolution::showCostCalc(const char *prompt)
{
    uint     ccost;
    uint     i,k;

    ccost = 0;
    k = clientCount - 1;

#if 1
    if(prompt)
        printf("%s",prompt);

    printf("<%hu,",clients[0]);
    for(i=1;i < clientCount;i++,k--) {
        ccost += k * weights[i];
        printf("%hu",clients[i]);
        if(i < clientCount - 1)
            printf(",");
    }
    printf("> = %d (calc=%d)\n",cost,ccost);
#else
    if(prompt)
        printf("%s",prompt);

    for(i=1;i < clientCount;i++,k--) {
        ccost += k * weights[i];
        printf("%2u-%2u: %3u x %3u = %5u -> %5u\n",
               clients[i - 1],clients[i],
               k,weights[i],
               k * weights[i],
               ccost);
    }
#endif

    return ccost;
}

void
MLSolution::assign(MLSolution *sol, bool eval)
{
	if(this == sol)
		return;

	l4printf("\tsolAssign(%p,%d)\tdataSize=%u\tadsDataSize=%u\tadsData=%p\n",
	                sol,eval,dataSize,adsDataSize,adsData);

	move = sol->move;
	time = sol->time;
	cost = sol->cost;
	clientCount = sol->clientCount;
	memcpy(clients,sol->clients,dataSize);

	if(eval && adsData && sol->adsData)
	    memcpy(adsData,sol->adsData,adsDataSize);
}

string
MLSolution::strCost()
{
    stringstream ss;

    if(cost != COST_INFTY)
        ss << cost;
    else
        ss << "oo";

    if(problem.params.costTour)
        ss << " (TOUR)";
    else
        ss << " (PATH)";

    return ss.str();
}

string
MLSolution::str(bool showCost)
{
    stringstream ss;

    ss << '<';
    for(uint i=0;i < clientCount;i++) {
        ss << clients[i];
        if(i < clientCount - 1)
            ss << ',';
    }
    ss << '>';

    if(showCost)
        ss << " = " << strCost();

    return ss.str();
}

void
MLSolution::adsAssign(const MLSolution &sol)
{
	if(sol.adsData) {
		if(adsData == NULL)
			adsAlloc();
		memcpy(adsData,sol.adsData,adsDataSize);
	}
}

void
MLSolution::adsAlloc()
{
	if(adsData)
	    gpuHostFree(adsData,adsFlags);

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
	 * info[0]  : size (ADS buffer size in bytes)
	 * info[1]  : rowElems
	 * info[2]  : solElems (sol size)
	 * info[3]  : solCost
	 * info[4]  : tour
	 * info[5]  : round
	 * info[6]  : reserved
	 * ...
	 * info[b-1]: reserved
	 */
    gpuHostMalloc(&adsData,adsDataSize,adsFlags);

    // Points to ADS time
    adsDataTime = puint(adsData + 1) + 2*adsRowElems;

#if LOG_LEVEL > 2
    lprintf("size\t\t: %u\n",size);
    lprintf("evalBuffer\t: %p\n",evalBuffer);
    lprintf("evalBufferSize\t: %u\n",evalBufferSize);
    lprintf("evalRowElems\t: %u\n",evalRowElems);
    lprintf("evalFlags\t: %x\n",evalFlags);
#endif
}

void
MLSolution::adsFree()
{
	if(adsData)
	    gpuHostFree(adsData,adsFlags);

	adsData = NULL;
	adsDataTime = NULL;
}

ullong
MLSolution::ldsChecksum(uint max)
{
    ullong  sum;
    byte   *data;

    if(max == 0)
        max = adsDataSize;

    data = (byte *) adsData;
    sum  = 0;
    for(uint i=0;i < max;i++)
        sum += data[i];
    return sum;
}

ullong
MLSolution::ldsUpdate()
{
    uint    *adsRowCost;
    uint    *adsCoords,
            *adsDataCost,
            *adsSolution;
    int      ic,jc,jp,
             i,j,n;
    ullong   time;


    if(adsData == NULL)
        adsAlloc();

    time = sysTimer();

    // Update ADS info
    adsData->s.size     = adsDataSize;
    adsData->s.rowElems = adsRowElems;
    adsData->s.solElems = problem.size;
    adsData->s.solCost  = cost;
    adsData->s.tour     = problem.params.costTour;
    adsData->s.round    = problem.params.distRound;

    adsCoords   = ADS_COORD_PTR(adsData);
    adsSolution = ADS_SOLUTION_PTR(adsData,adsRowElems);
    adsDataCost = ADS_COST_PTR(adsData,adsRowElems);

    n = (int) clientCount;
    adsCoords[0]   = GPU_USHORT2UINT(coordx(0),coordy(0));
    adsSolution[0] = clients[0];
    adsDataTime[0] = 0;

    // Calculate eval time (T)
    for(j=1;j < n;j++) {
        adsCoords[j]   = GPU_USHORT2UINT(coordx(j),coordy(j));
        adsSolution[j] = GPU_USHORT2UINT(clients[j],weights[j]);
        adsDataTime[j] = adsDataTime[j - 1] + weights[j];
    }

    // Calculate eval data for i <= j
    for(i=0;i < n;i++) {
        // Points to first element of line i
        adsRowCost = adsDataCost + adsRowElems*i;
        // C[i,i] = 0
        adsRowCost[i] = 0;

        for(j=i + 1;j < n;j++)
            adsRowCost[j] = adsRowCost[j - 1] + ldsTime(i,j);
    }

    // Calculate eval data for i > j
    for(i=n - 1;i >= 0;i--) {
        // Points to first element of line i
        adsRowCost = adsDataCost + adsRowElems*i;

        for(j=i - 1;j >= 0;j--)
            adsRowCost[j] = adsRowCost[j + 1] + ldsTime(j,i);
    }

//    printf("\tadsUpdate\t<");
//    for(i=0;i < n;i++)
//        printf(" %hu",GPU_HI_USHORT(adsSolution[i]));
//    printf(" > p=%p\n",adsSolution);

    return sysTimer() - time;
}

void
MLSolution::showWeight(ostream &os)
{
    uint i,j,n,w;

    n = problem.size;
    w = digits(problem.maxWeight) + 1;

    for(i=0;i < w + 1;i++)
        os << ' ';
    for(j=0;j < n;j++)
        os << setw(w) << j;
    os << '\n';
    for(i=0;i < w;i++)
        os << '-';
    os << '+';
    for(i=0;i < n*w;i++)
        os << '-';
    os << '\n';
    for(i=0;i < n;i++) {
        os << setw(w) << i << '|';
        for(j=0;j < n;j++)
            os << setw(w) << problem.clients[i].weight[j];
        os << '\n';
    }
    os << endl;
}

void
MLSolution::ldsShow(MLADSField field, ostream &os)
{
    const
	char        *label = "WTCD";
    uint        *adsRow,
                *data;
    uint 	     v,i,j,w,n;

    if(field == MLAF_COORD) {
        data = ADS_COORD_PTR(adsData);

        os << "coord(x,y)\t: ";
        for(i=0;i < clientCount;i++)
            os << '(' << GPU_HI_USHORT(data[i]) << ',' << GPU_LO_USHORT(data[i]) << ')';
        os << endl;
        return;
    }
    else if(field == MLAF_CLIENT) {
        data = ADS_SOLUTION_PTR(adsData,adsRowElems);

        os << "clnt(id,dis)\t: ";
        for(i=0;i < clientCount;i++)
            os << '(' << GPU_HI_USHORT(data[i]) << ',' << GPU_LO_USHORT(data[i]) << ')';
        os << endl;
        return;
    }

    n = clientCount - 1;
    w = 0;
    for(i=0;i < clientCount;i++) {
        for(j=0;j < clientCount;j++) {
            switch(field) {
            case MLAF_DIST:
                v = dist(i,j);
                break;
            case MLAF_WAIT:
                v  = (i > 0) && (i < n);
                v += i > j ? i - j : j - i;
                break;
            case MLAF_TIME:
                v = (i < j) ? ldsTime(i,j) : ldsTime(j,i);
                break;
            case MLAF_COST:
                adsRow = ADS_COORD_PTR(adsData) + adsRowElems*i;
                v = adsRow[j];
                break;
            default:
                v = COST_INFTY;
                break;
            }
            if((v != COST_INFTY) && (v > w))
                w = v;
        }
    }

    w = digits(w) + 1;
    if(w < 2)
        w = 2;

    for(j=0;j < w;j++)
    	os << (j == (w / 2) ? label[field] : ' ');
    os << '|';
    for(j=0;j < clientCount;j++)
        os << setw(w) << j;
    os << '\n';
    for(j=0;j < w;j++)
    	os << '-';
    os << '+';
    for(j=0;j < w*clientCount;j++)
        os << '-';
    os << '\n';
    for(i=0;i < clientCount;i++) {
        os << setw(w) << i << '|';
        for(j=0;j < clientCount;j++) {
            switch(field) {
            case MLAF_WAIT:
                v  = (i > 0) && (i < n);
                v += i > j ? i - j : j - i;
                break;
            case MLAF_TIME:
                v = (i < j) ? ldsTime(i,j) : ldsTime(j,i);
                break;
            case MLAF_COST:
                adsRow = ADS_COORD_PTR(adsData) + adsRowElems*i;
                v = adsRow[j];
                break;
            case MLAF_DIST:
                v = dist(i,j);
                break;
            default:
                v = COST_INFTY;
                break;
            }
            if(v == COST_INFTY)
                os << setw(w) << '-';
            else
                os << setw(w) << v;
        }
        os << '\n';
    }
    os << endl;
}

void
MLSolution::checkCost(const char *prompt)
{
	uint    ccost;
	uint    ids[clientCount];
	uint    i;
	bool    error;

	error = false;

	memset(ids,0,sizeof(ids));

	for(i=1;i < clientCount - 1;i++)
	    ids[ clients[i] ]++;

	if(clients[0] || (problem.params.costTour && clients[clientCount - 1]))
	    error = true;
	else {
        for(i=1;i < clientCount - 1;i++) {
            if(ids[i] != 1) {
                error = true;
                break;
            }
        }
	}

    if(error) {
        if(prompt)
            cout << prompt;
        show();
        EXCEPTION("INVALID SOLUTION: ");
    }

    ccost = costCalc();
    if(cost != ccost) {
        if(prompt)
            cout << prompt;
        show();
        EXCEPTION("INVALID COST: right=%u, wrong=%u",ccost,cost);
    }
}

bool
MLSolution::validate()
{
    uint i,j,id;
    bool found;

    for(i=0;i < clientCount - 1;i++) {
        id = problem.clients[i].id;
        found = false;
        for(j=0;j < clientCount;j++) {
            if(clients[j] == id) {
                found = true;
                break;
            }
        }
        if(!found)
            return false;
    }

    return true;
}

void
MLSolution::save(ostream &os)
{
    os << clientCount << '\n';
    os << cost << '\n';
    for(uint i=0;i < clientCount;i++) {
        os << clients[i];
        if(i < clientCount - 1)
            os << ' ';
    }
    os << '\n';
}

void
MLSolution::save(const char *fname)
{
    ofstream os(fname,ios_base::trunc);

    if(os.is_open())
        save(os);
    else
        EXCEPTION("Error saving solution to file: %s",fname);
}

void
MLSolution::load(istream &is)
{
    uint   v;

    clear();

    // Get problem size
    is >> v;

    if(v != problem.size) {
        bool error;

        if(problem.params.costTour)
            error = (problem.size - 1 != v);
        else
            error = (problem.size + 1 != v);

        if(error)
            EXCEPTION("Error loading solution: invalid problem size (loaded = %u, expected = %u)\n",
                        v,problem.size);
    }

    // Discard cost
    is >> v;

    for(uint i=0;i < problem.size;i++) {
        is >> v;
        add(ushort(v));
    }
    // Update weights and cost
    update();
}

void
MLSolution::load(const char *fname)
{
    ifstream is(fname);

    if(is.is_open())
        load(is);
    else
        EXCEPTION("Error loading solution from file: %s",fname);
}

void
MLSolution::random(MTRandom &rng, float alpha)
{
    PMLClientData  clData[problem.size],// buffer for candidates list
                  *clCands;             // candidates list (CL)
    MLClientData  *clLast,              // last client added to solution
                  *clCurr;              // client being added to solution
    uint           clCount,             // number of candidates at CL
                   rclCount,            // number of candidates at RCL
                   rclIndex;            // index of an RCL element
    uint           i,f;

    // copy client data references
    for(i=0;i < problem.size;i++)
        clData[i] = problem.clients + i;

    // set candidates list (CL)
    // all clients w/o first one (depot)
    clCands = clData + 1;
    // number of remaining candidates
    // if TOUR, ignore last one (depot again)
    clCount = problem.size - 1 - problem.params.costTour;

    // clear solution instance
    clear();

    // sol.weightShow(3);

    // add first client to solution
    clLast = clData[0];
    add(clLast->id);

    // reset solution cost. Will be computed during solution constructions.
    cost = 0;
    // solution cost calculation factor
    f = problem.size - 1;

    // construct the solution client by client
    while(clCount > 0) {

        if(clCount > 1) {
            // selects a random offset among alpha% elements of the RCL
            rclCount = (uint) ceil(alpha * clCount);
            rclIndex = (rclCount > 0) ? rng.rand(rclCount - 1) : 0;
            /*
             * For performance issues, CL is sorted in ascending or descending order
             * depending of the alpha value. This way is possible remove the selected
             * client from array with a minimum elements movement.
             *
             * - If alpha <  0.5, CL is sorted in descending order and a client is
             *   selected among last  elements of list.
             * - If alpha >= 0.5, CL is sorted in  ascending order and a client is
             *   selected among first elements of list.
             */
            if(alpha < 0.5) {
                // sort the candidates in descending order of their distances
                // to last client added
                sort(clCands,clCands + clCount,ClientCompDsc(clLast));
                // adjust index to get last elements of the RCL
                rclIndex = clCount - rclIndex - 1;
            }
            else {
                // sort the candidates in ascending order of their distances
                // to last client added
                sort(clCands,clCands + clCount,ClientCompAsc(clLast));
            }

#if LOG_LEVEL >= 4
            cout << "------" << endl;
            cout << "a=" << alpha << "\t|RCL|=" << rclCount << endl;
            cout << clLast->id << ": ";
            for(i=clCount;i > 0;i--)
                cout << '(' << clCands[i-1]->id << ',' << clLast->weight[clCands[i-1]->id] << ')' << ' ';
            cout << endl;
#endif
        }
        else {
            rclCount = 1;
            rclIndex = 0;
        }

#if LOG_LEVEL >= 4

        l1printf("Candidates <");
        for(i=0;i < clCount;i++)
            l1printf(" %u",clCands[i]->id);
        l1printf(" >\n");
#endif

        // choose the client indexed by 'rclIndex'
        clCurr = clCands[rclIndex];
        // add client to solution
        add(clCurr->id);
        // update solution cost
        cost += f * clLast->weight[clCurr->id];
        // prepare to next iteration
        clLast = clCurr;
        f--;

#if LOG_LEVEL >= 4
        sol.show();
#endif

        // remove client from CL
        for(i=rclIndex;i < clCount - 1;i++)
            clCands[i] = clCands[i + 1];
        clCount--;
    }

    // If TOUR, force depot as last client
    if(problem.params.costTour) {
        // add first client as last client of solution
        clCurr = clData[problem.size - 1];
        add(clCurr);
        // update solution cost if is a TOUR
        cost += clLast->weight[clCurr->id];
    }
    // Solution time
    time = sysTimer();

    update();
}

void
MLSolution::sample()
{
    int     route[] = { 0, 5, 1, 6, 3, 2, 4 },
            rsize = sizeof(route) / sizeof(*route);

    if(problem.size != 7 + problem.params.costTour)
        EXCEPTION("Invalid instance %s: trying to create SAMPLE instance: \n",problem.filename);

    clear();
    for(uint j=0;j < rsize;j++)
        add(route[j]);
    if(problem.params.costTour)
        add(route[0]);
    update();
}

void
MLSolution::sequence()
{
    uint    size;
    ushort  i;

    clear();
    size = problem.size - problem.params.costTour;
    for(i=0;i < size;i++)
        add(i);
    if(problem.params.costTour)
        add(ushort(0));
    update();
}
