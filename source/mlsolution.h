/**
 * @file   mlsolution.h
 *
 * @brief  Minimum Latency Problem solution.
 *
 * @author Eyder Rios
 * @date   2014-06-01
 */

#ifndef __mlsolution_h
#define __mlsolution_h

#include <iostream>
#include <string>
#include <math.h>
#include "types.h"
#include "except.h"
#include "gpu.h"
#include "mtrand.h"
#include "mlproblem.h"
#include "mlads.h"


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
 * MLRouteType
 */
enum MLRouteType {
        MLRT_PATH,      ///< PATH: don't returns to depot after visit last client
        MLRT_TOUR,      ///< TOUR: returns to depot after visit last client
};

/*!
 * MLADSField
 */
enum MLADSField {
    MLAF_COORD,
    MLAF_CLIENT,
    MLAF_DIST,
    MLAF_WAIT,
    MLAF_TIME,
    MLAF_COST,
};

// ################################################################################ //
// ##                                                                            ## //
// ##                              CLASS MLSolution                              ## //
// ##                                                                            ## //
// ################################################################################ //

class MLSolution
{
private:
	static
	uint			  idGen;					///< Solution ID generator
	static
	const char       *costTypeNames[];          ///< Cost type name

public:
    MLProblem        &problem;                  ///< Optimization problem
    uint              id;                       ///< Solution ID (unique if different of zero)
    uint              size;                     ///< Solution size

    uint              cost;                     ///< Solution cost
    ullong            time;                     ///< Solution creating time

    uint              clientCount;              ///< Number of clients in solution
    ushort           *clients;                  ///< Solution clients
    ushort           *weights;                  ///< Weight between solution clients
    uint              dataSize;                 ///< Solution data size in bytes
    MLMove            move;                     ///< Optional movement related to solution

    MLADSData        *adsData;                  ///< Evaluation data
    uint              adsDataSize;              ///< Evaluation data size in bytes
    uint             *adsDataTime;              ///< Evaluation data (T)
    uint              adsRowElems;              ///< Evaluation data row size in elements
    int               adsFlags;                 ///< Flags for cuda memory allocation
//    uint             *evalInfo;                 ///< ADS info
//    uint             *evalCoords;               ///< Coordinates by client index
//    uint             *evalSolution;             ///< Solution
//    uint             *evalDataCost;             ///< Evaluation data (C)

protected:
    /*!
     * Assign a new solution to this instance.
     */
    void
    adsAssign(const MLSolution &sol);
    /*!
     * Release memory allocated to cost evaluation data.
     */
    void
    adsFree();

private:
    /*!
     * Initialize object.
     */
    void
    init();
    /*
     * Square of a value.
     */
    inline
    float
    sqr(int x) {
        return float(x * x);
    }

public:
    /*!
     * Create a MLP solution instance.
     */
    MLSolution(MLProblem &prob, int flags = cudaHostAllocDefault);
    /*!
     * Destroy a MLP solution instance.
     */
    virtual
    ~MLSolution();
    /*!
     * Release problem instance.
     */
    void
    free();
    /*!
     * Check if solution is empty.
     *
     * Empty solutions has cost equal to \a COST_INFY.
     *
     * @return	Returns \a true if solution is empty, otherwise \a false.
     */
    inline
    bool
    empty() {
    	return (cost == COST_INFTY);
    }
    /*!
     * Get cost type name.
     *
     * @return  Returns a string to cost type name.
     */
    inline
    const char *
    costTypeName();
    /*!
     * Calculate solution cost based on clients route. The current solution's cost is not
     * Updated.
     *
     * @return	Returns the calculated cost.
     */
    uint
    costCalc();
    /*!
     * Update solution weights.
     */
    void
    update();
    /*!
     * Get X coordinate for a client.
     */
    inline
    int
    coordx(uint i) {
        return problem.clients[ clients[i] ].x;
    }
    /*!
     * Get Y coordinate for a client.
     */
    inline
    int
    coordy(uint i) {
        return problem.clients[ clients[i] ].y;
    }
    /*!
     * Get distance between two solution clients.
     */
    inline
    int
    dist(uint i, uint j) {
        return problem.clients[ clients[i] ].weight[ clients[j] ];
    }
    /*!
     * Get distance between two solution clients.
     */
    inline
    int
    calcDist(uint i, uint j) {
        return sqrtf( sqr(problem.clients[ clients[i] ].x - problem.clients[ clients[j] ].x) +
                      sqr(problem.clients[ clients[i] ].y - problem.clients[ clients[j] ].y) );
    }
    /*!
     * Clear solution (empty solution)
     */
    void
    clear() {
    	cost = COST_INFTY;
        clientCount = 0;
    	time = 0;
        move = MOVE_INFTY;
    }
    /*!
     * Performs memory allocation for cost evaluation data.
     */
    void
    adsAlloc();
    /*!
     * Add a client to solution.
     *
     * The client is added after the last client at solution. If there is no room,
     * nothing is done.
     *
     * @param	cdata		Pointer to client data
     *
     * @return	Returns \a true if client was added successfully, otherwise \a false.
     */
    inline
    bool
    add(MLClientData *cdata) {

#if LOG_LEVEL > 0
        if(clientCount >= size) {
        	WARNING("No more room for clients in solution.");
        	return false;
        }
#endif
        clients[clientCount++] = cdata->id;

      	return true;
    }
    /*!
     * Add a client to solution informing the client id.
     *
     * The client is added after the last client at solution. If there is no room,
     * nothing is done.
     *
     * @param	id		Client id
     *
     * @return	Returns \a true if client was added successfully, otherwise \a false.
     */
    inline
    bool
    add(ushort id) {
#if LOG_LEVEL > 0
        if(clientCount >= size)
        	EXCEPTION("No more room for clients in solution.");
#endif
        clients[clientCount++] = id;

      	return true;
    }
    /*!
     * Assign a solution.
     *
     * @param   sol     Solution to be assigned
     * @param   eval    Assign also evalData
     */
    /*!
     * Assign a solution.
     */
    void
    assign(MLSolution *sol, bool eval);
    /*!
     * Assign a solution.
     *
     * @param   sol     Solution to be assigned
     * @param   eval    Assign also evalData
     */
    /*!
     * Assign a solution.
     */
    void
    assign(MLSolution &sol, bool eval) {
        assign(&sol,eval);
    }
    /*!
     * Get solution as string.
     *
     * @return  Returns a string containing route.
     */
    /*!
     * Solution cost as string.
     */
    virtual
    std::string
    strCost();
    /*!
     * Solution as string.
     */
    virtual
    std::string
    str(bool showCost = true);
    /*!
     * Write solution to a stream.
     */
    void
    show(std::ostream &os = std::cout) {
        os << str() << std::endl;
    }
    /*!
     * Write solution to a stream.
     */
    void
    show(const char *prompt,std::ostream &os = std::cout) {
    	os << prompt;
    	show(os);
    }
    /*!
     * Calculate solution cost based on clients route showing result.
     * The current solution's cost is not Updated.
     *
     * @return  Returns the calculated cost.
     */
    uint
    showCostCalc(const char *prompt = NULL);
    /*!
     * Updates the auxiliary data structures of the solution.
     *
     * @return  Return the time to compute evaluation data (in milliseconds - 10^-6 s)
     */
    ullong
    ldsUpdate();
    /*!
     * Computes the sum of first \a max words in ADS. For debbuging purpose only.
     * If \a max = 0, the all words are added.
     *
     * @return  Return the check sum.
     */
    ullong
    ldsChecksum(uint max = 0);
    /*!
     * Show on the screen the cost evaluation matrix.
     *
     * @param   field   Cost evaluation field to show (0=W, 1=T, 2=C).
     * @param   os      Output stream
     */
    void
    ldsShow(MLADSField field, std::ostream &os = std::cout);
    /*!
     * Get eval data travel time between two clients.
     */
    inline
    uint
    ldsTime(uint i, uint j) {
        return adsDataTime[j] - adsDataTime[i];
    }
    /*!
     * Check if a solution is valid.

     * @returns Returns \a true if solution is valid, otherwise \a false.
     */
    bool
    validate();
    /*!
     * Recalculate solution cost and compare it to \a cost.
     *
     * @param	prompt	An optional string to be written before data.
     */
    void
    checkCost(const char *prompt = NULL);
    /*!
     * Show the weight matrix related to a solution.
     *
     * @param	os		Output stream
     */
    void
    showWeight(std::ostream &os = std::cout);
    /*!
     * Write solution to a stream.
     */
    void
    save(std::ostream &os);
    /*!
     * Save solution to a file.
     */
    void
    save(const char *fname);
    /*!
     * Load solution from a stream.
     */
    void
    load(std::istream &is);
    /*!
     * Load solution to a file.
     */
    void
    load(const char *fname);
    /*!
     * Generate a random solution
     */
    void
    random(MTRandom &rng, float alpha = 1.0F);
    /*!
     * Create a sequential initial solution.
     */
    void
    sequence();
    /*!
     * Create a sample initial solution.
     */
    void
    sample();
    /*!
     * Comparison operator
     */
    inline
    bool
    operator<(MLSolution &sol) {
        return cost < sol.cost;
    }
    /*!
     * Friend classes
     */
    friend class MLProblem;
    //friend class MLGPUTask;
};

// std::ostream &operator<<(std::ostream &os, MLSolution const &s);

#endif
