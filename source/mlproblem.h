/**
 * @file	mlproblem.h
 *
 * @brief	Handle a general optimization problem
 *
 * @author	Eyder Rios
 * @date    2015-05-28
 */


#include <limits.h>
#include "types.h"

//#include "mlparams.h"
#include "consts.h"


#ifndef __mlproblem_h
#define __mlproblem_h

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

/*
 * Classes
 */
class MLSolution;

/*!
 * MLClientData
 */
struct MLClientData {
    ushort      id;                     ///< client id
    int         x;                      ///< client coordinate X
    int         y;                      ///< client coordinate Y
    int        *weight;                 ///< arc weights from client \id to another ones
};

typedef MLClientData  *PMLClientData;


// ################################################################################ //
// ##                                                                            ## //
// ##                               CLASS MLProblem                              ## //
// ##                                                                            ## //
// ################################################################################ //

/*!
 * Class MLProblem
 */
class MLProblem
{
public:
    //MLParams       &params;                 ///< MLP parameters
	bool          	costTour;             ///< Cost calculation method (tour/path)
	bool          	distRound;            ///< Sum 0.5 to euclidean distance calculation?
	bool          	coordShift;           ///< Shift clients coordinates if necessary

    char            filename[OFM_LEN_PATH]; ///< Instance filename
    char            name[OFM_LEN_NAME];     ///< Instance filename

    ullong          timeLoad;               ///< Instance load time
    uint            maxWeight;              ///< Max weight value

    MLClientData   *clients;                ///< Client data
    uint            size;                   ///< Problem size

    int             shiftX;                 ///< X coordinate shift, if any
    int             shiftY;                 ///< Y coordinate shift, if any

protected:
    /*!
     * Allocate memory for clients data.
     *
     * @param   size    Instance size (number of clients)
     */
    void
    allocData(uint size);

public:
    /*!
     * Create an empty OFProblem instance.
     */
    MLProblem(bool _costTour, bool _distRound, bool _coordShift):
    	costTour(_costTour), distRound(_distRound), coordShift(_coordShift)  {
    	//(MLParams &pars) : params(pars) {

        clients = NULL;
        size = 0;
        timeLoad = 0;
        maxWeight = 0;
        shiftX = 0;
        shiftY = 0;
    }
    /*!
     * Destroy OFProblem instance.
     */
    virtual
   ~MLProblem() {
        free();
    }
    /*!
     * Was coordinates shifted?
     */
    bool
    coordShifted() {
        return shiftX || shiftY;
    }
    /*!
     * Release problem instance.
     */
    void
    free();
    /*!
     * Load from file a PFCL problem instance.
     *
     * @param   fname   Instance filename
     */
    void
    load(const char *fname);
    /*!
     * Save PFCL to a file. If \a fname is NULL, then the
     * loaded instance filename is used replacing the extension
     * by 'dat'
     *
     * @param   fname   Instance filename
     */
    void
    save(const char *fname = NULL);
    /*!
     * Create a new instance of a problem solution.
     *
     * @return  Returns a pointer to a new problem solution.
     */
    MLSolution *
    createSolution();
    /*!
     * Friend classes
     */
    friend class MLSolution;
};

#endif	// __mlproblem_h
