/**
 * @file   mlk2opt.h
 *
 * @brief  MLP 2Opt search in GPU.
 *
 * @author Eyder Rios
 * @date   2014-06-01
 */

#include "mlkernel.h"
#include "mlsolution.h"


// ################################################################################# //
// ##                                                                             ## //
// ##                              CONSTANTS & MACROS                             ## //
// ##                                                                             ## //
// ################################################################################# //

// ################################################################################ //
// ##                                                                            ## //
// ##                            CLASS MLKernel2Opt                              ## //
// ##                                                                            ## //
// ################################################################################ //

class MLKernel2Opt : public MLKernel
{
public:
    /*!
     * Create a MLSwapKernelTask instance.
     */
    MLKernel2Opt(MLProblem &_problem, bool isTotal = false) : MLKernel(_problem, isTotal, MLMI_2OPT) {
    }
    /*!
     * Define kernel launching grid
     */
    void
    defineKernelGrid();
    /*!
     * Launch kernel for SWAP local search
     */
    void
    launchKernel();
    /*!
     * Apply movement to 'solBest' solution.
     */
    void
    applyMove(MLMove &move);
};
