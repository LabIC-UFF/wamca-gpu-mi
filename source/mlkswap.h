/**
 * @file   mlkswap.h
 *
 * @brief  MLP Swap search in GPU.
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
// ##                            CLASS MLKernelSwap                              ## //
// ##                                                                            ## //
// ################################################################################ //

class MLKernelSwap : public MLKernel
{
public:
    /*!
     * Create a MLSwapKernelTask instance.
     */
    MLKernelSwap(MLGPUTask &parent) : MLKernel(parent,MLMI_SWAP) {
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
     * Apply movement to 'solBase' solution.
     */
    void
    applyMove(MLMove &move);
};
