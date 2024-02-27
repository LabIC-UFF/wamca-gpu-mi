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

// #################################################################################
// //
// ## ## //
// ##                              CONSTANTS & MACROS ## //
// ## ## //
// #################################################################################
// //

// ################################################################################
// //
// ## ## //
// ##                            CLASS MLKernelSwap ## //
// ## ## //
// ################################################################################
// //

class MLKernelSwap : public MLKernel {
 public:
  /*!
   * Create a MLSwapKernelTask instance.
   */
  explicit MLKernelSwap(MLProblem& _problem, bool isTotal = false)
      : MLKernel(_problem, isTotal, MLMI_SWAP) {}
  /*!
   * Define kernel launching grid
   */
  void defineKernelGrid() override;
  /*!
   * Launch kernel for SWAP local search
   */
  void launchKernel() override;
  /*!
   * Apply movement to 'solBase' solution.
   */
  void applyMove(MLMove& move) override;
};
