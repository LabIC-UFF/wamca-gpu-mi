/**
 * @file   mlkoropt.h
 *
 * @brief  MLP OrOpt search in GPU.
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
// ##                           CLASS MLKernelOrOpt ## //
// ## ## //
// ################################################################################
// //

class MLKernelOrOpt : public MLKernel {
 public:
  /*!
   * Create a MLSwapKernelTask instance.
   */
  MLKernelOrOpt(MLProblem& _problem, uint tag, bool isTotal = false)
      : MLKernel(_problem, isTotal, MLMI_OROPT(tag), tag) {}
  /*!
   * Define kernel launching grid
   */
  void defineKernelGrid() override;
  /*!
   * Launch kernel for SWAP local search
   */
  void launchKernel() override;
  /*!
   * Apply movement to 'solBest' solution.
   */
  void applyMove(MLMove& move) override;
};
