/**
 * @file   mlads.cpp
 *
 * @brief  Auxiliary Data Structure (ADS)
 *
 * @author Eyder Rios
 * @date   2014-06-01
 */

#include "mlads.h"


// ################################################################################# //
// ##                                                                             ## //
// ##                              GLOBAL VARIABLES                               ## //
// ##                                                                             ## //
// ################################################################################# //

const
char    *nameArch[] = {
                "CPU",
                "SGPU",
                "MGPU",
};

const
char    *nameHeuristic[] = {
                "RVND",
                "DVND",
                "GDVND",
};

const
char    *nameLongArch[] = {
                "CPU",
                "Single GPU",
                "Multi GPU",
};

const
char    *nameMove[] = {
                "SWAP",
                "2OPT",
                "OROPT1",
                "OROPT2",
                "OROPT3",
};

const
char       *nameImprov[] = {
                "BI",
                "MI",
};

const
char       *nameLongImprov[] = {
                "Best Move",
                "Merge Move",
};

const
MLMove64    MOVE64_NONE  = { 0 },
            MOVE64_INFTY = { 0, 0, 0, COST_INFTY };

const
MLMove      MOVE_NONE  = { MLMoveId(0), 0, 0, 0 },
            MOVE_INFTY = { MLMoveId(0), 0, 0, COST_INFTY };

