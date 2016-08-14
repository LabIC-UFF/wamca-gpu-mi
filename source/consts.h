/**
 * @file   consts.h
 *
 * @brief  System constants
 *
 * @author Eyder Rios
 * @date   2016-01-22
 */

#ifndef __consts_h
#define __consts_h

// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //

#define MLP_MAX_NEIGHBOR            5
#define MLP_MAX_HISTORY             5
#define MLP_MAX_ALPHA               26
#define MLP_MAX_CPU                 ( 8U*uint(sizeof(uint)) )
#define MLP_MAX_GPU                 2U

#endif
