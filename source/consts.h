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

#define MLP_MAX_NEIGHBOR            30
#define MLP_MAX_HISTORY             5
#define MLP_MAX_ALPHA               26
#define MLP_MAX_CPU                 ( 8U*uint(sizeof(uint)) )
#define MLP_MAX_GPU                 2U


// TODO: got from mlparams.h
#define OFM_LEN_APP             15    ///< Max length for app name
#define OFM_LEN_LIB             16    ///< Max length for lib version
#define OFM_LEN_BUILD           63    ///< Max length for build time
#define OFM_LEN_PATH            128   ///< Max length for paths
#define OFM_LEN_NAME            32    ///< Max length for names


#endif
