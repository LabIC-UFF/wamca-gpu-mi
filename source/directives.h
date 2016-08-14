/**
 * @file   directives.h
 *
 * @brief  Global compilation directives.
 *
 * @author Eyder Rios
 * @date   2015-06-18
 */


#ifndef __directives_h
#define __directives_h

/*!
 * Show detailed exception/warning messages
 */
#define EXCEPT_DETAIL

/*!
 * For debugging purpose only.
 * Save all generated solutions in GPU global memory
 */
// #define GPU_SAVE_SOLUTIONS

/*!
 * For debugging purpose only.
 * Always use same sequence of random number. Same numbers in google spreadsheet 'sample!RANDOM'
 */
// #define GPU_FIX_RAND

/*!
 * Eval data row size is multiple of GPU cache line size
 */
#define GPU_EVAL_ALIGNED

/*!
 * If defined, implements log cost engine
 */
//#define MLP_COST_LOG

/*!
 * If defined, set some additional information for profiling purposes.
 */
//#define GPU_PROFILE

/*!
 * Enable solution cost checking after movement
 */
#define MLP_COST_CHECK

/*!
 * If defined, CPU/GPU should use ADS, otherwise use calculate distances on the fly.
 */
//#define MLP_CPU_ADS
//#define MLP_GPU_ADS

/*
 * DO NOT ALTER CODE BELOW THIS LINE
 */
#ifdef  MLP_CPU_ADS
#define MLP_CPU_ADS_FLAG    true
#define MLP_CPU_ADS_STR     "CA"
#else
#define MLP_CPU_ADS_FLAG    false
#define MLP_CPU_ADS_STR     "CC"
#endif

#ifdef  MLP_GPU_ADS
#define MLP_GPU_ADS_FLAG    true
#define MLP_GPU_ADS_STR     "GA"
#else
#define MLP_GPU_ADS_FLAG    false
#define MLP_GPU_ADS_STR     "GC"
#endif

#endif
