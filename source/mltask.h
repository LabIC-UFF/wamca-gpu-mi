/**
 * @file	mltask.h
 *
 * @brief	Handle a CPU task.
 *
 * @author	Eyder Rios
 * @date    2015-05-28
 */

#ifndef __mltask_h
#define __mltask_h

#include <pthread.h>
#include <string.h>
#include "types.h"
#include "except.h"
#include "log.h"


// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //

#define RNG_RAND_MAX            0xffff

#define GPU_RMAKE(b,t,c)        ( (ullong(b) << 48) | (ullong(t) << 32) | ullong(c) )
#define GPU_RCOST(v)            ( uint(v) )
#define GPU_RBLOCK(v)           ( uint((v) >> 48) )
#define GPU_RTHREAD(v)          ( uint((v) >> 32) & 0x0000ffffU )

#define GPU_MAX_BLOCK           (1 << 16)
#define GPU_MAX_THREAD          (1 << 16)


// ################################################################################ //
// ##                                                                            ## //
// ##                                  DATA TYPES                                ## //
// ##                                                                            ## //
// ################################################################################ //

/*!
 * Incomplete classes
 */
class MLSolver;
class MLProblem;
class MLParams;
class MLTask;

/*!
 * MLTaskType
 */
enum MLTaskType {
    TTYP_CPUTASK,                       ///< CPU task
    TTYP_GPUTASK,                       ///< GPU task
};

/*!
 * MLTaskStatus
 */
enum MLTaskStatus {
    TSTA_STOPPED,
    TSTA_RUNNING,
};

/*!
 * MLSResult
 */
typedef ullong  MLResult;

/*!
 * MLSResult
 */
struct  MLSResult
{
    uint    cost   : 32;
    uint    thread : 16;
    uint    block  : 16;
};

/*!
 * MLUResult
 */
union MLUResult {
    MLResult    i;
    MLSResult   s;
};


/*!
 * PMLTask
 */
typedef MLTask      *PMLTask;


// ################################################################################ //
// ##                                                                            ## //
// ##                                 CLASS MLTask                               ## //
// ##                                                                            ## //
// ################################################################################ //

class MLTask
{
public:
    MLSolver    &solver;
    MLProblem   &problem;
    MLParams    &params;

    uint         taskId;                     ///< Thread id
    pthread_t    thread;                     ///< Thread handler

    MLTaskType   type;                       ///< Task type
    MLTaskStatus status;                     ///< Task status

    FILE        *logExec;                    ///< Execution log file (default is stdout)

    ullong       timeStart;                  ///< Time application started
    ullong       timeExec;                   ///< Application execution time

public:
    /*!
     * Create a OFTask instance.
     *
     * @param   solver  Main thread
     * @param   ttyp    Solver task type
     */
    MLTask(MLSolver &solver, MLTaskType ttyp, uint id);
    /*!
     * Destroy a OFTask instance.
     */
    virtual
    ~MLTask();
    /*!
     * Create log files.
     */
    void
    logCreate();
    /*!
     * Close log files.
     */
    void
    logClose();
    /*!
     * Write log header to file.
     */
    void
    logHeader();
    /*!
     * Write log footer to file.
     */
    void
    logFooter();
    /*!
     * Friend classes
     */
    friend class MLSolver;
};

#endif	// __task_h
