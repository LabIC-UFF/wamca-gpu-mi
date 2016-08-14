/**
 * @file	task.cpp
 *
 * @brief    Handle a solver thread.
 *
 * @author	Eyder Rios
 * @date    2015-05-28
 */

#include <unistd.h>
#include <libgen.h>
#include <string.h>
#include "mltask.h"
#include "mlsolver.h"
//#include "cputask.h"
//#include "gputask.h"


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

// ################################################################################ //
// ##                                                                            ## //
// ##                                  FUNCTIONS                                 ## //
// ##                                                                            ## //
// ################################################################################ //

// ################################################################################ //
// ##                                                                            ## //
// ##                                 CLASS MLTask                               ## //
// ##                                                                            ## //
// ################################################################################ //

MLTask::MLTask(MLSolver &solv, MLTaskType ttyp, uint id) :
            solver(solv), problem(solver.problem), params(solver.params)
{
    taskId = id;
    thread = 0;
    type = ttyp;
    status = TSTA_STOPPED;

    timeStart = 0;
    timeExec = 0;

    logExec = NULL;
}

MLTask::~MLTask()
{
    if(logExec && (logExec != stdout))
        fclose(logExec);
}

void
MLTask::logCreate()
{
   const
   char *odir;
   char  path[256],
         ctype;

   if(params.logResultTask) {
       ctype = (type == TTYP_CPUTASK) ? 'C' : 'G';

       if((odir = params.outputDir) == NULL)
           odir = ".";

       sprintf(path,"%s/%s_%c%02u.log",odir,problem.name,ctype,taskId);

       printf("Task%u - Creating log file: %s\n",taskId,path);
       if((logExec = fopen(path,"wt")) == NULL)
           EXCEPTION("Error creating log file '%s'.\n",path);
   }
}

void
MLTask::logClose()
{
    if(logExec && (logExec != stdout))
        fclose(logExec);

    logExec = NULL;
}

void
MLTask::logHeader()
{
    time_t     now;
    struct tm  tm;
    char       buffer[64];

    if(logExec == NULL)
        return;

    // Log command line
    fprintf(logExec,"<<< COMMAND\n");

    for(int i=0;i < params.argc;i++)
        fprintf(logExec,"%s ",params.argv[i]);

    fprintf(logExec,"\n>>> COMMAND\n");

    // Log app parameters
    fprintf(logExec,"\n<<< HEADER\n");

    // Get build date/time
    strptime(__DATE__ " " __TIME__,"%b %2d %Y %H:%M:%S",&tm);
    strftime(buffer,sizeof(buffer),"%Y-%m-%d %H:%M:%S",&tm);
    fprintf(logExec,"DATE_BUILD\t: %s\n",buffer);

    // Get current date/time
    now = time(NULL);
    tm  = *localtime(&now);

    strftime(buffer,sizeof(buffer),"%Y-%m-%d %H:%M:%S",&tm);
    fprintf(logExec,"DATE_EXEC\t: %s\n",buffer);

    fprintf(logExec,"INSTANCE\t: %s\n",problem.filename);
    fprintf(logExec,"SIZE\t\t: %u\n",problem.size);
    fprintf(logExec,"TIME_LOAD\t: %llu ms\n",US2MS(problem.timeLoad));
    fprintf(logExec,"NODE_ID\t\t: %u\n",0);

    gethostname(buffer,sizeof(buffer));
    fprintf(logExec,"NODE_NAME\t: %s\n",buffer);

    fprintf(logExec,"NODE_QTY\t: 1\n");
    fprintf(logExec,"THREAD_ID\t: %u\n",taskId);
    //fprintf(logExec,"THREAD_QTY\t: %u\n",solver.taskCount);
    fprintf(logExec,"RAND_SEED\t: %u\n",params.rngSeed);
    fprintf(logExec,">>> HEADER\n\n");
}

void
MLTask::logFooter()
{
//    if(logExec == NULL)
//        return;
//
//    // Log solution
//    fprintf(logExec,"<<< SOLUTION\n");
//    //fprintf(logExec,"%u\n",solution->cost);
//    fprintf(logExec,">>> SOLUTION\n");
      fprintf(logExec,"-----------------------------------------------------\n");
}

#if 0
void
MLTask::launch(uint ident)
{
    pthread_attr_t attr;
    int            error;

    taskId = ident;

    // Initialize and set thread detached attribute
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    if((error = pthread_create(&thread,&attr,threadMain,(void *) this)) != 0)
        EXCEPTION("Error launching solver thread (error = %d)", error);

    pthread_attr_destroy(&attr);
}
/*!
 * Join thread.
 */
MLTask *
MLTask::join()
{
    int   error;
    void *status;


    if((error = pthread_join(thread, &status)) != 0)
        EXCEPTION("Error joining thread (error = %d): %s",
                  error,strerror(error));
    if(status == NULL)
        EXCEPTION("Invalid thread result");

    return (MLTask *) status;
}

void *
MLTask::threadMain(void *data)
{
    MLTask *task;

    // Get task
    task = PMLTask(data);

    try {
        // Change status to RUNNING
        task->status = TSTA_RUNNING;

        // Prepare task
        task->initTask();

        // Call main method
        task->main();

        // Finish task
        task->termTask();
    }
    catch(const Exception &e) {
        task->termTask();
        throw e;
    }
    catch(const SilentException &e) {
        task->termTask();
        throw e;
    }
    catch(const std::exception &e) {
        task->termTask();
        throw e;
    }
    catch(...) {
        task->termTask();
        EXCEPTION("Unknown exception.");
    }

    // Change status to STOPPED
    task->status = TSTA_STOPPED;

    return data;
}

void
MLTask::main()
{

}
#endif
