/**
 * @file	main.cpp
 *
 * @brief	Application entry point.
 *
 * @author	Eyder Rios
 * @date	2015-05-28
 */

#include <unistd.h>
#include <papi/papi.h>
#include <iostream>
#include "except.h"
#include "log.h"
#include "gpu.h"
#include "mlparser.h"
#include "mlproblem.h"
#include "mlsolution.h"
#include "mlsolver.h"


using namespace std;

/*!
 * Initialize environment.
 */
void
envInit()
{
    int     error;

    // Initializes log system
    logInit();

    // Get currenct directory
//    char    cwd[128];
//    if(getcwd(cwd,sizeof(cwd)) != NULL)
//        lprintf("Current working dir: %s\n", cwd);

    // Initialize PAPI library
    if((error = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
        EXCEPTION("Error initializing PAPI library: %s", PAPI_strerror(error));

#ifndef GPU_CUDA_DISABLED

    cudaDeviceProp  prop;
    int             count;

    // Set thread GPU
    cudaSetDevice(0);

    // Detect CUDA driver and GPU devices
    switch(cudaGetDeviceCount(&count)) {
    case cudaSuccess:
        for(int d;d < count;d++) {
            if(cudaGetDeviceProperties(&prop,d) == cudaSuccess) {
                if(prop.major < 2)
                    WARNING("Device '%s' is not suitable to this application. Device capability %d.%d < 2.0\n",
                            prop.name,prop.major,prop.minor);
            }
        }
        break;
    case cudaErrorNoDevice:
        WARNING("No GPU Devices detected.");
        break;
    case cudaErrorInsufficientDriver:
        WARNING("No CUDA driver installed.");
        break;
    default:
        EXCEPTION("Unknown error detecting GPU devices.");
    }

#endif
}

/*!
 * Finalize environment.
 */
void
envTerm()
{
}

/*!
 * Parse command line options
 *
 * @param   params  Application parameters
 * @param   argc    Command line number of parameters
 * @param   argv    Command line parameters
 */
void
parseParams(MLParams &params, int argc, char **argv)
{
    MLParser    parser(params);

    // Parse command line arguments
    parser.parse(argc,argv);

    switch(parser.status) {
    case PARS_STATUS_OK:            ///< No errors or warning in parsing
        if(params.outputDir && (access(params.outputDir,F_OK) != 0))
            EXCEPTION("Output directory doesn't exists: %s",params.outputDir);
        break;
    case PARS_STATUS_NOMEM:         ///< Error allocating memory
        EXCEPTION("Command line parser error: memory allocation error");
        break;
    case PARS_STATUS_INVALID:       ///< Invalid argument
        SILENT_EXCEPTION;
        break;
    case PARS_STATUS_ERROR:         ///< Error in parsing
        EXCEPTION("Command line parser error: unknown error");
        break;
    case PARS_STATUS_MANYNOPTS:     ///< too much non-options arguments
        EXCEPTION("To many instance files informed (only one is allowed)");
        break;
    case PARS_STATUS_HELP:          ///< --help flag was informed
        SILENT_EXCEPTION;
        break;
    case PARS_STATUS_USAGE:         ///< --usage flag was informed
        SILENT_EXCEPTION;
        break;
    case PARS_STATUS_VERSION:       ///< --version flag was informed
        SILENT_EXCEPTION;
        break;
    case PARS_STATUS_BASH:          ///< --bash-completion flag was informed
        SILENT_EXCEPTION;
        break;
    default:
        EXCEPTION("Command line parser error: unknown parser status (%d)",parser.status);
        break;
    }

    if(parser.fileCount == 0)
        EXCEPTION("Nothing to do: no instance file informed");

    if(params.logDisabled)
        params.logResultTask = false;
}

void
showCommandLine(MLParams &params)
{
    for(int i=0;i < params.argc;i++)
        printf("%s ",params.argv[i]);
}

/*!
 * Solve PFCL problem instances.
 *
 * @param   params  Application parameters
 */
void
launchApp(MLParams &params)
{
    MLProblem   problem(params);
    MLSolver    solver(problem);

    // Load problem instance
    problem.load(params.files[0]);
    // Solve problem
    solver.solve();
}

/*!
 * Test some new feature on application.
 *
 * @param   params  Application parameters
 */
void
testFeature(MLParams &params)
{
    MLProblem   problem(params);
    int         route[] = { 0, 5, 1, 6, 3, 2, 4, 0 };

    for(uint i=0;i < params.fileCount;i++) {
        // Load problem instance
        problem.load(params.files[i]);

        MLSolution  sol(problem);

        for(uint j=0;j < problem.size;j++)
            sol.add(route[j]);
        sol.update();
        sol.show();

        sol.ldsUpdate();
        sol.ldsShow(MLAF_TIME);
        sol.ldsShow(MLAF_COST);
    }
}

void
loadInstance(MLParams &params)
{
    MLProblem   problem(params);

    l4printf("files = %u\n",params.fileCount);
    for(uint i=0;i < params.fileCount;i++) {
        // Load problem instance
        problem.load(params.files[i]);
    }
}

/*!
 * The main function.
 *
 * @param   argc  Command line number of parameters
 * @param   argv  Command line parameters
 *
 * @return  Returns 0 after a successful execution, otherwise a negative value
 */
int
main(int argc, char **argv)
{
    MLParams    params;

    try {
        // Initialize environment
        envInit();

        // Parse command line parameters
        parseParams(params,argc,argv);

        // Update memory allocation flags
        params.allocFlags = params.pinnedAlloc ? cudaHostAllocDefault : cudaHostAllocPaged;

        // Accept only one instance file per execution
        if(params.fileCount > 1)
            EXCEPTION("Only one instance file per execution is permitted (%u informed)",params.fileCount);

#if 1
        // Launch application
        launchApp(params);
#else
        // Test features implementations
        // testFeature(params);
        // Load/save instances
        loadInstance(params);
#endif

        // Finalize environment
        envTerm();
    }
    catch(const Exception &e) {
        cerr << e.what();
    }
    catch(const SilentException &) {
    }
    catch(const exception &e) {
        cerr << e.what();
    }
    catch(...) {
        WARNING("Unknown exception.");
    }

    l4printf(">>>> BUILT AT %s %s\n",__DATE__,__TIME__);

    return 0;
}
