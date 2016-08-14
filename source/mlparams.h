/**
 * @file   mlparams.h
 *
 * @brief   application parameters.
 *
 * @author  Eyder Rios
 * @date    2015-05-28
 */

#ifndef __mlparams_h
#define __mlparams_h

#include <string.h>
#include "consts.h"


// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //

// TODO: MOVED TO "consts.h"
#define OFM_LEN_APP             15    ///< Max length for app name
#define OFM_LEN_LIB             16    ///< Max length for lib version
#define OFM_LEN_BUILD           63    ///< Max length for build time
#define OFM_LEN_PATH            128   ///< Max length for paths
#define OFM_LEN_NAME            32    ///< Max length for names


// ################################################################################ //
// ##                                                                            ## //
// ##                                  DATA TYPES                                ## //
// ##                                                                            ## //
// ################################################################################ //

/*!
 * MLParams
 */
struct MLParams
{
public:
    const
    char        **argv;                 ///< Command line arguments
    int           argc;                 ///< Number of command line arguments

    char          appName[OFM_LEN_APP + 1];
    char          build[OFM_LEN_BUILD + 1];
    char          libGlib[OFM_LEN_LIB + 1];
    char          libPAPI[OFM_LEN_LIB + 1];
    char          libCudaDrv[OFM_LEN_LIB + 1];
    char          libCuda[OFM_LEN_LIB + 1];

    const
    char         *outputDir;            ///< output directory name
    const
    char         *inputDir;             ///< input directory name

    const
    char        **files;                ///< Instance files
    uint          fileCount;            ///< Number of instance files

    uint          rngSeed;              ///< random number generator seed

    uint          maxCPU;               ///< Max number of CPU cores to use (0 = all)
    uint          maxGPU;               ///< Max number of GPU cores to use (0 = all)
    uint          blockSize;            ///< Grid block size

    const
    char         *logCostFile;          ///< Log time and cost to file
    uint          logCostPeriod;        ///< Log time period in ms

    bool          distRound;            ///< Sum 0.5 to euclidean distance calculation?
    bool          coordShift;           ///< Shift clients coordinates if necessary
    bool          logDisabled;          ///< Do not log messages
    bool          logResult;            ///< Log program output to a file
    bool          logResultTask;        ///< Log task result to a file
    bool          pinnedAlloc;          ///< Pinned memory allocation?
    bool          checkCost;            ///< Check solutions cost
    bool          costTour;             ///< Cost calculation method (tour/path)

    bool          cpuAds;               ///< CPU uses ADS
    bool          gpuAds;               ///< GPU uses ADS

    int           allocFlags;           ///< Allocation flags

    uint          maxExec;              ///< Number of executions
    uint          maxDiver;             ///< Maximum diversification iterations
    uint          maxInten;             ///< Maximum intensification iterations
    uint          maxMerge;             ///< Maximum independent moves that can be merged
    int           experNo;              ///< Experiment number

    uint          nsSearch;             ///< Active searches (bits mask)
    uint          nsKernel[MLP_MAX_GPU];///< Active kernels  (bits mask)
    uint          nsThreshold;          ///< Threshold between MI/BI

public:

public:
    MLParams(int argc = 0, const char **argv = NULL) {

        memset(this,0,sizeof(*this));

        this->argc = argc;
        this->argv = argv;

        files = NULL;
        fileCount = 0;

        maxCPU = 0;
        maxCPU = 0;
        blockSize = 0;

        distRound = false;
        coordShift = true;

        pinnedAlloc = true;
        costTour = false;

        cpuAds = MLP_CPU_ADS_FLAG;
        gpuAds = MLP_GPU_ADS_FLAG;

        allocFlags = 0;

        maxExec  = 1;
        maxInten = 0;
        maxDiver = 0;
        maxMerge = 0;

        nsSearch = ~0U;
        memset(&nsKernel,0xff,sizeof(nsKernel));
        nsThreshold = 0;
    }
    ~MLParams() {
        if(files)
            delete[] files;
    }
};

#endif
