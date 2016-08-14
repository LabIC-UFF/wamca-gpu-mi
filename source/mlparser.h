/**
 * @file   mlparams.h
 *
 * @brief  Command line argument parser.
 *
 * @author  Eyder Rios
 * @date    2015-05-28
 */


#ifndef __mlparser_h
#define __mlparser_h


#include <argp.h>
#include "consts.h"
#include "mlparams.h"


// ################################################################################ //
// ##                                                                            ## //
// ##                              CONSTANTS & MACROS                            ## //
// ##                                                                            ## //
// ################################################################################ //

#define PARS_CHUNK               1          ///< Filename buffer chunk size

// Command line parser result flags
#define PARS_STATUS_OK           0          ///< No errors or warning in parsing
// Errors   --> application should abort execution
#define PARS_STATUS_NOMEM       -1          ///< Error allocating memory
#define PARS_STATUS_INVALID     -2          ///< Invalid argument
#define PARS_STATUS_ERROR       -3          ///< Error in parsing
#define PARS_STATUS_MANYNOPTS   -4          ///< too much non-options arguments
// Warnings --> application decides what to do
#define PARS_STATUS_HELP         1          ///< --help flag was informed
#define PARS_STATUS_USAGE        2          ///< --usage flag was informed
#define PARS_STATUS_VERSION      3          ///< --version flag was informed
#define PARS_STATUS_BASH         4          ///< --bash-completion flag was informed

// Macro expansions
#define STR(arg)        #arg
#define STRVAL(arg)     STR(arg)

/*!
 * Application name & usage
 */
#define APP_NAME        "gpfclp"
#define APP_VERNO       "1.0"
#define APP_VERSION     "Version\t" APP_VERNO "\nBuild\t" __DATE__ " " __TIME__
#define APP_EMAIL       "\b:\nEyder Rios\t<eyder.rios@gmail.com>\nIgor Machado\t<igor.machado@gmail.com>"
#define APP_DOC         "An heuristic for Minimum Latency Problem"

/*!
 * Prefix legend
 *  POL     Long name
 *  POK     Key name
 *  POP     If non-zero, this is the name of an argument associated with this option
 *  POO     Option flags
 *  POV     Default value
 *  POD     Documentation string
 */

/*!
 * Command line arguments
 */
#define POL_RSEED       "rand-seed"
#define POK_RSEED       's'
#define POP_RSEED       "SEED"
#define POO_RSEED       0
#define POV_RSEED       0
#define POD_RSEED       "Seed for pseudo-random numbers generator. If no seed or zero is " \
                        "supplied then a random seed is used by random numbers generator."

#define POL_ODIR        "output-dir"
#define POK_ODIR        'o'
#define POP_ODIR        "PATH"
#define POO_ODIR        0
#define POV_ODIR        "."
#define POD_ODIR        "Output directory where results will be saved."

#define POL_IDIR        "input-dir"
#define POK_IDIR        'i'
#define POP_IDIR        "PATH"
#define POO_IDIR        0
#define POV_IDIR        "."
#define POD_IDIR        "Input directory from where the problem instances should be loaded."

#define POL_NOLOG       "log-disabled"
#define POK_NOLOG       'q'
#define POP_NOLOG       NULL
#define POO_NOLOG       0
#define POV_NOLOG       false
#define POD_NOLOG       "Disable application log message."

#define POL_FLOG        "log-result"
#define POK_FLOG        'l'
#define POP_FLOG        NULL
#define POO_FLOG        0
#define POV_FLOG        true
#define POD_FLOG        "Log application result to file. The filename is composed by instance filename " \
                        "with extension '.log'"

#define POL_LOGTASK     "log-task"
#define POK_LOGTASK     -98000
#define POP_LOGTASK     NULL
#define POO_LOGTASK     0
#define POV_LOGTASK     false
#define POD_LOGTASK     "Log results for each launched task."

#define POL_LOGCOST     "log-cost"
#define POK_LOGCOST     -98010
#define POP_LOGCOST     "FILE"
#define POO_LOGCOST     0
#define POV_LOGCOST     ""
#define POD_LOGCOST     "Log solution cost to file FILE during the search. Use " \
                        "option --" POL_LOGPER " to adjust log."

#define POL_LOGPER      "log-period"
#define POK_LOGPER      -98020
#define POP_LOGPER      "N"
#define POO_LOGPER      0
#define POV_LOGPER      0
#define POD_LOGPER      "This option is used together with option --" POL_LOGCOST ". "\
                        "If N = 0 (default), the data is logged whenever solution cost is improved. " \
                        "If N > 0, data is logged every N milliseconds."

#define POL_PINNED       "mem-pinned"
#define POK_PINNED      -99010
#define POP_PINNED      NULL
#define POO_PINNED      0
#define POV_PINNED      true
#define POD_PINNED      "Makes all memory allocated in host be pinned (default). " \
                        "This increase performance of data transfer between CPU and GPU."

#define POL_PAGED       "mem-paged"
#define POK_PAGED       -99020
#define POP_PAGED       NULL
#define POO_PAGED       0
#define POV_PAGED       false
#define POD_PAGED       "Makes all memory allocated in host be paged (not pinned). " \
                        "This decrease performance of data transfer between CPU and GPU."

#define POL_BASH        "bash-completion"
#define POK_BASH        -99030
#define POP_BASH        NULL
#define POO_BASH        0
#define POV_BASH        false
#define POD_BASH        "Generate a bash-complete script for " APP_NAME " command line " \
                        "arguments"

#define POL_CHKCOST     "check-cost"
#define POK_CHKCOST     -99040
#define POP_CHKCOST     NULL
#define POO_CHKCOST     0
#define POV_CHKCOST     0
#define POD_CHKCOST     "Check cost for each solution generated."

#define POL_MAXEXEC     "max-exec"
#define POK_MAXEXEC     -99045
#define POP_MAXEXEC     "N"
#define POO_MAXEXEC     0
#define POV_MAXEXEC     0
#define POD_MAXEXEC     "Number of executions of the application."

#define POL_MAXCPU      "max-cpu"
#define POK_MAXCPU      -99050
#define POP_MAXCPU      "N"
#define POO_MAXCPU      0
#define POV_MAXCPU      0
#define POD_MAXCPU      "Limits the number of CPU cores to allocate for problem resolution. If N=0 (default), all cores available will be allocated."

#define POL_MAXGPU      "max-gpu"
#define POK_MAXGPU      -99060
#define POP_MAXGPU      "N"
#define POO_MAXGPU      0
#define POV_MAXGPU      0
#define POD_MAXGPU      "Limits the number of GPU devices to allocate for problem resolution. If N=0 (default), all devices available will be allocated."

#define POL_BLKSIZE     "max-block"
#define POK_BLKSIZE     -99070
#define POP_BLKSIZE     "N"
#define POO_BLKSIZE     0
#define POV_BLKSIZE     0
#define POD_BLKSIZE     "Compute grid block size. If N=0 (default), application calculates block size."

#define POL_DROUND      "dist-round"
#define POK_DROUND      -98075
#define POP_DROUND      NULL
#define POO_DROUND      0
#define POV_DROUND      0
#define POD_DROUND      "Indicates that float Euclidean distance calculated should be add to 0.5 before result truncation."

#define POL_CPATH       "cost-path"
#define POK_CPATH       -98070
#define POP_CPATH       NULL
#define POO_CPATH       0
#define POV_CPATH       0
#define POD_CPATH       "Solution cost should be calculated as PATH, without returning to depot."

#define POL_CTOUR       "cost-tour"
#define POK_CTOUR       -98080
#define POP_CTOUR       NULL
#define POO_CTOUR       0
#define POV_CTOUR       0
#define POD_CTOUR       "Solution cost should be calculated as PATH, returning to depot."

#define POL_MAXDIV       "max-diver"
#define POK_MAXDIV       -99090
#define POP_MAXDIV       "N"
#define POO_MAXDIV       0
#define POV_MAXDIV       10
#define POD_MAXDIV       "Number of diversification iterations."

#define POL_MAXINT       "max-inten"
#define POK_MAXINT       -99100
#define POP_MAXINT       "N"
#define POO_MAXINT       0
#define POV_MAXINT       100
#define POD_MAXINT       "Number of intensification iterations."

#define POL_MAXMERGE     "max-merge"
#define POK_MAXMERGE     -99110
#define POP_MAXMERGE     "N"
#define POO_MAXMERGE     0
#define POV_MAXMERGE     0
#define POD_MAXMERGE     "Maximum number of non-conflictants movements that can be merged " \
                         "(N = 0 means no limit, and it is default value)."

#define POL_EXPER        "exper"
#define POK_EXPER        -99120
#define POP_EXPER        "N"
#define POO_EXPER        0
#define POV_EXPER        0
#define POD_EXPER        "Experiment number to execute (N = 0, normal execution)"

#define POL_CSHIFT       "no-shift"
#define POK_CSHIFT       -99130
#define POP_CSHIFT       NULL
#define POO_CSHIFT       0
#define POV_CSHIFT       0
#define POD_CSHIFT       "Disable client coordinates shifting. Shifting ensures non " \
                         "negative coordinates."

#define POL_NSKRNL        "active-kernels"
#define POK_NSKRNL        -99140
#define POP_NSKRNL        "LIST"
#define POO_NSKRNL        0
#define POV_NSKRNL        0
#define POD_NSKRNL        "Active kernels per GPU. N is a bit mask (default: all bits set)"

#define POL_NSSRCH        "active-searches"
#define POK_NSSRCH        -99150
#define POP_NSSRCH        "LIST"
#define POO_NSSRCH        0
#define POV_NSSRCH        0
#define POD_NSSRCH        "Active searches per CPU. N=m0,m1,... where mX is a bit mask " \
                          "(default: all bits set)"

#define POL_NSTHRD        "ns-threshold"
#define POK_NSTHRD        -99160
#define POP_NSTHRD        "N"
#define POO_NSTHRD        0
#define POV_NSTHRD        0
#define POD_NSTHRD        "Neighborhood search threshold between multiple/best improvement"


// ################################################################################ //
// ##                                                                            ## //
// ##                            CLASS ParserOption                              ## //
// ##                                                                            ## //
// ################################################################################ //

struct ArgParserOption : public argp_option
{
    ArgParserOption(const char *name, int key, const char *arg, int flags,
                    const char *doc, int group) {
        this->name  = name;
        this->key   = key;
        this->arg   = arg;
        this->flags = flags;
        this->doc   = doc;
        this->group = group;
    }
};

// ################################################################################ //
// ##                                                                            ## //
// ##                              CLASS MLParser                                ## //
// ##                                                                            ## //
// ################################################################################ //

class MLParser
{
public:
    const
    char        **files;              ///< Points to a list of instance filenames
    uint          fileMax;            ///< Max number of instance filenames in 'files'
    uint          fileCount;          ///< Number of instance filenames in 'files'

    int           status;             ///< Parser status

    MLParams     &params;

public:
    MLParser(MLParams &pars) : params(pars) {
        files = NULL;
        fileMax = 0;
        fileCount = 0;
        status = PARS_STATUS_OK;
    }

    ~MLParser() {
        if(files)
            delete[] files;
    }
    bool
    addFile(const char *fname);

    void
    parse(int argc, char **argv);

/*
private:
    static
    error_t
    parser(int key, char *arg, struct argp_state *state);

    static
    char *
    filter(int key, const char *text, void *input);

    static
    error_t
    listValue(const char *opt, const char *arg, const char *list, int  &value);

    static
    void
    bashScript();

    static
    void
    getVersion(OFParams &params);

    static
    void
    writeVersion(FILE *fd, struct argp_state *state);
*/
};

#endif  // __mlparser_h
