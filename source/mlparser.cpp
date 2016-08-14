/**
 * @file   mlparser.cpp
 *
 * @brief  Command line arguments
 *
 * @author Eyder Rios
 * @date   2011-09-12
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <papi/papi.h>
#include "except.h"
#include "utils.h"
#include "log.h"
#include "gpu.h"
#include "mlparser.h"


using namespace std;


// ################################################################################ //
// ##                                                                            ## //
// ##                              CONSTANTS & MACROS                            ## //
// ##                                                                            ## //
// ################################################################################ //

#if LOG_LEVEL > 2
    #define ARGSHOW(k,a)  { std::cout << "key = " << #k; \
                            if(a) std::cout << "\targ = '" << (char *) a << "'"; \
                            std::cout << endl; }
#else
    #define ARGSHOW(k,a)
#endif

// ################################################################################ //
// ##                                                                            ## //
// ##                                 DATA TYPES                                 ## //
// ##                                                                            ## //
// ################################################################################ //


// ################################################################################ //
// ##                                                                            ## //
// ##                              GLOBAL VARIABLES                              ## //
// ##                                                                            ## //
// ################################################################################ //

/*
 * FIELD1: char *name
 *          The long option name.  For more than one name for the same option, you can use following options with the
 *          OPTION_ALIAS flag set.
 *
 * FIELD2: int key
 *          What key is returned for this option.  If > 0 and printable, then it's also accepted as a short option.
 *
 * FIELD3: char *arg
 *          If non-NULL, this is the name of the argument associated with this option, which is required unless the
 *          OPTION_ARG_OPTIONAL flag is set.
 *
 * FIELD4:  int flags
 *          Option flags.
 *
 * FIELD5: char *doc
 *          The doc string for this option.  If both NAME and KEY are 0, This string will be printed outdented from
 *          the normal option column, making it useful as a group header (it will be the first thing printed in its
 *          group); in this usage, it's conventional to end the string with a `:'.
 *
 * FIELD6: int grpoup
 *          The group this option is in.  In a long help message, options are sorted alphabetically within each group,
 *          and the groups presented in the order 0, 1, 2, ..., n, -m, ..., -2, -1.  Every entry in an options array with
 *          if this field 0 will inherit the group number of the previous entry, or zero if it's the first one, unless its
 *          a group header (NAME and KEY both 0), in which case, the previous entry + 1 is the default.  Automagic
 *          options such as --help are put into group -1.
 */

// Command line options expansion macros
#define ARG_OPT(tag,filter) { POL_##tag, POK_##tag, POP_##tag, POO_##tag, POD_##tag, filter },
#define ARG_TERM            { NULL, 0, NULL, 0, NULL, 0 }

// Command line options
static
argp_option parserOptions[] = {
        ARG_OPT(BASH,    0)
        ARG_OPT(RSEED,   0)
        ARG_OPT(ODIR,    0)
        ARG_OPT(IDIR,    0)
        ARG_OPT(NOLOG,   0)
        ARG_OPT(LOGTASK, 0)
        ARG_OPT(FLOG,    0)
        ARG_OPT(DROUND,  0)
        ARG_OPT(PINNED,  0)
        ARG_OPT(PAGED,   0)
        ARG_OPT(MAXEXEC, 0)
        ARG_OPT(MAXCPU,  0)
        ARG_OPT(MAXGPU,  0)
        ARG_OPT(BLKSIZE, 0)
        ARG_OPT(CHKCOST, 0)
        ARG_OPT(CPATH,   0)
        ARG_OPT(CTOUR,   0)
        ARG_OPT(MAXDIV,  0)
        ARG_OPT(MAXINT,  0)
        ARG_OPT(MAXMERGE,0)
        ARG_OPT(EXPER,   0)
        ARG_OPT(CSHIFT,  0)
        ARG_OPT(NSKRNL,  0)
        ARG_OPT(NSSRCH,  0)
        ARG_OPT(NSTHRD,  0)

#ifdef MLP_COST_LOG
        ARG_OPT(LOGCOST ,0)
        ARG_OPT(LOGPER,  0)
#endif

        ARG_TERM
};

static
bool
argVersionCalled = false;


// ################################################################################ //
// ##                                                                            ## //
// ##                              PARSER FUNCTIONS                              ## //
// ##                                                                            ## //
// ################################################################################ //

#define snprintfz(s,f,...)			snprintf(s,sizeof(s)-1,f,##__VA_ARGS__)

void
parserGetVersion(MLParams &params)
{
    struct tm   tm;
    char	    buffer[128];
    int         version;

    snprintfz(params.appName,"%s",APP_NAME);

    strptime(__DATE__ " " __TIME__,"%b %2d %Y %H:%M:%S",&tm);
    strftime(buffer,sizeof(buffer),"%Y-%m-%d %H:%M:%S",&tm);
    snprintfz(params.build,"%s",buffer);

    confstr(_CS_GNU_LIBC_VERSION,buffer,sizeof(buffer));
    snprintfz(params.libGlib,"%s",buffer + 6);

    snprintfz(params.libPAPI,"%d.%d.%d.%d",PAPI_VERSION_MAJOR(PAPI_VER_CURRENT),
            							   PAPI_VERSION_MINOR(PAPI_VER_CURRENT),
            							   PAPI_VERSION_REVISION(PAPI_VER_CURRENT),
            							   PAPI_VERSION_INCREMENT(PAPI_VER_CURRENT));

#ifndef GPU_CUDA_DISABLED
    gpuDriverGetVersion(&version);
    if(version)
        snprintfz(params.libCudaDrv,"%d.%d",version / 1000,(version % 100) / 10);

    gpuRuntimeGetVersion(&version);
    snprintfz(params.libCuda,"%d.%d",version / 1000,(version % 100) / 10);
#else
    strcpy(params.libCudaDrv,"N/A");
    strcpy(params.libCuda,"N/A");
#endif
}

void
parserWriteVersion(FILE *fd, struct argp_state *state)
{
    MLParams  params;

    parserGetVersion(params);

    cout << params.appName << '\t' << '\t' << APP_DOC   << endl;
    cout << "Version\t\t"  << APP_VERNO         << endl;
    cout << "Build\t\t"    << params.build      << endl;
    cout << "PAPI\t\t"     << params.libPAPI    << endl;

    cout << "Cuda Driver\t";
    if(*params.libCudaDrv)
        cout << params.libCudaDrv << endl;
    else
        cout << "No library detected" << endl;
    cout << "Cuda Runtime\t";
    if(*params.libCuda)
        cout << params.libCuda << endl;
    else
        cout << "No library detected" << endl;

    argVersionCalled = true;
}

void
parserBashScript()
{
/*
#
# gmlp bash-completion script
#
# Save this script in ~/.bash_completion.d/gmlp
# Load it in .bashrc
#

_gmlp()
{
	local opts="..."

	local prv cur
    _init_completion || return

	COMPREPLY=()

	cur=${COMP_WORDS[COMP_CWORD]}
	prv=${COMP_WORDS[COMP_CWORD-1]}

	if [[ "$cur" == -* ]]; then
		COMPREPLY=( $( compgen -W "$opts" -- "$cur" ) )
	else
		COMPREPLY=( $( compgen -o plusdirs -f -- "$cur" ) )
	fi

} &&
complete -F _gmlp gmlp

*/

    argp_option *arg;

    cout << "#\n# " << APP_NAME << " bash-completion script\n#\n";
    cout << "# Save this script in ~/.bash_completion.d/gmlp\n";
    cout << "# Load it in .bashrc\n#\n\n";
    cout << "_" << APP_NAME << "()\n{\n";
    cout << "\tlocal opts=\"";

    for(arg = parserOptions;arg->name;arg++) {
        cout << "--" << arg->name;
        if(arg->key > 0)
            cout << " -" << (char) arg->key;
        if(arg[1].name || (arg[1].key > 0))
            cout << ' ';
        else
            cout << '"';
    }

    cout << "\n\tlocal prv cur\n\n";
    cout << "\tCOMPREPLY=()\n\n";
    cout << "\tcur=${COMP_WORDS[COMP_CWORD]}\n";
    cout << "\tprv=${COMP_WORDS[COMP_CWORD-1]}\n\n";

    cout << "\tif [[ \"$cur\" == -* ]]; then\n";
    cout << "\t\tCOMPREPLY=( $( compgen -W \"$opts\" -- \"$cur\" ) )\n";
    cout << "\telse\n";
    cout << "\t\tCOMPREPLY=( $( compgen -o plusdirs -f -- \"$cur\" ) )\n\tfi\n\n";
    cout << "} &&\ncomplete -F _" << APP_NAME << ' ' << APP_NAME << "\n" << endl;
}

template<typename T>
error_t
parserArgValue(const char *opt, const char *arg,
         T &value,
         T  min = 0,
         T  max = (T) -1)
{
	istringstream iss(arg);
    int  		  error = 0;
    T			  result;

	iss >> result;
	if (!iss.fail()) {
		if ((value >= min) && (value <= max))
			value = result;
		else {
			EXCEPTION("Value for option '--%s' out of range: %s", opt, arg);
			error = -1;
		}
	}
	else {
		EXCEPTION("Invalid value for option '--%s': %s", opt, arg);
		error = -1;
	}

    return error;
}

error_t
parserListIndex(const char *opt, const char *arg, const char *list, int  &value)
{
    char   buffer[1024],
          *elem;
    uint   v;

    strcpy(buffer,list);

    if((elem = strtok(buffer,"|")) != NULL) {
        v = 0;
        while(elem != NULL) {
            if(!strcmp(elem,arg)) {
                value = v;
                l4printf("Option --%s = %s (%d)\n",opt,elem,value);
                break;
            }
            elem = strtok(NULL,"|");
            v++;
        };
    }

    if(elem == NULL)
        EXCEPTION("Invalid value for option '--%s': %s", opt, arg);

    return 0;
}

template<typename T>
error_t
parserListValues(const char *opt, const char *arg, T *list, uint  &count, uint max = 0)
{
    char   buffer[1024],
          *elem,
          *perr;
    long   v;

    if(max == 0)
        max = ~0U;

    strcpy(buffer,arg);

    count = 0;
    if((elem = strtok(buffer,",")) != NULL) {
        while((elem != NULL) && (count < max)) {
            v = strtol(elem,&perr,0);
            if(perr == NULL)
                EXCEPTION("Invalid value for option '--%s': %s", opt, arg);
            list[count++] = (T) v;
            elem = strtok(NULL,",");
        };
    }

    if(elem == NULL)

    return 0;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"

char *
parserFilter(int key, const char *text, void *input)
{
    char  *doc;

    doc = (char *) text;

    switch(key) {
    // The argument doc string; formally the args_doc field from the argp parser.
    case ARGP_KEY_HELP_ARGS_DOC:
        ((MLParser *) input)->status = PARS_STATUS_USAGE;
        ARGSHOW(ARGP_KEY_HELP_ARGS_DOC,NULL);
        //asprintf(&doc,"ARGP_KEY_HELP_ARGS_DOC\n");
        break;
    // The help text preceding options.
    case ARGP_KEY_HELP_PRE_DOC:
        ARGSHOW(ARGP_KEY_HELP_PRE_DOC,NULL);
        asprintf(&doc,"\n%s.",text);
        break;
    // The help text following options.
    case ARGP_KEY_HELP_POST_DOC:
        ((MLParser *) input)->status = PARS_STATUS_HELP;
        ARGSHOW(ARGP_KEY_HELP_POST_DOC,NULL);
        //asprintf(&doc,"ARGP_KEY_HELP_POST_DOC\n");
        break;
    // The option header string.
    case ARGP_KEY_HELP_HEADER:
        ARGSHOW(ARGP_KEY_HELP_HEADER,NULL);
        //asprintf(&doc,"ARGP_KEY_HELP_HEADER\n");
        break;
    // This is used after all other documentation; text is zero for this key.
    case ARGP_KEY_HELP_EXTRA:
        ARGSHOW(ARGP_KEY_HELP_EXTRA,NULL);
        //asprintf(&doc,"ARGP_KEY_HELP_EXTRA\n");
        break;
    // The explanatory note printed when duplicate option arguments have been suppressed.
    case ARGP_KEY_HELP_DUP_ARGS_NOTE:
        ARGSHOW(ARGP_KEY_HELP_DUP_ARGS_NOTE,NULL);
        //asprintf(&doc,"ARGP_KEY_HELP_DUP_ARGS_NOTE\n");
        break;
    }

    return doc;
}

#pragma GCC diagnostic pop

error_t
parserParser(int key, char *arg, struct argp_state *state)
{
    MLParser  *parser;
    error_t    result;
    uint       n;

    parser = (MLParser *) state->input;

    result = 0;
    switch(key) {
    /*
     * Application options.
     */
    case POK_BASH:
        parserBashScript();
        parser->status = PARS_STATUS_BASH;
        result = PARS_STATUS_BASH;
        break;
    case POK_ODIR:
        parser->params.outputDir = arg;
        break;
    case POK_IDIR:
        parser->params.inputDir = arg;
        break;
    case POK_LOGCOST:
        parser->params.logCostFile = arg;
        break;
    case POK_LOGPER:
        parserArgValue(POL_LOGPER,arg,parser->params.logCostPeriod);
        break;
    case POK_NOLOG:
        parser->params.logDisabled = true;
        break;
    case POK_LOGTASK:
        parser->params.logResultTask = true;
        break;
    case POK_FLOG:
        parser->params.logResult = true;
        break;
    case POK_CSHIFT:
        parser->params.coordShift = false;
        break;
    case POK_PINNED:
        parser->params.pinnedAlloc = true;
        break;
    case POK_PAGED:
        parser->params.pinnedAlloc = false;
        break;
    case POK_CHKCOST:
        parser->params.checkCost = true;
        break;
    case POK_DROUND:
        parser->params.distRound = true;
        break;
    case POK_CPATH:
        parser->params.costTour = false;
        break;
    case POK_CTOUR:
        parser->params.costTour = true;
        break;
    case POK_RSEED:
        result = parserArgValue(POL_RSEED,arg,parser->params.rngSeed);
        break;
    case POK_MAXMERGE:
        result = parserArgValue(POL_MAXMERGE,arg,parser->params.maxMerge,0U,uint(INT_MAX));
        break;
    case POK_MAXEXEC:
        result = parserArgValue(POL_MAXEXEC,arg,parser->params.maxExec,0U,uint(INT_MAX));
        break;
    case POK_MAXCPU:
        result = parserArgValue(POL_MAXCPU,arg,parser->params.maxCPU,0U,MLP_MAX_CPU);
        break;
    case POK_MAXGPU:
        result = parserArgValue(POL_MAXGPU,arg,parser->params.maxGPU,0U,MLP_MAX_GPU);
        break;
    case POK_BLKSIZE:
        result = parserArgValue(POL_BLKSIZE,arg,parser->params.blockSize);
        break;
    case POK_MAXDIV:
        result = parserArgValue(POL_MAXDIV,arg,parser->params.maxDiver);
        break;
    case POK_MAXINT:
        result = parserArgValue(POL_MAXINT,arg,parser->params.maxInten);
        break;
    case POK_EXPER:
        result = parserArgValue(POL_EXPER,arg,parser->params.experNo,0,99);
        break;
    case POK_NSSRCH:
        result = parserArgValue(POL_NSSRCH,arg,parser->params.nsSearch);//,0,uint((1 << MLP_MAX_NEIGHBOR) - 1));
        break;
    case POK_NSKRNL:
        result = parserListValues(POL_NSKRNL,arg,parser->params.nsKernel,n,uint(MLP_MAX_GPU));
        break;
    case POK_NSTHRD:
        result = parserArgValue(POL_NSTHRD,arg,parser->params.nsThreshold,n,uint(MLP_MAX_GPU));
        break;
    /* This is not an option at all, but rather a command line argument.  If a
     * parser receiving this key returns success, the fact is recorded, and the
     * ARGP_KEY_NO_ARGS case won't be used.  HOWEVER, if while processing the
     * argument, a parser function decrements the NEXT field of the state it's
     * passed, the option won't be considered processed; this is to allow you to
     * actually modify the argument (perhaps into an option), and have it processed again.
     */
    case ARGP_KEY_ARG:
        ARGSHOW(ARGP_KEY_ARG,arg);
        if(!parser->addFile(arg))
            parser->status = PARS_STATUS_MANYNOPTS;
        break;
    /* There are no more command line arguments at all.
     */
    case ARGP_KEY_END:
        if(argVersionCalled)
            parser->status = PARS_STATUS_VERSION;
        ARGSHOW(ARGP_KEY_END,arg);
        break;
    /* Because it's common to want to do some special processing if there aren't
     * any non-option args, user parsers are called with this key if they didn't
     * successfully process any non-option arguments.  Called just before
     * ARGP_KEY_END (where more general validity checks on previously parsed
     * arguments can take place).
     */
    case ARGP_KEY_NO_ARGS:
        ARGSHOW(ARGP_KEY_NO_ARGS,arg);
        break;
    /* Passed in before any parsing is done.  Afterwards, the values of each
     * element of the CHILD_INPUT field, if any, in the state structure is
     * copied to each child's state to be the initial value of the INPUT field.
     */
    case ARGP_KEY_INIT:
        ARGSHOW(ARGP_KEY_INIT,arg);
        parser->params.argc = state->argc;
        parser->params.argv = (const char **) state->argv;
        parserGetVersion(parser->params);
        break;
    /* Passed in when parsing has successfully been completed (even if there are
     * still arguments remaining).
     */
    case ARGP_KEY_SUCCESS:
        ARGSHOW(ARGP_KEY_SUCCESS,arg);
        break;
    /* Passed in if an error occurs.
     */
    case ARGP_KEY_ERROR:
        ARGSHOW(ARGP_KEY_ERROR,arg);
        result = -1;
        break;
    /* There are remaining arguments not parsed by any parser, which may be found
     * starting at (STATE->argv + STATE->next).  If success is returned, but
     * STATE->next left untouched, it's assumed that all arguments were consume,
     * otherwise, the parser should adjust STATE->next to reflect any arguments consumed.
     */
    case ARGP_KEY_ARGS:
        ARGSHOW(ARGP_KEY_ARGS,arg);
        break;
    /* Use after all other keys, including SUCCESS & END.
     */
    case ARGP_KEY_FINI:
        ARGSHOW(ARGP_KEY_FINI,arg);
        break;
    default:
        cerr << "key=0x" << std::hex << key << "\targ='" << arg << "'" << endl;
        result = -1;
        break;
    }

    return result;
}

#if LOG_LEVEL > 2

#define SHOW_RESULT

void
showResult(ArgParams &argParams)
{
    const char *msg;

    switch(argParams.result) {
    case PARS_STATUS_OK:
        msg = "OK";
        break;
    case PARS_STATUS_HELP:
        msg = "PARS_STATUS_HELP";
        break;
    case PARS_STATUS_USAGE:
        msg = "PARS_STATUS_USAGE";
        break;
    case PARS_STATUS_VERSION:
        msg = "PARS_STATUS_VERSION";
        break;
    case PARS_STATUS_BASH:
        msg = "PARS_STATUS_BASH";
        break;
    case PARS_STATUS_MANYNOPTS:
        msg = "PARS_STATUS_MANYNOPTS";
        break;
    default:
        msg = "Invalid result";
        break;
    }
    cout << "Parse result: " << msg << endl;
}

#endif

// ################################################################################ //
// ##                                                                            ## //
// ##                              CLASS MLParser                                ## //
// ##                                                                            ## //
// ################################################################################ //

bool
MLParser::addFile(const char *fname)
{
    if(fileCount == fileMax) {
        const
        char   **p;
        uint     size;

        // Allocate new array
        size = fileCount + PARS_CHUNK;
        p = new const char *[size];
        if(p == NULL)
            return false;

        // Update buffer size
        fileMax = size;

        // Copy filenames to new array, if any
        if(files) {
            memcpy(p,files,sizeof(*files) * fileCount);
            delete[] files;
        }
        // Points to new buffer
        files = p;
    }
    // Add new filename
    files[fileCount++] = fname;

    return true;
}

void
MLParser::parse(int argc, char **argv)
{
    argp      argp;
    int       result;

    argp_program_bug_address  = APP_EMAIL;
    argp_program_version      = NULL;
    argp_program_version_hook = parserWriteVersion;

    argp.options     = parserOptions;
    argp.parser      = parserParser;
    argp.args_doc    = "FILE";
    argp.argp_domain = NULL;
    argp.children    = NULL;
    argp.doc         = APP_DOC;
    argp.help_filter = parserFilter;

    result = argp_parse(&argp,argc,argv,ARGP_NO_EXIT,0,this);

    if(fileCount > 0) {
        if(params.files)
            delete[] params.files;

        params.files = new const char *[fileCount];
        for(uint i=0;i < fileCount;i++)
            params.files[i] = files[i];
        params.fileCount = fileCount;
    }

#ifdef SHOW_RESULT
    showResult(argParams);
#endif

    switch(result) {
    case PARS_STATUS_OK:
        status = PARS_STATUS_OK;
        break;
    case PARS_STATUS_BASH:
        status = PARS_STATUS_BASH;
        break;
    case ENOMEM:
        status = PARS_STATUS_NOMEM;
        break;
    case EINVAL:
        status = PARS_STATUS_INVALID;
        break;
    default:
        result = PARS_STATUS_ERROR;
        break;
    }
}
