/**
 * @file   except.cpp
 *
 * @brief  Basic exception handling
 *
 * @author Eyder Rios
 * @date   2013-06-25
 */

#include <stdarg.h>
#include <unistd.h>
#include <string.h>
#include "except.h"


// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //

#define EXCP_CLI_FILENAME        "/proc/self/cmdline"


// ################################################################################ //
// ##                                                                            ## //
// ##                                GLOBAL VARIABLES                            ## //
// ##                                                                            ## //
// ################################################################################ //

char
Exception::appName[EXCP_PREFIX_LEN + 1];


// ################################################################################ //
// ##                                                                            ## //
// ##                                 Class Exception                            ## //
// ##                                                                            ## //
// ################################################################################ //

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"

/*
 * This makes initialize() function to be called before main()
 */
void initialize(void) __attribute__((constructor));

/*!
 * Initializes Exception class static members.
 * This is called before main() function
 */
void
initialize()
{
    FILE   *fcmd;
    char    buffer[EXCP_PREFIX_LEN + 1],
           *p;
    int     len;

    // Open system file that contains application command line
	if((fcmd = fopen(EXCP_CLI_FILENAME,"rb")) != NULL) {
        // Copy file contents
        fread(buffer,sizeof(*buffer),EXCP_PREFIX_LEN,fcmd);
        // Close command line /proc file
        fclose(fcmd);
        // Ensure string terminator
        buffer[EXCP_PREFIX_LEN] = '\0';

        // Get position of last char in 'buffer'
        len = strlen(buffer);
        if(len > 0)
            len--;

        // Search for beginning of the word
        p = buffer + len;
        while((p != buffer) && (*p != '/'))
            p--;
        if(*p == '/')
            p++;

        strcpy(Exception::appName,p);
	}
	else
	    *Exception::appName = '\0';
}

#pragma GCC diagnostic pop


char *
function_name(char *s, const char *fnc, int n)
{
    int   i;

    // Find starting char
    for(i=0;fnc[i];i++) {
        if((fnc[i] == ':') && (fnc[i + 1] == ':')) {
            fnc += i + 2;
            break;
        }
    }

    // Copy function name
    for(i=0;fnc[i] && (fnc[i] != '(') && (i < n);i++)
        s[i] = fnc[i];
    s[i] = '\0';

    return s;
}

#ifdef EXCEPT_DETAIL

void warning(const char *file, const char *fnc, int line, const char *fmt, ...)
{
    va_list args;
    char    buffer[1024],
			fname[64],
           *s;

    va_start(args,fmt);

    function_name(fname,fnc,sizeof(fname) - 1);

    s  = buffer;
    s += sprintf(s,"In %s\nFile %s, line %d\n",fname,file,line);
    s += vsprintf(s,fmt,args);
    fprintf(stderr,"%s\n",buffer);
    fflush(stderr);

    va_end(args);
}

#else

void warning(const char *fmt, ...)
{
    va_list args;
    char    buffer[1024],
           *s;

    va_start(args,fmt);

    s  = buffer;
    s += sprintf(s,"%s: ",Exception::appName);
    s += vsprintf(s,fmt,args);
    fprintf(stderr,"%s\n",buffer);
    fflush(stderr);

    va_end(args);
}

#endif
