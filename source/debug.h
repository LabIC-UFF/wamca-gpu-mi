/**
 * @file   debug.h
 *
 * @brief  Backtrace for debugging purposes
 *
 * @author Eyder Rios
 * @date   2016-01-20
 */

#ifndef __debug_h
#define __debug_h

#include <execinfo.h>

// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //

#define TRACE_MAX_ENTRY     10


// ################################################################################# //
// ##                                                                             ## //
// ##                                 DATA TYPES                                  ## //
// ##                                                                             ## //
// ################################################################################# //

// ################################################################################# //
// ##                                                                             ## //
// ##                              GLOBAL VARIABLES                               ## //
// ##                                                                             ## //
// ################################################################################# //

// ################################################################################# //
// ##                                                                             ## //
// ##                                 FUNCTIONS                                   ## //
// ##                                                                             ## //
// ################################################################################# //

/*
 * IMPORTANT
 *
 * The symbols are taken from the dynamic symbol table.
 * You need the '-rdynamic' option to gcc, which makes it pass a flag to the linker
 * which ensures that all symbols are placed in the table.
 */

inline
void
debugBacktrace()
{
    void    *buffer[TRACE_MAX_ENTRY];
    int      bufferSize;
    char   **text;

    bufferSize = backtrace(buffer,TRACE_MAX_ENTRY);

    text = backtrace_symbols(buffer,bufferSize);
    if(text == NULL) {
        perror("mlsolver");
        exit(-1);
    }

    for(int i=0;i < bufferSize;i++)
        printf("%s\n",text[i]);
    free(text);
}

#endif
