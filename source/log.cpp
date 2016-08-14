/**
 * @file   log.cpp
 *
 * @brief  Message logging
 *
 * @author Eyder Rios
 * @date   2014-11-29
 */

#include <stdio.h>
#include <stdarg.h>
#include "log.h"

// ################################################################################ //
// ##                                                                            ## //
// ##                                GLOBAL VARIABLES                            ## //
// ##                                                                            ## //
// ################################################################################ //

int
logCounter = 0;

int
logFlags = 0;

// ################################################################################ //
// ##                                                                            ## //
// ##                                  FUNCTIONS                                 ## //
// ##                                                                            ## //
// ################################################################################ //

void
logInit()
{
    logCounter = 0;
}

#ifdef LOG_PREFIX

int
logPrintf(const char *fnc, const char *fmt, ...)
{
    va_list args;
    char    buffer[128];
    int     r;

#ifdef LOG_PRETTY_FUNCTION
    for(r=0;fnc[r];r++) {
        if(fnc[r] == ':')
            break;
    }
    if(fnc[r] == ':') {
        for(;r >= 0;r--) {
            if(fnc[r] == ' ') {
                fnc = fnc + r + 1;
                break;
            }
        }
    }
#endif

    va_start(args,fmt);
    sprintf(buffer,"%3d [%-*.*s] %s",++logCounter,LOG_FUNC_WIDTH,LOG_FUNC_WIDTH,fnc,fmt);
    r = vfprintf(stdout,buffer,args);
    fflush(stdout);
    va_end(args);

    return r;
}

#else

int
logPrintf(const char *fnc, const char *fmt, ...)
{
    va_list args;
    char    buffer[128];
    int     r;

#ifdef LOG_PRETTY_FUNCTION
    for(r=0;fnc[r];r++) {
        if(fnc[r] == ':')
            break;
    }
    if(fnc[r] == ':') {
        for(;r >= 0;r--) {
            if(fnc[r] == ' ') {
                fnc = fnc + r + 1;
                break;
            }
        }
    }
#endif

    va_start(args,fmt);
    r = vfprintf(stdout,fmt,args);
    fflush(stdout);
    va_end(args);

    return r;
}

#endif


#if 0

/*
 * printf() support for 'b' format: binary number
 */

#include <printf.h>

static
int
printf_arginfo_binary(const  printf_info *info,
                      size_t n,
                      int   *argtypes)
{
    if(n > 0)
        argtypes[0] = PA_INT;

    return 1;
}

static
int
printf_output_binary(FILE *stream,
                     const struct printf_info *info,
                     const void  *const       *args)
{
    unsigned long   wv;         // Value to be converted
    int             ws,         // Word size in bits
                    wb;         // Word value size in bits
    char            ch;
    int             i,len;

    if(info->is_char) {
        ws = 8 * sizeof(unsigned char);
        wv = *((unsigned char *) args[0]);
    }
    else
    if(info->is_short) {
        ws = 8 * sizeof(unsigned short);
        wv = *((unsigned short *) args[0]);
    }
    else
    if(info->is_long) {
        ws = 8 * sizeof(unsigned long);
        wv = *((unsigned long *) args[0]);
    }
    else
    if(info->is_long_double) {
        ws = 8 * sizeof(unsigned long long);
        wv = *((unsigned long long *) args[0]);
    }
    else {
        ws = 8 * sizeof(unsigned int);
        wv = *((unsigned long long *) args[0]);
    }

    wb = ws;
    while((wb > 1) && !((wv >> (wb - 1)) & 1))
        wb--;

    len  = 0;

    if(info->width > wb) {
        for(i=info->width - wb;i > 0;i--)
            len += fprintf(stream,"%c",info->pad);
    }

    while(wb > 0) {
        wb--;
        ch   = ((wv >> wb) & 1) ? '1' : '0';
        len += fprintf(stream,"%c",ch);
    }
    len += fprintf(stream,"b");

    return len;
}

void
printf_register_binary()
{
    if(register_printf_function ('b',printf_output_binary, printf_arginfo_binary))
        printf("Error registering new printf() convertion.\n");
}

#endif
