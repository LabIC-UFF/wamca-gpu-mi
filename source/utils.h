/**
 * @file   utils.h
 *
 * @brief  General auxiliary functions.
 *
 * @author Eyder Rios
 * @date   2011-09-12
 */


#ifndef __utils_h
#define __utils_h

#include <iostream>
#include <iomanip>
#include <papi/papi.h>
#include "types.h"

/******************************************************************************************
 *
 *                                    MACROS
 *
 ******************************************************************************************/

#define IDIVSAFE(val,ref)       ( ((ref) != 0) ? ( (val) / (ref) ) : 0 )
#define FDIVSAFE(val,ref)	    ( ((ref) != 0) ? ( ((double) (val)) / ((double) (ref)) ) : 0.0 )
#define FDIV(val,ref)           ( (double) (val) / (double) (ref) )

#define DIVCEIL(n,d)            ( (n + d - 1) / (d) )
#define DIV2CEIL(n)             ( ((n) >> 1) + ((n) & 1) )

#define TOEVEN(n)               ( (n) + (n & 1) )

#define PERCENT(val,ref)	    ( 100.0 * FDIVSAFE(val,ref) )
#define PERCTVAL(val,ref)       FDIVSAFE(val,ref)

#define FILLZ(v)                memset(v,0,sizeof(*(v)))

/*
 * Used by log methods.
 */
#ifdef DEBUG
#define EOL         std::endl
#else
#define EOL         '\n'
#endif

#define TAB         '\t'
#define SETP(p)     std::setprecision(p)
#define SETW(w)     std::setw(w)
#define SETWP(w,p)  std::setw(w) << std::setprecision(p)

#define US2MS(t)    ((t) / 1000)
#define MS2US(t)    ((t) * 1000)

#define BIT_WORD(b)     ( 1 << (b) )
#define BIT_SET(a,b)    ( (a) |=  BIT_WORD(b) )
#define BIT_CLEAR(a,b)  ( (a) &= ~BIT_WORD(b) )
#define BIT_FLIP(a,b)   ( (a) ^=  BIT_WORD(b) )
#define BIT_CHECK(a,b)  ( (a)  &  BIT_WORD(b) )

/******************************************************************************************
 *
 *                                   FUNCTIONS
 *
 ******************************************************************************************/

/*!
 * Get real time passed since some arbitrary starting point.
 * The time is returned in microseconds (10^-6 s). This call is equivalent to wall clock time.
 *
 * @return  real time passed since some arbitrary starting point.
 */
inline
ullong sysTimer() {
    return PAPI_get_real_usec();
}
/*!
 * Get number of significant digits of a number.
 *
 * @param	n		Number
 * @param	base	Numeric base (default is 10)
 * @return	Returns the number of significant digits of \a n.
 */
uint
digits(llong n, uint base = 10);

/*!
 * Get system bogomips parameter.
 *
 * @return  Returns the bogomips value.
 */
double
bogomips();

/*!
 * Get IP address from hostname.
 *
 * @param   hostname    Machine name
 * @param   ip          IP address
 * @return  Returns \a true if hostname was successfully solved, otherwise \a false.
 */
bool
getHostIPAddr(char *hostname, char *ip);

/*!
 * Strip filename extension.
 *
 * @param   path    Path to file
 */
char *
stripext(char *path);

/*!
 * Strip directory from path.
 *
 * @param   path    Returns a pointer to filename
 */
char *
stripdir(char *path);

/*!
 * Replace filename extension.
 *
 * If 'ext' is NULL, then filename extension is stripped.
 *
 * @param   path    Path to file
 * @param   ext     New extension
 */
char *
replaceext(char *path, char *ext);

/*!
 * Get the number of set bits in a integer.
 *
 * @param   n   Integer to count 1's bits
 * @return  Returns the number of 1s in integer.
 */
uint
bitCount(uint n);

/*!
 * Get memory alignment used by compiler.
 */
inline
int
alignWordSize() {
    struct align_t {
        char  f1;
        char  f2;
    } s;

    return &s.f2 - &s.f1;
}

/*!
 * Get memory alignment used by compiler.
 */
inline
size_t
alignBlockSize(size_t size) {
    size_t  wsize = alignWordSize();
    return  ( (size / wsize) + (size % wsize != 0) ) * wsize;
}

/*!
 * Get memory alignment used by compiler.
 */
inline
void *
alignNextAddr(void *p, size_t size) {
    return (void *) ( ((byte *) p) + size );
}

inline
const char *
yesNo(bool flag) {
    return flag ? "yes" : "no";
}

inline
const char *
onOff(bool flag) {
    return flag ? "on" : "off";
}

inline
const char *
trueFals(bool flag) {
    return flag ? "true" : "false";
}

#endif
