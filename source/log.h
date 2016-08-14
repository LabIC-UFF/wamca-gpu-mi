/**
 * @file   log.h
 *
 * @brief  Message logging
 *
 * @author Eyder Rios
 * @date   2014-11-29
 */

#include <cstdio>


#ifndef __log_h
#define __log_h

#define LOG_FUNC_WIDTH              25

// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //

//#define LOG_PREFIX
//#define LOG_COUNTER
#define LOG_PRETTY_FUNCTION

#ifdef  LOG_PRETTY_FUNCTION
#define FUNCTION_NAME       __PRETTY_FUNCTION__
#else
#define FUNCTION_NAME       __FUNCTION__
#endif

#define lprintf(fmt,...)            logPrintf(FUNCTION_NAME,fmt,##__VA_ARGS__)
#define ltrace()                    logPrintf(FUNCTION_NAME,"line %d, file %s\n",__LINE__,__FILE__)
#define ltracef(fmt,...)            logPrintf(FUNCTION_NAME,"line %d, file %s\t: " fmt,__LINE__,__FILE__,##__VA_ARGS__)

//#define kprintf(fmt,...)            printf("[%-*.*s] " fmt,LOG_FUNC_WIDTH,LOG_FUNC_WIDTH,__FUNCTION__,##__VA_ARGS__)
#define kprintf(fmt,...)            printf(fmt,##__VA_ARGS__)

#if LOG_LEVEL >= 1

#define l1trace()                   ltrace()
#define l1tracef                    ltracef
#define l1printf                    lprintf
#define k1printf                    kprintf

#else

#define l1trace()
#define l1tracef
#define l1printf
#define k1printf

#endif

#if LOG_LEVEL >= 2

#define l2trace()                   ltrace()
#define l2tracef                    ltracef
#define l2printf                    lprintf
#define k2printf                    kprintf

#else

#define l2trace
#define l2tracef
#define l2printf
#define k2printf

#endif

#if LOG_LEVEL >= 3

#define l3trace()                   ltrace()
#define l3tracef                    ltracef
#define l3printf                    lprintf
#define k3printf                    kprintf

#else

#define l3trace
#define l3tracef
#define l3printf
#define k3printf

#endif

#if LOG_LEVEL >= 4

#define l4trace()                   ltrace()
#define l4tracef                    ltracef
#define l4printf                    lprintf
#define k4printf                    kprintf

#else

#define l4trace()
#define l4tracef
#define l4printf
#define k4printf

#endif

// ################################################################################ //
// ##                                                                            ## //
// ##                                  FUNCTIONS                                 ## //
// ##                                                                            ## //
// ################################################################################ //

/*!
 * Initializes log system.
 */
void
logInit();
/*!
 * Print a log message in printf() style.
 *
 * @param   fnc         Name of function where log_printf() is called
 * @param   fmt         Format string
 *
 * @return  Returns the number of characters sent to output
 */
int
logPrintf(const char *fnc, const char *fmt, ...);

#endif	// __log_h
