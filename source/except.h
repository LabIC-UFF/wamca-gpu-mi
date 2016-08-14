/**
 * @file   except.h
 *
 * @brief  Basic exception handling
 *
 * @author Eyder Rios
 * @date   2013-06-20
 */

#ifndef __except_h
#define __except_h

#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <exception>
#include <string>


// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //

#define EXCP_PREFIX_LEN          63


// ################################################################################ //
// ##                                                                            ## //
// ##                                 Class Exception                            ## //
// ##                                                                            ## //
// ################################################################################ //

/*!
 * General exception class
 */
class Exception : public std::exception
{
protected:
    static
    char         appName[EXCP_PREFIX_LEN + 1];   ///< App name (got from CLI)

protected:
    std::string  msg;                            ///< Exception message

protected:

#ifdef EXCEPT_DETAIL
    /*!
     * Format an exception message that is stored in \a msg variable.
     *
     * @param fnc   Function where exception was throw
     * @param fmt   Message format string
     * @param args  Variadic arguments
     */
    void
    setMsg(const char *file, const char *fnc, int line, const char *fmt, va_list args) {
        char  buffer[1024];
        int   i,n;

        if(*appName)
            n = snprintf(buffer,sizeof(buffer),"%s: ",appName);
        else
            n = 0;

        i = strlen(file) - 1;
        while((i >= 0) && (file[i] != '/'))
            i--;
        if(i > 0)
            file = file + i + 1;

        n += snprintf(buffer + n,sizeof(buffer),
                      "Exception in %s\nFile %s, line %d\n",fnc,file,line);
        vsnprintf(buffer + n,sizeof(buffer) - n,fmt,args);
        msg = buffer;
        msg += '.';
        msg += '\n';
    }
#else
    /*!
     * Format an exception message that is stored in \a msg variable.
     *
     * @param fmt   Message format string
     * @param args  Variadic arguments
     */
    void
    setMsg(const char *fmt, va_list args) {
        char  buffer[1024];
        int   n;

        if(*appName)
            n = snprintf(buffer,sizeof(buffer),"%s: ",appName);
        else
            n = 0;
        vsnprintf(buffer + n,sizeof(buffer) - n,fmt,args);
        msg = buffer;
        msg += '.';
        msg += '\n';
    }
#endif

public:
    /*!
     * Create exception instance.
     */
    Exception() {
    }
#ifdef EXCEPT_DETAIL
    /*!
     * Create exception instance in \a printf() style.
     *
     * @param fnc   Function where exception was throw
     * @param fmt   Format string in \a printf() style
     */
    Exception(const char *file, const char *fnc, int line, const char *fmt, ...) throw() {
      va_list args;

      va_start(args,fmt);
      setMsg(file,fnc,line,fmt,args);
      va_end(args);
    }
#else
    /*!
     * Create exception instance in \a printf() style.
     *
     * @param fmt   Format string in \a printf() style
     */
    Exception(const char *fmt, ...) throw() {
      va_list args;

      va_start(args,fmt);
      setMsg(fmt,args);
      va_end(args);
    }
#endif
    /*!
     * Destroy exception class
     */
   ~Exception() throw() {
    }
   /*!
    * Returns the exception message.
    *
    * @return A pointer to the exception message string.
    */
    virtual
    const
    char *
    what()  const throw()  {
        return msg.c_str();
    }
    /*!
     * Friend function
     */
    friend void initialize();

#ifdef  EXCEPT_DETAIL
    friend void warning(const char *file, const char *fnc, int line, const char *fmt, ...);
#else
    friend void warning(const char *fmt, ...);
#endif
};

/*!
 * Silent exception class
 */
class SilentException : public std::exception
{
};

/*!
 * Display a warning message
 */
#ifdef  EXCEPT_DETAIL
    void warning(const char *file, const char *fnc, int line, const char *fmt, ...);
#else
    void warning(const char *fmt, ...);
#endif

char *
function_name(char *s, const char *fnc, int n);


/*!
 * Macros
 */

#define SILENT_EXCEPTION         throw SilentException()

#ifdef  EXCEPT_DETAIL

#define EXCEPTION(fmt,...)       throw Exception(__FILE__,__PRETTY_FUNCTION__,__LINE__,fmt,##__VA_ARGS__)
#define WARNING(fmt,...)         warning(__FILE__,__PRETTY_FUNCTION__,__LINE__,fmt,##__VA_ARGS__)

#else

#define EXCEPTION(fmt,...)       throw Exception(fmt,##__VA_ARGS__)
#define WARNING(fmt,...)         warning(fmt,##__VA_ARGS__)

#endif


#ifndef NDEBUG
#define ASSERT(expr,fmt,...)    if(!(expr)) EXCEPTION(fmt,##__VA_ARGS__)
#else
#define ASSERT(expr,fmt,...)
#endif

#endif
