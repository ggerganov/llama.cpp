#pragma once

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <chrono>
#include <atomic>
#include <cstring>

// Specifies a log target.
//  default uses log_handler() with "llama.log" log file
//  this can be changed, by defining LOG_TARGET
//  like so:
//
//  #define LOG_TARGET (a valid FILE*)
//  #include "log.h"
//
//  or it can be simply redirected to stdout or stderr
//  like so:
//
//  #define LOG_TARGET stderr
//  #include "log.h"
//
//  The log target can also be redirected to a diffrent function
//  like so:
//
//  #define LOG_TARGET log_handler_diffrent()
//  #include "log.h"
//
//  FILE* log_handler_diffrent()
//  {
//      return stderr;
//  }
//
//  or:
//
//  #define LOG_TARGET log_handler_another_one("somelog.log")
//  #include "log.h"
//
//  FILE* log_handler_another_one(char*filename)
//  {
//      static FILE* logfile = nullptr;
//      (...)
//      if( !logfile )
//      {
//          fopen(...)
//      }
//      (...)
//      return logfile
//  }
//
#ifndef LOG_TARGET
#define LOG_TARGET log_handler()
#endif

// Allows disabling timestamps.
//  in order to disable, define LOG_NO_TIMESTAMPS
//  like so:
//
//  #define LOG_NO_TIMESTAMPS
//  #include "log.h"
//
#ifndef LOG_NO_TIMESTAMPS
#define LOG_TIMESTAMP_FMT "[%lu]"
#define LOG_TIMESTAMP_VAL , (std::chrono::duration_cast<std::chrono::duration<std::uint64_t>>(std::chrono::system_clock::now().time_since_epoch())).count()
#else
#define LOG_TIMESTAMP_FMT
#define LOG_TIMESTAMP_VAL
#endif

// Allows disabling file/line/function prefix
//  in order to disable, define LOG_NO_FILE_LINE_FUNCTION
//  like so:
//
//  #define LOG_NO_FILE_LINE_FUNCTION
//  #include "log.h"
//
#ifndef LOG_NO_FILE_LINE_FUNCTION
#define LOG_FLF_FMT "[%24s:%5d][%24s] "
#define LOG_FLF_VAL , __FILE__, __LINE__, __FUNCTION__
#else
#define LOG_FLF_FMT
#define LOG_FLF_VAL
#endif

#define _LOG(str, ...)                                                                                        \
    {                                                                                                             \
        fprintf(LOG_TARGET, LOG_TIMESTAMP_FMT LOG_FLF_FMT str "%c" LOG_TIMESTAMP_VAL LOG_FLF_VAL, ##__VA_ARGS__); \
        fflush(LOG_TARGET);                                                                                       \
    }

// This us a trick to bypass the silly
// "warning: ISO C++11 requires at least one argument for the "..." in a variadic macro"
// so we xan gave a single macro which can be called just like printf.
#define LOG(...) _LOG(__VA_ARGS__, '\n')

inline FILE *log_handler(std::string s = "llama.log")
{
    static std::atomic_bool _initialized{false};
    static std::atomic_bool _logfileopened{false};

    static FILE* logfile = nullptr;

    if (_initialized)[[likely]]
    {
        // with fallback in case something went wrong
        return logfile ? logfile : stderr;
    }
    else
    {
        // Mutex-less threadsafe synchronisation.
        //  we need to make sure not more than one invocation of this function
        //  attempts to open a file at once.
        //  compare_exchange_strong checks and updates a flag
        //  in a single atomic operation.
        bool expected{false};
        if( _logfileopened.compare_exchange_strong(expected,true) )
        {
            // If the flag was previously false, and we managed to turn it true
            //  ew are now responsible for opening the log file
            logfile = fopen( s.c_str(), "wa" );

            if( !logfile )
            {
                // Verify whether the file was opened, otherwise fallback to stderr
                logfile = stderr;

                fprintf(stderr, "Failed to open logfile '%s' with error '%s'\n", s.c_str(), std::strerror(errno));
                fflush(stderr);
            }

            _initialized.store(true);
        }
        else
        {
            // We are not first to open the log file
            //
            //  TODO: Better thread-safe option, possibly std::condition_variable

            return stderr;
        }

        // with fallback in case something went wrong
        return logfile ? logfile : stderr;
    }

    return stderr;
}