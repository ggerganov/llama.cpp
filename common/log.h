#pragma once

#include <chrono>

// Specifies a log target.
//  default simply prints log to stderr
//  this can be changed, by defining LOG_TARGET
//  like so:
//
//  #define LOG_TARGET (a valid FILE*)
//  #include "log.h"
//
//  The log target can also be redirected to a function
//  like so:
//
//  #define LOG_TARGET log_handler()
//  #include "log.h"
//
//  FILE* log_handler()
//  {
//      return stderr;
//  }
//
//  or:
//
//  #define LOG_TARGET log_handler("somelog.log")
//  #include "log.h"
//
//  FILE* log_handler(char*filename)
//  {
//      (...)
//      return fopen(...)
//  }
//
#ifndef LOG_TARGET
#define LOG_TARGET stderr
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