#pragma once

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <chrono>
#include <cstring>
#include <sstream>
#include <iostream>

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

#define _LOG(str, ...)                                                                                            \
    {                                                                                                             \
        /*fprintf(stderr, "DBG:" str, ##__VA_ARGS__);*/                                                           \
        fprintf(LOG_TARGET, LOG_TIMESTAMP_FMT LOG_FLF_FMT str "%c" LOG_TIMESTAMP_VAL LOG_FLF_VAL, ##__VA_ARGS__); \
        fflush(LOG_TARGET); /*fprintf(stderr, "DBGEND\n");*/                                                      \
    }

// This is a trick to bypass the silly
// "warning: ISO C++11 requires at least one argument for the "..." in a variadic macro"
// so we can gave a single macro which can be called just like printf.
#define LOG(...) _LOG(__VA_ARGS__, '\0')

inline std::string log_get_pid()
{
    static std::string pid;
    if (pid.empty()) [[unlikely]]
    {
        // std::this_thread::get_id() is the most portable way of obtaining a "process id"
        //  it's not the same as "pid" but is unique enough to solve multiple instances
        //  trying to write to the same log.
        std::stringstream ss;
        ss << std::this_thread::get_id();
        pid = ss.str();
    }

    return pid;
}

#define LOG_DEFAULT_FILE_NAME std::string().append("llama.").append(log_get_pid()).append(".log")

inline FILE *_log_handler1(bool change = false, std::string filename = LOG_DEFAULT_FILE_NAME, FILE *target = nullptr)
{
    // std::cerr << "\tFNM:" << filename << "TGT:" << (uint64_t)target << std::endl;
    static bool _initialized{false};
    static std::string log_current_filename{filename};
    static FILE *log_current_target{target};
    static FILE *logfile = nullptr;

    if (change && log_current_filename.compare(filename) != 0) [[unlikely]]
    {
        // std::cerr << "\t\tFNM changed, deinit" << std::endl;
        _initialized = false;
    }

    if (change && log_current_target != target) [[unlikely]]
    {
        // std::cerr << "\t\tTGT changed, deinit" << std::endl;
        _initialized = false;
    }

    // std::cerr << "\tINIT:" << (_initialized ? "true" : "false") << std::endl;

    if (_initialized) [[likely]]
    {
        // std::cerr << "\t\tIS Inited" << std::endl;
        //  with fallback in case something went wrong
        // std::cerr << "\t\tEarly Done" << std::endl;
        return logfile ? logfile : stderr;
    }
    else
    {
        // std::cerr << "\t\tIS NOT Inited" << std::endl;
        if (target != nullptr)
        {
            // std::cerr << "\t\t\tTGT not nullptr" << std::endl;
            if (logfile != nullptr && logfile != stdout && logfile != stderr)
            {
                // std::cerr << "\t\t\t\tF close" << std::endl;
                fclose(logfile);
            }

            log_current_filename = LOG_DEFAULT_FILE_NAME;
            log_current_target = target;

            // std::cerr << "\t\t\tTGT set to new target" << std::endl;
            logfile = target;
        }
        else
        {
            // std::cerr << "\t\t\tTGT IS nullptr" << std::endl;
            if (log_current_filename.compare(filename) != 0) [[likely]]
            {
                // std::cerr << "\t\t\t\tFNM changed" << std::endl;

                if (logfile != nullptr && logfile != stdout && logfile != stderr)
                {
                    // std::cerr << "\t\t\t\t\tF close 2" << std::endl;
                    fclose(logfile);
                }

                // std::cerr << "\t\t\t\tF reopen" << std::endl;
                logfile = nullptr;
                logfile = fopen(filename.c_str(), "a");
            }
            else
            {
                // std::cerr << "\t\t\t\tF open" << std::endl;
                // logfile = fopen(filename.c_str(), "wa");
                logfile = fopen(filename.c_str(), "a");
            }
        }

        if (!logfile)
        {
            // std::cerr << "\t\t\tF invalid" << std::endl;
            //  Verify whether the file was opened, otherwise fallback to stderr
            logfile = stderr;

            fprintf(stderr, "Failed to open logfile '%s' with error '%s'\n", filename.c_str(), std::strerror(errno));
            fflush(stderr);
        }

        _initialized = true;
    }

    // std::cerr << "\tDone" << std::endl;
    return logfile ? logfile : stderr;
}
inline FILE *_log_handler2(bool change = false, FILE *target = nullptr, std::string filename = LOG_DEFAULT_FILE_NAME)
{
    return _log_handler1(change, filename, target);
}

inline FILE *log_set_target(std::string filename) { return _log_handler1(true, filename); }
inline FILE *log_set_target(FILE *target) { return _log_handler2(true, target); }
inline FILE *log_handler() { return _log_handler1(); }