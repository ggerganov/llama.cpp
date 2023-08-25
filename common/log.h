#pragma once

#include <chrono>
#include <cstring>
#include <sstream>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>

// --------------------------------
//
// Basic usage:
//
// --------
//
//  The LOG() and LOG_TEE() macros are ready to go by default
//   they do not require any initialization.
//
//  LOGLN() and LOG_TEELN() are variants which automatically
//   include \n character at the end of the log string.
//
//  LOG() behaves exactly like printf, by default writing to a logfile.
//  LOG_TEE() additionally, prints to the screen too ( mimics Unix tee command ).
//
//  Default logfile is named
//   "llama.<threadID>.log"
//  Default LOG_TEE() secondary output target is
//   stderr
//
//  Logs can be dynamically disabled or enabled using functions:
//   log_disable()
//  and
//   log_enable()
//
//  A log target can be changed with:
//   log_set_target( string )
//    creating and opening, or re-opening a file by string filename
//  or
//   log_set_target( FILE* )
//    allowing to point at stderr, stdout, or any valid FILE* file handler.
//
// --------
//
// End of Basic usage.
//
// --------------------------------

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

#ifndef LOG_TEE_TARGET
#define LOG_TEE_TARGET stderr
#endif

// Utility to obtain "pid" like unique process id and use it when creating log files.
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

// Utility function for generating log file names with unique id based on thread id.
//  invocation with LOG_FILENAME_GENERATOR( "llama", "log" ) creates a string "llama.<number>.log"
//  where the number is a runtime id of the current thread.

#define LOG_FILENAME_GENERATOR(log_file_basename, log_file_extension) _log_filename_generator(log_file_basename, log_file_extension)

// INTERNAL, DO NOT USE
inline std::string _log_filename_generator(std::string log_file_basename, std::string log_file_extension)
{
    return std::string().append(log_file_basename).append(".").append(log_get_pid()).append(".").append(log_file_extension);
}

#ifndef LOG_DEFAULT_FILE_NAME
#define LOG_DEFAULT_FILE_NAME LOG_FILENAME_GENERATOR("llama", "log")
#endif

// Utility for turning #define values into string literals
//  so we can have a define for stderr and
//  we can print "stderr" instead of literal stderr, etc.
#define _LOG_STRINGIZE(s) #s
#define LOG_STRINGIZE(s) _LOG_STRINGIZE(s)

#define LOG_TEE_TARGET_STRING LOG_STRINGIZE(LOG_TEE_TARGET)

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

#ifdef LOG_TEE_TIMESTAMPS
#define LOG_TEE_TIMESTAMP_FMT "[%lu]"
#define LOG_TEE_TIMESTAMP_VAL , (std::chrono::duration_cast<std::chrono::duration<std::uint64_t>>(std::chrono::system_clock::now().time_since_epoch())).count()
#else
#define LOG_TEE_TIMESTAMP_FMT
#define LOG_TEE_TIMESTAMP_VAL
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

#ifdef LOG_TEE_FILE_LINE_FUNCTION
#define LOG_TEE_FLF_FMT "[%24s:%5d][%24s] "
#define LOG_TEE_FLF_VAL , __FILE__, __LINE__, __FUNCTION__
#else
#define LOG_TEE_FLF_FMT
#define LOG_TEE_FLF_VAL
#endif

// Utility for synchronizing log configuration state
//  since std::optional was introduced only in c++17
enum LogTriState
{
    LogTriStateSame,
    LogTriStateFalse,
    LogTriStateTrue
};

// INTERNAL, DO NOT USE
//  USE LOG() INSTEAD
//
#define _LOG(str, ...)                                                                                                \
    {                                                                                                                 \
        /*fprintf(stderr, "DBG:" str, ##__VA_ARGS__);*/                                                               \
        if (LOG_TARGET != nullptr)                                                                                    \
        {                                                                                                             \
            fprintf(LOG_TARGET, LOG_TIMESTAMP_FMT LOG_FLF_FMT str "%s" LOG_TIMESTAMP_VAL LOG_FLF_VAL, ##__VA_ARGS__); \
            fflush(LOG_TARGET); /*fprintf(stderr, "DBGEND\n");*/                                                      \
        }                                                                                                             \
    }

// INTERNAL, DO NOT USE
//  USE LOG_TEE() INSTEAD
//
#define _LOG_TEE(str, ...)                                                                                                                \
    {                                                                                                                                     \
        /*fprintf(stderr, "DBG:" str, ##__VA_ARGS__);*/                                                                                   \
        if (LOG_TARGET != nullptr)                                                                                                        \
        {                                                                                                                                 \
            fprintf(LOG_TARGET, LOG_TIMESTAMP_FMT LOG_FLF_FMT str "%s" LOG_TIMESTAMP_VAL LOG_FLF_VAL, ##__VA_ARGS__);                     \
            fflush(LOG_TARGET); /*fprintf(stderr, "DBGEND\n");*/                                                                          \
        }                                                                                                                                 \
        if (LOG_TARGET != nullptr && LOG_TARGET != stdout && LOG_TARGET != stderr && LOG_TEE_TARGET != nullptr)                           \
        {                                                                                                                                 \
            fprintf(LOG_TEE_TARGET, LOG_TEE_TIMESTAMP_FMT LOG_TEE_FLF_FMT str "%s" LOG_TEE_TIMESTAMP_VAL LOG_TEE_FLF_VAL, ##__VA_ARGS__); \
            fflush(LOG_TEE_TARGET); /*fprintf(stderr, "DBGEND\n");*/                                                                      \
        }                                                                                                                                 \
    }

// The '\0' as a last argument, is a trick to bypass the silly
//  "warning: ISO C++11 requires at least one argument for the "..." in a variadic macro"
//  so we can have a single macro which can be called just like printf.

// Main LOG macro.
//  behaves like printf, and supports arguments the exact same way.
//
#define LOG(...) _LOG(__VA_ARGS__, "")

// Main TEE macro.
//  does the same as LOG
//  and
//  simultaneously writes stderr.
//
// Secondary target can be changed just like LOG_TARGET
//  by defining LOG_TEE_TARGET
//
#define LOG_TEE(...) _LOG_TEE(__VA_ARGS__, "")

// LOG macro variants with auto endline.
#define LOGLN(...) _LOG(__VA_ARGS__, "\n")
#define LOG_TEELN(...) _LOG_TEE(__VA_ARGS__, "\n")

// INTERNAL, DO NOT USE
inline FILE *_log_handler1(bool change = false, LogTriState disable = LogTriStateSame, std::string filename = LOG_DEFAULT_FILE_NAME, FILE *target = nullptr)
{
    // std::cerr << "\tFNM:" << filename << "TGT:" << (uint64_t)target << std::endl;
    static bool _initialized{false};
    static bool _disabled{(filename.empty() && target == nullptr)};
    static std::string log_current_filename{filename};
    static FILE *log_current_target{target};
    static FILE *logfile = nullptr;

    if (change) [[unlikely]]
    {
        if (disable == LogTriStateTrue)
        {
            // Disable primary target
            _disabled = true;
        }
        // If previously disabled, only enable, and keep previous target
        else if (disable == LogTriStateFalse)
        {
            _disabled = false;
        }
        // Otherwise, process the arguments
        else
        {
            if (log_current_filename.compare(filename) != 0)
            {
                // std::cerr << "\t\tFNM changed, deinit" << std::endl;
                _initialized = false;
            }

            if (log_current_target != target)
            {
                // std::cerr << "\t\tTGT changed, deinit" << std::endl;
                _initialized = false;
            }
        }
    }

    if (_initialized) [[likely]]
    {
        // std::cerr << "\t\tIS Inited" << std::endl;
        //  with fallback in case something went wrong
        // std::cerr << "\t\tEarly Done" << std::endl;
        if (_disabled)
        {
            // Log is disabled
            return nullptr;
        }
        else
        {
            return logfile ? logfile : stderr;
        }
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

        // At this point we set init flag to true, and let the target fallback to stderr
        //  otherwise we would repeatedly fopen() which was already unsuccessful
        _initialized = true;
    }

    // std::cerr << "\tDone" << std::endl;
    return logfile ? logfile : stderr;
}

// INTERNAL, DO NOT USE
inline FILE *_log_handler2(bool change = false, LogTriState disable = LogTriStateSame, FILE *target = nullptr, std::string filename = LOG_DEFAULT_FILE_NAME)
{
    return _log_handler1(change, disable, filename, target);
}

// Disables logs entirely at runtime.
//  Makes LOG() and LOG_TEE() produce no output,
//  untill enabled back.
#define LOG_DISABLE() _log_disable()

// INTERNAL, DO NOT USE
inline FILE *_log_disable()
{
    return _log_handler1(true, LogTriStateTrue);
}

// Enables logs at runtime.
#define LOG_ENABLE() _log_enable()

// INTERNAL, DO NOT USE
inline FILE *_log_enable()
{
    return _log_handler1(true, LogTriStateFalse);
}

// Sets target fir logs, either by a file name or FILE* pointer (stdout, stderr, or any valid FILE*)
#define LOG_SET_TARGET(target) _log_set_target(target)

// INTERNAL, DO NOT USE
inline FILE *_log_set_target(std::string filename) { return _log_handler1(true, LogTriStateSame, filename); }
inline FILE *_log_set_target(FILE *target) { return _log_handler2(true, LogTriStateSame, target); }

// INTERNAL, DO NOT USE
inline FILE *log_handler() { return _log_handler1(); }

inline void log_test()
{
    std::cerr << "LOGDBG: LOGTEST" << std::endl;

    LOG_DISABLE();
    LOG("01 Hello World to nobody, because logs are disabled!\n")
    LOG_ENABLE();
    LOG("02 Hello World to default output, which is \"%s\" ( Yaaay, arguments! )!\n", LOG_STRINGIZE(LOG_TARGET))
    LOG_TEE("03 Hello World to **both** default output and " LOG_TEE_TARGET_STRING "!\n")
    LOG_SET_TARGET(stderr);
    LOG("04 Hello World to stderr!\n")
    LOG_TEE("05 Hello World TEE with double printing to stderr prevented!\n")
    LOG_SET_TARGET(LOG_DEFAULT_FILE_NAME);
    LOG("06 Hello World to default log file!\n")
    LOG_SET_TARGET(stdout);
    LOG("07 Hello World to stdout!\n")
    LOG_SET_TARGET(LOG_DEFAULT_FILE_NAME);
    LOG("08 Hello World to default log file again!\n")
    LOG_DISABLE();
    LOG("09 Hello World _1_ into the void!\n")
    LOG_ENABLE();
    LOG("10 Hello World back from the void ( you should not see _1_ in the log or the output )!\n")
    LOG_DISABLE();
    LOG_SET_TARGET("llama.anotherlog.log");
    LOG("11 Hello World _2_ to nobody, new target was selected but logs are still disabled!\n")
    LOG_ENABLE();
    LOG("12 Hello World this time in a new file ( you should not see _2_ in the log or the output )?\n")
    LOG_SET_TARGET("llama.yetanotherlog.log");
    LOG("13 Hello World this time in yet new file?\n")
    LOG_SET_TARGET(LOG_FILENAME_GENERATOR("llama_autonamed", "log"));
    LOG("14 Hello World in log with generated filename!\n")

    // exit(0);
}

inline bool log_param_single_parse(std::string param)
{
    std::cerr << "LOGDBG: single param: " << param << std::endl;

    if (std::string("--log-test").compare(param) == 0)
    {
        log_test();
        return true;
    }
    else if (std::string("--log-disable").compare(param) == 0)
    {
        LOG_DISABLE();
        return true;
    }
    else if (std::string("--log-enable").compare(param) == 0)
    {
        LOG_ENABLE();
        return true;
    }

    std::cerr << "LOGDBG: single param NO MATCH " << param << std::endl;

    return false;
}

inline bool log_param_pair_parse(bool check_but_dont_parse, std::string param, std::string next = std::string())
{
    std::cerr << "LOGDBG: pair param: " << param << "/" << next << std::endl;

    if (std::string("--log-file").compare(param) == 0)
    {
        if (check_but_dont_parse)
        {
            return true;
        }
        LOG_SET_TARGET(LOG_FILENAME_GENERATOR(next.empty() ? "unnamed" : next, "log"));
        return true;
    }

    std::cerr << "LOGDBG: pair param NO MATCH " << param << "/" << next << std::endl;

    return false;
}

inline void log_print_usage()
{
    fprintf(stdout, "log options:\n");
    /*
    fprintf(stdout, "  -h, --help            show this help message and exit\n");
    // spacing
    fprintf(stdout, "__-param----------------Description\n");*/
    fprintf(stdout, "  --log-test            Run simple logging test\n");
    fprintf(stdout, "  --log-disable         Disable trace logs\n");
    fprintf(stdout, "  --log-enable          Enable trace logs\n");
    fprintf(stdout, "  --log-file            Specify a log filename (without extension)\n");
    fprintf(stdout, "                        Log file will be tagged with unique ID and written as \"<name>.<ID>.log\"\n"); /*  */
}

#ifndef _WIN32
// TODO:
//  Windows doesn't seem to like this somehow

#define LOG_DUMP_CMDLINE( argc, argv ) _log_dump_cmdline(argc,argv)

// INTERNAL, DO NOT USE
inline void _log_dump_cmdline(int argc, char **argv)
{
    std::string buf;
    for (int i = 0; i < argc; ++i)
    {
        if (std::string(argv[i]).find(' ') != std::string::npos)
        {
            buf.append(" \"").append(argv[i]).append("\"");
        }
        else
        {
            buf.append(" ").append(argv[i]);
        }
    }
    LOGLN("Cmd:%s", buf.c_str())
}

#else
#define LOG_DUMP_CMDLINE(...) // dummy stub
#endif

#define LOG_TOSTR(var) _log_var_to_string(var).c_str()

inline std::string _log_var_to_string(bool var)
{
    return var ? "true" : "false";
}

inline std::string _log_var_to_string(std::string var)
{
    return var;
}

inline std::string _log_var_to_string(std::vector<int> var)
{
    std::string buf;
    buf.append("[ ");
    bool first = true;
    for (auto e : var)
    {
        if (first)
        {
            first = false;
        }
        else
        {
            buf.append(", ");
        }
        buf.append(std::to_string(e));
    }
    buf.append(" ]");

    return buf;
}

#define LOG_TOKENS_TOSTR_PRETTY(tokens, ctx)                        \
    [&tokens, &ctx]()                                               \
    {                                                               \
        std::string buf("[ ");                                      \
        bool first = true;                                          \
        for (const auto &token : tokens)                            \
        {                                                           \
            if (!first)                                             \
                buf.append(", ");                                   \
            else                                                    \
                first = false;                                      \
                                                                    \
            auto detokenized = llama_token_to_str(ctx, token);      \
                                                                    \
            detokenized.erase(                                      \
                std::remove_if(                                     \
                    detokenized.begin(),                            \
                    detokenized.end(),                              \
                    [](const char c) { return !std::isprint(c); }), \
                detokenized.end());                                 \
                                                                    \
            buf                                                     \
                .append("'")                                        \
                .append(detokenized)                                \
                .append("'")                                        \
                .append(":")                                        \
                .append(std::to_string(token));                     \
        }                                                           \
        return buf.append(" ]");                                    \
    }()                                                             \
        .c_str()


#ifdef LOG_DISABLE_LOGS

#undef LOG
#define LOG(...) // dummy stub
#undef LOGLN
#define LOGLN(...) // dummy stub

#undef LOG_TEE
#define LOG_TEE(...) fprintf(stderr, __VA_ARGS__); // convert to normal fprintf

#undef LOG_TEELN
#define LOG_TEELN(...) fprintf(stderr, __VA_ARGS__); // convert to normal fprintf

#undef LOG_DISABLE
#define LOG_DISABLE() // dummy stub

#undef LOG_ENABLE
#define LOG_ENABLE() // dummy stub

#undef LOG_ENABLE
#define LOG_ENABLE() // dummy stub

#undef LOG_SET_TARGET
#define LOG_SET_TARGET(...) // dummy stub

#undef LOG_DUMP_CMDLINE
#define LOG_DUMP_CMDLINE(...) // dummy stub

#endif // LOG_DISABLE_LOGS
