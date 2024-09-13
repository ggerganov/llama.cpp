#pragma once

#include "ggml.h"

#ifndef __GNUC__
#    define LOG_ATTRIBUTE_FORMAT(...)
#elif defined(__MINGW32__)
#    define LOG_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#    define LOG_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif

#define LOG_DEFAULT_DEBUG 1
#define LOG_DEFAULT_LLAMA 0

// needed by the LOG_TMPL macro to avoid computing log arguments if the verbosity lower
// set via gpt_log_set_verbosity()
extern int gpt_log_verbosity_thold;

void gpt_log_set_verbosity_thold(int verbosity); // not thread-safe

// the gpt_log uses an internal worker thread to print/write log messages
// when the worker thread is paused, incoming log messages are discarded
struct gpt_log;

struct gpt_log * gpt_log_init();
struct gpt_log * gpt_log_main(); // singleton, automatically destroys itself on exit
void             gpt_log_pause (struct gpt_log * log); // pause  the worker thread, not thread-safe
void             gpt_log_resume(struct gpt_log * log); // resume the worker thread, not thread-safe
void             gpt_log_free  (struct gpt_log * log);

LOG_ATTRIBUTE_FORMAT(3, 4)
void gpt_log_add(struct gpt_log * log, enum ggml_log_level level, const char * fmt, ...);

void gpt_log_set_file      (struct gpt_log * log, const char * file);       // not thread-safe
void gpt_log_set_colors    (struct gpt_log * log,       bool   colors);     // not thread-safe
void gpt_log_set_timestamps(struct gpt_log * log,       bool   timestamps);

// helper macros for logging
// use these to avoid computing log arguments if the verbosity is lower than the threshold
//
// for example:
//
//   LOG_DBG("this is a debug message: %d\n", expensive_function());
//
// this will avoid calling expensive_function() if the verbosity is lower than LOG_DEFAULT_DEBUG
//

#define LOG_TMPL(level, verbosity, ...) \
    do { \
        if ((verbosity) <= gpt_log_verbosity_thold) { \
            gpt_log_add(gpt_log_main(), (level), __VA_ARGS__); \
        } \
    } while (0)

#define LOG(...)             LOG_TMPL(GGML_LOG_LEVEL_NONE, 0,         __VA_ARGS__)
#define LOGV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_NONE, verbosity, __VA_ARGS__)

#define LOG_INF(...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  0,                 __VA_ARGS__)
#define LOG_WRN(...) LOG_TMPL(GGML_LOG_LEVEL_WARN,  0,                 __VA_ARGS__)
#define LOG_ERR(...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, 0,                 __VA_ARGS__)
#define LOG_DBG(...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, LOG_DEFAULT_DEBUG, __VA_ARGS__)

#define LOG_INFV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  verbosity, __VA_ARGS__)
#define LOG_WRNV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_WARN,  verbosity, __VA_ARGS__)
#define LOG_ERRV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, verbosity, __VA_ARGS__)
#define LOG_DBGV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, verbosity, __VA_ARGS__)
