#pragma once

#define KOMPUTE_LOG_LEVEL_TRACE 0
#define KOMPUTE_LOG_LEVEL_DEBUG 1
#define KOMPUTE_LOG_LEVEL_INFO 2
#define KOMPUTE_LOG_LEVEL_WARN 3
#define KOMPUTE_LOG_LEVEL_ERROR 4
#define KOMPUTE_LOG_LEVEL_CRITICAL 5
#define KOMPUTE_LOG_LEVEL_OFF 6

// Logging is disabled entirely.
#if KOMPUTE_OPT_LOG_LEVEL_DISABLED
#define KP_LOG_TRACE(...)
#define KP_LOG_DEBUG(...)
#define KP_LOG_INFO(...)
#define KP_LOG_WARN(...)
#define KP_LOG_ERROR(...)
#else

#if !KOMPUTE_OPT_USE_SPDLOG
#if VK_USE_PLATFORM_ANDROID_KHR
#include <android/log.h>
#include <fmt/core.h>
static const char* KOMPUTE_LOG_TAG = "KomputeLog";
#else
#if KOMPUTE_BUILD_PYTHON
#include <pybind11/pybind11.h>
namespace py = pybind11;
// from python/src/main.cpp
extern py::object kp_trace, kp_debug, kp_info, kp_warning, kp_error;
#else
#include <fmt/core.h>
#endif // KOMPUTE_BUILD_PYTHON
#endif // VK_USE_PLATFORM_ANDROID_KHR
#else
#include <spdlog/spdlog.h>
#endif // !KOMPUTE_OPT_USE_SPDLOG
#include <set>
#include <string>
#include <vector>
namespace logger {
// Setup the logger, note the loglevel can not be set below the CMake log level
// (To change this use -DKOMPUTE_OPT_LOG_LEVEL=...)
void
setupLogger();

// Logging is enabled, but we do not use Spdlog. So we use fmt in case nothing
// else is defined, overriding logging.
#if !KOMPUTE_OPT_USE_SPDLOG

#ifndef KP_LOG_TRACE
#if KOMPUTE_OPT_ACTIVE_LOG_LEVEL <= KOMPUTE_LOG_LEVEL_TRACE
#if VK_USE_PLATFORM_ANDROID_KHR
#define KP_LOG_TRACE(...)                                                      \
    ((void)__android_log_write(                                                \
      ANDROID_LOG_VERBOSE, KOMPUTE_LOG_TAG, fmt::format(__VA_ARGS__).c_str()))
#else
#if KOMPUTE_BUILD_PYTHON
#define KP_LOG_DEBUG(...) kp_trace(fmt::format(__VA_ARGS__))
#else
#define KP_LOG_TRACE(...)                                                      \
    fmt::print("[{} {}] [trace] [{}:{}] {}\n",                                 \
               __DATE__,                                                       \
               __TIME__,                                                       \
               __FILE__,                                                       \
               __LINE__,                                                       \
               fmt::format(__VA_ARGS__))
#endif // KOMPUTE_BUILD_PYTHON
#endif // VK_USE_PLATFORM_ANDROID_KHR
#else
#define KP_LOG_TRACE(...)
#endif
#endif // !KP_LOG_TRACE

#ifndef KP_LOG_DEBUG
#if KOMPUTE_OPT_ACTIVE_LOG_LEVEL <= KOMPUTE_LOG_LEVEL_DEBUG
#if VK_USE_PLATFORM_ANDROID_KHR
#define KP_LOG_DEBUG(...)                                                      \
    ((void)__android_log_write(                                                \
      ANDROID_LOG_DEBUG, KOMPUTE_LOG_TAG, fmt::format(__VA_ARGS__).c_str()))
#else
#if KOMPUTE_BUILD_PYTHON
#define KP_LOG_DEBUG(...) kp_debug(fmt::format(__VA_ARGS__))
#else
#ifdef __FILE_NAME__ // gcc 12 provides only file name without path
#define KP_LOG_DEBUG(...)                                                      \
    fmt::print("[{} {}] [debug] [{}:{}] {}\n",                                 \
               __DATE__,                                                       \
               __TIME__,                                                       \
               __FILE_NAME__,                                                       \
               __LINE__,                                                       \
               fmt::format(__VA_ARGS__))
#else
#define KP_LOG_DEBUG(...)                                                      \
    fmt::print("[{} {}] [debug] [{}:{}] {}\n",                                 \
               __DATE__,                                                       \
               __TIME__,                                                       \
               __FILE__,                                                       \
               __LINE__,                                                       \
               fmt::format(__VA_ARGS__))
#endif // __FILE__NAME__
#endif // KOMPUTE_BUILD_PYTHON
#endif // VK_USE_PLATFORM_ANDROID_KHR
#else
#define KP_LOG_DEBUG(...)
#endif
#endif // !KP_LOG_DEBUG

#ifndef KP_LOG_INFO
#if KOMPUTE_OPT_ACTIVE_LOG_LEVEL <= KOMPUTE_LOG_LEVEL_INFO
#if VK_USE_PLATFORM_ANDROID_KHR
#define KP_LOG_INFO(...)                                                       \
    ((void)__android_log_write(                                                \
      ANDROID_LOG_INFO, KOMPUTE_LOG_TAG, fmt::format(__VA_ARGS__).c_str()))
#else
#if KOMPUTE_BUILD_PYTHON
#define KP_LOG_DEBUG(...) kp_info(fmt::format(__VA_ARGS__))
#else
#define KP_LOG_INFO(...)                                                       \
    fmt::print("[{} {}] [info] [{}:{}] {}\n",                                  \
               __DATE__,                                                       \
               __TIME__,                                                       \
               __FILE__,                                                       \
               __LINE__,                                                       \
               fmt::format(__VA_ARGS__))
#endif // KOMPUTE_BUILD_PYTHON
#endif // VK_USE_PLATFORM_ANDROID_KHR
#else
#define KP_LOG_INFO(...)
#endif
#endif // !KP_LOG_INFO

#ifndef KP_LOG_WARN
#if KOMPUTE_OPT_ACTIVE_LOG_LEVEL <= KOMPUTE_LOG_LEVEL_WARN
#if VK_USE_PLATFORM_ANDROID_KHR
#define KP_LOG_WARN(...)                                                       \
    ((void)__android_log_write(                                                \
      ANDROID_LOG_WARN, KOMPUTE_LOG_TAG, fmt::format(__VA_ARGS__).c_str()))
#else
#if KOMPUTE_BUILD_PYTHON
#define KP_LOG_DEBUG(...) kp_warning(fmt::format(__VA_ARGS__))
#else
#define KP_LOG_WARN(...)                                                       \
    fmt::print("[{} {}] [warn] [{}:{}] {}\n",                                  \
               __DATE__,                                                       \
               __TIME__,                                                       \
               __FILE__,                                                       \
               __LINE__,                                                       \
               fmt::format(__VA_ARGS__))
#endif // KOMPUTE_BUILD_PYTHON
#endif // VK_USE_PLATFORM_ANDROID_KHR
#else
#define KP_LOG_WARN(...)
#endif
#endif // !KP_LOG_WARN

#ifndef KP_LOG_ERROR
#if KOMPUTE_OPT_ACTIVE_LOG_LEVEL <= KOMPUTE_LOG_LEVEL_ERROR
#if VK_USE_PLATFORM_ANDROID_KHR
#define KP_LOG_ERROR(...)                                                      \
    ((void)__android_log_write(                                                \
      ANDROID_LOG_ERROR, KOMPUTE_LOG_TAG, fmt::format(__VA_ARGS__).c_str()))
#else
#if KOMPUTE_BUILD_PYTHON
#define KP_LOG_DEBUG(...) kp_error(fmt::format(__VA_ARGS__))
#else
#define KP_LOG_ERROR(...)                                                      \
    fmt::print("[{} {}] [error] [{}:{}] {}\n",                                 \
               __DATE__,                                                       \
               __TIME__,                                                       \
               __FILE__,                                                       \
               __LINE__,                                                       \
               fmt::format(__VA_ARGS__))
#endif // KOMPUTE_BUILD_PYTHON
#endif // VK_USE_PLATFORM_ANDROID_KHR
#else
#define KP_LOG_ERROR(...)
#endif
#endif // !KP_LOG_ERROR
#else

#define KP_LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#define KP_LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define KP_LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define KP_LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#define KP_LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)

void
setLogLevel(spdlog::level::level_enum level);

spdlog::level::level_enum
getLogLevel();

#endif // !KOMPUTE_OPT_USE_SPDLOG
} // namespace logger

#endif // KOMPUTE_OPT_LOG_LEVEL_DISABLED
