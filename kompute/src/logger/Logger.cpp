#include "kompute/logger/Logger.hpp"

#if !KOMPUTE_OPT_LOG_LEVEL_DISABLED
#if !KOMPUTE_OPT_USE_SPDLOG
#else
#include <cassert>
#include <iostream>
#include <memory>
#include <mutex>
#include <spdlog/async.h>
#include <spdlog/common.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string>
#endif // !KOMPUTE_OPT_USE_SPDLOG

namespace logger {
#if !KOMPUTE_OPT_USE_SPDLOG

void
setupLogger()
{
}

#else
constexpr int THREAD_QUEUE_LENGTH = 8192;

void
setupLogger()
{
    // Ensure we setup the logger only once
    static bool setup = false;
    static std::mutex setupMutex{};
    setupMutex.lock();
    if (setup) {
        setupMutex.unlock();
        return;
    }
    setup = true;
    setupMutex.unlock();

    spdlog::init_thread_pool(THREAD_QUEUE_LENGTH, 1);
    spdlog::sink_ptr console_sink =
      std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
#if SPDLOG_ACTIVE_LEVEL < SPDLOG_LEVEL_INFO
    console_sink->set_pattern("[%H:%M:%S %z] [%^%=9l%$] [%=21s] %v");
#else
    console_sink->set_pattern("[%H:%M:%S %z] [%^%=9l%$] [%=15s] %v");
#endif
    std::vector<spdlog::sink_ptr> sinks{ console_sink };
    // TODO: Add flag in compile flags
    std::shared_ptr<spdlog::logger> logger =
#if KOMPUTE_SPDLOG_ASYNC_LOGGING
          std::make_shared<spdlog::async_logger>(
            "",
            sinks.begin(),
            sinks.end(),
            spdlog::thread_pool(),
            spdlog::async_overflow_policy::block);
#else
          std::make_shared<spdlog::logger>(
            "",
            sinks.begin(),
            sinks.end());
#endif

    logger->set_level(getLogLevel());

    spdlog::set_default_logger(logger);
}

spdlog::level::level_enum
getLogLevel()
{
#if SPDLOG_ACTIVE_LEVEL == SPDLOG_LEVEL_TRACE
    return spdlog::level::trace;
#elif SPDLOG_ACTIVE_LEVEL == SPDLOG_LEVEL_DEBUG
    return spdlog::level::debug;
#elif SPDLOG_ACTIVE_LEVEL == SPDLOG_LEVEL_INFO
    return spdlog::level::info;
#elif SPDLOG_ACTIVE_LEVEL == SPDLOG_LEVEL_WARN
    return spdlog::level::warn;
#elif SPDLOG_ACTIVE_LEVEL == SPDLOG_LEVEL_ERROR
    return spdlog::level::error;
#elif SPDLOG_ACTIVE_LEVEL == SPDLOG_LEVEL_CRITICAL
    return spdlog::level::critical;
#else
    return spdlog::level::off;
#endif
}

void
setLogLevel(const spdlog::level::level_enum level)
{
    spdlog::default_logger()->set_level(level);
}
#endif // !KOMPUTE_OPT_USE_SPDLOG
} // namespace logger

#endif
