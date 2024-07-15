#pragma once

#include <stdint.h>

#include "ggml.h"

#include "QnnCommon.h"
#include "QnnInterface.h"
#include "QnnTypes.h"
#include "System/QnnSystemInterface.h"

#define QNN_LOGBUF_LEN 4096

namespace qnn {
void internal_log(ggml_log_level level, const char *file, const char *func, int line, const char *format, ...);

void sdk_logcallback(const char *fmt, QnnLog_Level_t level, uint64_t timestamp, va_list argp);
} // namespace qnn

// =================================================================================================
//
//  QNN backend internal log function
//
// =================================================================================================
#define QNN_LOG_ERROR(...) qnn::internal_log(GGML_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#define QNN_LOG_WARN(...) qnn::internal_log(GGML_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#define QNN_LOG_INFO(...) qnn::internal_log(GGML_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#ifdef NDEBUG
#define ENABLE_QNNBACKEND_DEBUG 0 // for troubleshooting QNN backend
#define ENABLE_QNNSDK_LOG 0       // enable/disable QNN SDK's internal log
#else
#define ENABLE_QNNBACKEND_DEBUG 1 // for troubleshooting QNN backend
#define ENABLE_QNNSDK_LOG 1       // enable/disable QNN SDK's internal log
#endif

#if ENABLE_QNNBACKEND_DEBUG
#define QNN_LOG_DEBUG(...) qnn::internal_log(GGML_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define QNN_LOG_DEBUG(...)
#endif
