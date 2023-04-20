#pragma once

// OpenCL Utils includes
#include "OpenCLUtils_Export.h"

// OpenCL includes
#include <CL/cl.h>

UTILS_EXPORT
cl_ulong cl_util_get_event_duration(const cl_event event,
                                    const cl_profiling_info start,
                                    const cl_profiling_info end,
                                    cl_int* const error);