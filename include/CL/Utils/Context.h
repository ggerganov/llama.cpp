#pragma once

// OpenCL Utils includes
#include "OpenCLUtils_Export.h"

// OpenCL includes
#include <CL/cl.h>

// STL includes
#include <time.h>

UTILS_EXPORT
cl_context cl_util_get_context(const cl_uint plat_id, const cl_uint dev_id,
                               const cl_device_type type, cl_int* const error);
UTILS_EXPORT
cl_device_id cl_util_get_device(const cl_uint plat_id, const cl_uint dev_id,
                                const cl_device_type type, cl_int* const error);

UTILS_EXPORT
cl_int cl_util_print_device_info(const cl_device_id device);

UTILS_EXPORT
char* cl_util_get_device_info(const cl_device_id device,
                              const cl_device_info info, cl_int* const error);
UTILS_EXPORT
char* cl_util_get_platform_info(const cl_platform_id platform,
                                const cl_platform_info info,
                                cl_int* const error);

// build program and show log if build is not successful
UTILS_EXPORT
cl_int cl_util_build_program(const cl_program pr, const cl_device_id dev,
                             const char* const opt);

#define GET_CURRENT_TIMER(time)                                                \
    struct timespec time;                                                      \
    timespec_get(&time, TIME_UTC);                                             \
    {                                                                          \
    }

#define TIMER_DIFFERENCE(dt, time1, time2)                                     \
    {                                                                          \
        dt = (time2.tv_sec - time1.tv_sec) * 1000000000                        \
            + (time2.tv_nsec - time1.tv_nsec);                                 \
    }

#define START_TIMER GET_CURRENT_TIMER(start_timer1)
#define STOP_TIMER(dt)                                                         \
    GET_CURRENT_TIMER(stop_timer2)                                             \
    TIMER_DIFFERENCE(dt, start_timer1, stop_timer2)
