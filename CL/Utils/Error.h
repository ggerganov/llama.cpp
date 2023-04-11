#pragma once

// OpenCL Utils includes
#include "OpenCLUtils_Export.h"

// OpenCL Utils includes
#include <CL/Utils/ErrorCodes.h>

// STL includes
#include <stdio.h> // fprintf

// OpenCL includes
#include <CL/cl.h>

// RET = function returns error code
// PAR = functions sets error code in the paremeter

#ifdef _DEBUG

#define OCLERROR_RET(func, err, label)                                         \
    do                                                                         \
    {                                                                          \
        err = func;                                                            \
        if (err != CL_SUCCESS)                                                 \
        {                                                                      \
            cl_util_print_error(err);                                          \
            fprintf(stderr, "on line %d, in file %s\n%s\n", __LINE__,          \
                    __FILE__, #func);                                          \
            goto label;                                                        \
        }                                                                      \
    } while (0)

#define OCLERROR_PAR(func, err, label)                                         \
    do                                                                         \
    {                                                                          \
        func;                                                                  \
        if (err != CL_SUCCESS)                                                 \
        {                                                                      \
            cl_util_print_error(err);                                          \
            fprintf(stderr, "on line %d, in file %s\n%s\n", __LINE__,          \
                    __FILE__, #func);                                          \
            goto label;                                                        \
        }                                                                      \
    } while (0)

#define MEM_CHECK(func, err, label)                                            \
    do                                                                         \
    {                                                                          \
        if ((func) == NULL)                                                    \
        {                                                                      \
            err = CL_OUT_OF_HOST_MEMORY;                                       \
            cl_util_print_error(err);                                          \
            fprintf(stderr, "on line %d, in file %s\n%s\n", __LINE__,          \
                    __FILE__, #func);                                          \
            goto label;                                                        \
        }                                                                      \
    } while (0)

#else

#define OCLERROR_RET(func, err, label)                                         \
    do                                                                         \
    {                                                                          \
        err = func;                                                            \
        if (err != CL_SUCCESS) goto label;                                     \
    } while (0)

#define OCLERROR_PAR(func, err, label)                                         \
    do                                                                         \
    {                                                                          \
        func;                                                                  \
        if (err != CL_SUCCESS) goto label;                                     \
    } while (0)

#define MEM_CHECK(func, err, label)                                            \
    do                                                                         \
    {                                                                          \
        if ((func) == NULL)                                                    \
        {                                                                      \
            err = CL_OUT_OF_HOST_MEMORY;                                       \
            goto label;                                                        \
        }                                                                      \
    } while (0)

#endif

UTILS_EXPORT
void cl_util_print_error(cl_int error);
