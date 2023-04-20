#pragma once

// OpenCL Utils includes
#include "OpenCLUtils_Export.h"

// OpenCL includes
#include <CL/cl.h>

// read all the text file contents securely in ANSI C89
// return pointer to C-string with file contents
// can handle streams with no known size and no support for fseek
// based on https://stackoverflow.com/questions/14002954/ by Nominal Animal
UTILS_EXPORT
char* cl_util_read_text_file(const char* const filename, size_t* const length,
                             cl_int* const error);

// read all the binary file contents securely in ANSI C89
// return pointer to file contents
// can handle streams with no known size and no support for fseek
// based on https://stackoverflow.com/questions/14002954/ by Nominal Animal
UTILS_EXPORT
unsigned char* cl_util_read_binary_file(const char* const filename,
                                        size_t* const length,
                                        cl_int* const error);

// write binaries of OpenCL compiled program
// binaries are written as separate files for each device
// with file name "(program_file_name)_(name of device).bin"
// based on variant of Logan
// http://logan.tw/posts/2014/11/22/pre-compile-the-opencl-kernel-program-part-2/
UTILS_EXPORT
cl_int cl_util_write_binaries(const cl_program program,
                              const char* const program_file_name);

// read binaries of OpenCL compiled program
// from files of file names "(program_file_name)_(name of device).bin"
UTILS_EXPORT
cl_program cl_util_read_binaries(const cl_context context,
                                 const cl_device_id* const devices,
                                 const cl_uint num_devices,
                                 const char* const program_file_name,
                                 cl_int* const error);
