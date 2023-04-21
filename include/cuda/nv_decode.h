/*
 * NVIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA_COPYRIGHT_END
 */

/*
nv_decode.h -- API for the CUDA C++ name demangler.
*/

/* Avoid including these declarations more than once: */
#ifndef DECODE_H
#define DECODE_H 1

#ifdef __cplusplus
extern "C" {
#endif

/* Cuda C++ demangling API.

   Parameters:
     id: Input mangled string
     output_buffer: Pointer to where the demangled buffer will
                    be stored. This memory must be allocated with malloc.
                    If output-buffer is NULL, memory will be malloc'd to
                    store the demangled name and returned through the 
                    function return value.
                    If the output-buffer is too small, it is expanded using
                    realloc.
     length: It is necessary to provide the size of the output buffer if the user
             is providing pre-allocated memory. This is needed by the demangler 
             in case the size needs to be reallocated.
             If the length is non-null, the length of the demangled buffer
             is placed in length.
     status: *status is set to one of the following values. 0 - The
             demangling operation succeeded; -1 - A memory allocation
             failure occurred. -2 - Not a valid mangled id. -3 - An
             input validation failure has occurred (one or more
             arguments are invalid).

   Return Value: A pointer to the start of the NUL-terminated demangled name,
                 or NULL if the demangling fails. The caller is responsible for
                 deallocating this memory using free.

   Note: This function is thread-safe.
*/

char* __cu_demangle(const char *id,
	                char *output_buffer,
	                size_t *length,
	                int *status);

#ifdef __cplusplus
}
#endif
#endif /* ifndef DECODE_H */
