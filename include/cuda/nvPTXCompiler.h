/*
 * NVIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2016-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * NVIDIA_COPYRIGHT_END
 */

#ifndef nvPTXCompiler_INCLUDED
#define nvPTXCompiler_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

/* --- Dependency --- */
#include <stddef.h>   /* For size_t */

/*************************************************************************//**
 *
 * \defgroup handle PTX-Compiler Handle
 *
 ****************************************************************************/


/**
 * \ingroup handle
 * \brief   nvPTXCompilerHandle represents a handle to the PTX Compiler.
 *
 * To compile a PTX program string, an instance of nvPTXCompiler
 * must be created and the handle to it must be obtained using the
 * API nvPTXCompilerCreate(). Then the compilation can be done
 * using the API nvPTXCompilerCompile().
 *
 */
typedef struct nvPTXCompiler* nvPTXCompilerHandle;

/**
 *
 * \defgroup error Error codes
 *
 */

/** \ingroup error
 *
 * \brief     The nvPTXCompiler APIs return the nvPTXCompileResult codes to indicate the call result
 */

typedef enum {

    /* Indicates the API completed successfully */
    NVPTXCOMPILE_SUCCESS = 0,

    /* Indicates an invalid nvPTXCompilerHandle was passed to the API */
    NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE = 1,

    /* Indicates invalid inputs were given to the API  */
    NVPTXCOMPILE_ERROR_INVALID_INPUT = 2,

    /* Indicates that the compilation of the PTX program failed */
    NVPTXCOMPILE_ERROR_COMPILATION_FAILURE = 3,

    /* Indicates that something went wrong internally */
    NVPTXCOMPILE_ERROR_INTERNAL = 4,

    /* Indicates that the API was unable to allocate memory */
    NVPTXCOMPILE_ERROR_OUT_OF_MEMORY = 5,

    /* Indicates that the handle was passed to an API which expected */
    /* the nvPTXCompilerCompile() to have been called previously */
    NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE = 6,

    /* Indicates that the PTX version encountered in the PTX is not */
    /* supported by the current compiler */
    NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION = 7,
} nvPTXCompileResult;

/* ----------------------------- PTX Compiler APIs ---------------------------- */

/**
 *
 * \defgroup versioning API Versioning
 *
 * The PTX compiler APIs are versioned so that any new features or API
 * changes can be done by bumping up the API version.
 */

/** \ingroup versioning
 *
 * \brief            Queries the current \p major and \p minor version of
 *                   PTX Compiler APIs being used
 *
 * \param            [out] major   Major version of the PTX Compiler APIs
 * \param            [out] minor   Minor version of the PTX Compiler APIs
 * \note                           The version of PTX Compiler APIs follows the CUDA Toolkit versioning.
 *                                 The PTX ISA version supported by a PTX Compiler API version is listed
 *                                 <a href="https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes">here</a>.
 *
 * \return
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
 */
nvPTXCompileResult nvPTXCompilerGetVersion (unsigned int* major, unsigned int* minor);

/**
 *
 * \defgroup compilation Compilation APIs
 *
 */

/** \ingroup compilation
 *
 * \brief            Obtains the handle to an instance of the PTX compiler
 *                   initialized with the given PTX program \p ptxCode
 *
 * \param            [out] compiler  Returns a handle to PTX compiler initialized
 *                                   with the PTX program \p ptxCode
 * \param            [in] ptxCodeLen Size of the PTX program \p ptxCode passed as string
 * \param            [in] ptxCode    The PTX program which is to be compiled passed as string.
 *
 *
 * \return
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_OUT_OF_MEMORY \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
 */
nvPTXCompileResult nvPTXCompilerCreate (nvPTXCompilerHandle *compiler, size_t ptxCodeLen, const char* ptxCode);

/** \ingroup compilation
 *
 * \brief            Destroys and cleans the already created PTX compiler
 *
 * \param            [in] compiler  A handle to the PTX compiler which is to be destroyed
 *
 * \return
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_OUT_OF_MEMORY \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE \endlink
 *
 */
nvPTXCompileResult nvPTXCompilerDestroy (nvPTXCompilerHandle *compiler);

/** \ingroup compilation
 *
 * \brief          Compile a PTX program with the given compiler options
 *
 * \param            [in,out] compiler      A handle to PTX compiler initialized with the
 *                                          PTX program which is to be compiled.
 *                                          The compiled program can be accessed using the handle
 * \param            [in] numCompileOptions Length of the array \p compileOptions
 * \param            [in] compileOptions   Compiler options with which compilation should be done.
 *                                         The compiler options string is a null terminated character array.
 *                                         A valid list of compiler options is at
 *                                         <a href="http://docs.nvidia.com/cuda/ptx-compiler-api/index.html#compile-options">link</a>.
 * \note                                   --gpu-name (-arch) is a mandatory option.
 *
 * \return
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_OUT_OF_MEMORY \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_COMPILATION_FAILURE  \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION  \endlink
 *
 */
nvPTXCompileResult nvPTXCompilerCompile (nvPTXCompilerHandle compiler, int numCompileOptions, const char* const * compileOptions);

/** \ingroup compilation
 *
 * \brief            Obtains the size of the image of the compiled program
 *
 * \param            [in] compiler          A handle to PTX compiler on which nvPTXCompilerCompile() has been performed.
 * \param            [out] binaryImageSize  The size of the image of the compiled program
 *
 * \return
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE \endlink
 *
 * \note             nvPTXCompilerCompile() API should be invoked for the handle before calling this API.
 *                   Otherwise, NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE is returned.
 */
nvPTXCompileResult nvPTXCompilerGetCompiledProgramSize (nvPTXCompilerHandle compiler, size_t* binaryImageSize);

/** \ingroup compilation
 *
 * \brief            Obtains the image of the compiled program
 *
 * \param            [in] compiler          A handle to PTX compiler on which nvPTXCompilerCompile() has been performed.
 * \param            [out] binaryImage      The image of the compiled program.
 *                                         Client should allocate memory for \p binaryImage
 *
 * \return
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE \endlink
 *
 * \note             nvPTXCompilerCompile() API should be invoked for the handle before calling this API.
 *                   Otherwise, NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE is returned.
 *
 */

nvPTXCompileResult nvPTXCompilerGetCompiledProgram (nvPTXCompilerHandle compiler, void*   binaryImage);

/** \ingroup compilation
 *
 * \brief            Query the size of the error message that was seen previously for the handle
 *
 * \param            [in] compiler          A handle to PTX compiler on which nvPTXCompilerCompile() has been performed.
 * \param            [out] errorLogSize     The size of the error log in bytes which was produced
 *                                          in previous call to nvPTXCompilerCompiler().
 *
 * \return
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE \endlink
 *
 */
nvPTXCompileResult nvPTXCompilerGetErrorLogSize (nvPTXCompilerHandle compiler, size_t* errorLogSize);

/** \ingroup compilation
 *
 * \brief            Query the error message that was seen previously for the handle
 *
 * \param            [in] compiler         A handle to PTX compiler on which nvPTXCompilerCompile() has been performed.
 * \param            [out] errorLog        The error log which was produced in previous call to nvPTXCompilerCompiler().
 *                                         Clients should allocate memory for \p errorLog
 *
 * \return
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE \endlink
 *
 */
nvPTXCompileResult nvPTXCompilerGetErrorLog (nvPTXCompilerHandle compiler, char*   errorLog);

/** \ingroup compilation
 *
 * \brief            Query the size of the information message that was seen previously for the handle
 *
 * \param            [in] compiler        A handle to PTX compiler on which nvPTXCompilerCompile() has been performed.
 * \param            [out] infoLogSize    The size of the information log in bytes which was produced
 *                                         in previous call to nvPTXCompilerCompiler().
 *
 * \return
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE \endlink
 *
 */
nvPTXCompileResult nvPTXCompilerGetInfoLogSize (nvPTXCompilerHandle compiler, size_t* infoLogSize);

/** \ingroup compilation
 *
 * \brief           Query the information message that was seen previously for the handle
 *
 * \param            [in] compiler        A handle to PTX compiler on which nvPTXCompilerCompile() has been performed.
 * \param            [out] infoLog        The information log which was produced in previous call to nvPTXCompilerCompiler().
 *                                        Clients should allocate memory for \p infoLog
 *
 * \return
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
 *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE \endlink
 *
 */
nvPTXCompileResult nvPTXCompilerGetInfoLog (nvPTXCompilerHandle compiler, char*   infoLog);

#ifdef __cplusplus
}
#endif

#endif // nvPTXCompiler_INCLUDED
