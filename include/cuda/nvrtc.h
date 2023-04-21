//
// NVIDIA_COPYRIGHT_BEGIN
//
// Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
// NVIDIA_COPYRIGHT_END
//

#ifndef __NVRTC_H__
#define __NVRTC_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <stdlib.h>


/*************************************************************************//**
 *
 * \defgroup error Error Handling
 *
 * NVRTC defines the following enumeration type and function for API call
 * error handling.
 *
 ****************************************************************************/


/**
 * \ingroup error
 * \brief   The enumerated type nvrtcResult defines API call result codes.
 *          NVRTC API functions return nvrtcResult to indicate the call
 *          result.
 */
typedef enum {
  NVRTC_SUCCESS = 0,
  NVRTC_ERROR_OUT_OF_MEMORY = 1,
  NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
  NVRTC_ERROR_INVALID_INPUT = 3,
  NVRTC_ERROR_INVALID_PROGRAM = 4,
  NVRTC_ERROR_INVALID_OPTION = 5,
  NVRTC_ERROR_COMPILATION = 6,
  NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
  NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
  NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
  NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
  NVRTC_ERROR_INTERNAL_ERROR = 11
} nvrtcResult;


/**
 * \ingroup error
 * \brief   nvrtcGetErrorString is a helper function that returns a string
 *          describing the given nvrtcResult code, e.g., NVRTC_SUCCESS to
 *          \c "NVRTC_SUCCESS".
 *          For unrecognized enumeration values, it returns
 *          \c "NVRTC_ERROR unknown".
 *
 * \param   [in] result CUDA Runtime Compilation API result code.
 * \return  Message string for the given #nvrtcResult code.
 */
const char *nvrtcGetErrorString(nvrtcResult result);


/*************************************************************************//**
 *
 * \defgroup query General Information Query
 *
 * NVRTC defines the following function for general information query.
 *
 ****************************************************************************/


/**
 * \ingroup query
 * \brief   nvrtcVersion sets the output parameters \p major and \p minor
 *          with the CUDA Runtime Compilation version number.
 *
 * \param   [out] major CUDA Runtime Compilation major version number.
 * \param   [out] minor CUDA Runtime Compilation minor version number.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *
 */
nvrtcResult nvrtcVersion(int *major, int *minor);


/**
 * \ingroup query
 * \brief   nvrtcGetNumSupportedArchs sets the output parameter \p numArchs 
 *          with the number of architectures supported by NVRTC. This can 
 *          then be used to pass an array to ::nvrtcGetSupportedArchs to
 *          get the supported architectures.
 *
 * \param   [out] numArchs number of supported architectures.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *
 * see    ::nvrtcGetSupportedArchs
 */
nvrtcResult nvrtcGetNumSupportedArchs(int* numArchs);


/**
 * \ingroup query
 * \brief   nvrtcGetSupportedArchs populates the array passed via the output parameter 
 *          \p supportedArchs with the architectures supported by NVRTC. The array is
 *          sorted in the ascending order. The size of the array to be passed can be
 *          determined using ::nvrtcGetNumSupportedArchs.
 *
 * \param   [out] supportedArchs sorted array of supported architectures.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *
 * see    ::nvrtcGetNumSupportedArchs
 */
nvrtcResult nvrtcGetSupportedArchs(int* supportedArchs);


/*************************************************************************//**
 *
 * \defgroup compilation Compilation
 *
 * NVRTC defines the following type and functions for actual compilation.
 *
 ****************************************************************************/


/**
 * \ingroup compilation
 * \brief   nvrtcProgram is the unit of compilation, and an opaque handle for
 *          a program.
 *
 * To compile a CUDA program string, an instance of nvrtcProgram must be
 * created first with ::nvrtcCreateProgram, then compiled with
 * ::nvrtcCompileProgram.
 */
typedef struct _nvrtcProgram *nvrtcProgram;


/**
 * \ingroup compilation
 * \brief   nvrtcCreateProgram creates an instance of nvrtcProgram with the
 *          given input parameters, and sets the output parameter \p prog with
 *          it.
 *
 * \param   [out] prog         CUDA Runtime Compilation program.
 * \param   [in]  src          CUDA program source.
 * \param   [in]  name         CUDA program name.\n
 *                             \p name can be \c NULL; \c "default_program" is
 *                             used when \p name is \c NULL or "".
 * \param   [in]  numHeaders   Number of headers used.\n
 *                             \p numHeaders must be greater than or equal to 0.
 * \param   [in]  headers      Sources of the headers.\n
 *                             \p headers can be \c NULL when \p numHeaders is
 *                             0.
 * \param   [in]  includeNames Name of each header by which they can be
 *                             included in the CUDA program source.\n
 *                             \p includeNames can be \c NULL when \p numHeaders
 *                             is 0.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_OUT_OF_MEMORY \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_PROGRAM_CREATION_FAILURE \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcDestroyProgram
 */
nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog,
                               const char *src,
                               const char *name,
                               int numHeaders,
                               const char * const *headers,
                               const char * const *includeNames);


/**
 * \ingroup compilation
 * \brief   nvrtcDestroyProgram destroys the given program.
 *
 * \param    [in] prog CUDA Runtime Compilation program.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcCreateProgram
 */
nvrtcResult nvrtcDestroyProgram(nvrtcProgram *prog);


/**
 * \ingroup compilation
 * \brief   nvrtcCompileProgram compiles the given program.
 *
 * \param   [in] prog       CUDA Runtime Compilation program.
 * \param   [in] numOptions Number of compiler options passed.
 * \param   [in] options    Compiler options in the form of C string array.\n
 *                          \p options can be \c NULL when \p numOptions is 0.
 *
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_OUT_OF_MEMORY \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_OPTION \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_COMPILATION \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_BUILTIN_OPERATION_FAILURE \endlink
 *
 * It supports compile options listed in \ref options.
 */
nvrtcResult nvrtcCompileProgram(nvrtcProgram prog,
                                int numOptions, const char * const *options);


/**
 * \ingroup compilation
 * \brief   nvrtcGetPTXSize sets \p ptxSizeRet with the size of the PTX
 *          generated by the previous compilation of \p prog (including the
 *          trailing \c NULL).
 *
 * \param   [in]  prog       CUDA Runtime Compilation program.
 * \param   [out] ptxSizeRet Size of the generated PTX (including the trailing
 *                           \c NULL).
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetPTX
 */
nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t *ptxSizeRet);


/**
 * \ingroup compilation
 * \brief   nvrtcGetPTX stores the PTX generated by the previous compilation
 *          of \p prog in the memory pointed by \p ptx.
 *
 * \param   [in]  prog CUDA Runtime Compilation program.
 * \param   [out] ptx  Compiled result.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetPTXSize
 */
nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char *ptx);


/**
 * \ingroup compilation
 * \brief   nvrtcGetCUBINSize sets \p cubinSizeRet with the size of the cubin
 *          generated by the previous compilation of \p prog. The value of
 *          cubinSizeRet is set to 0 if the value specified to \c -arch is a
 *          virtual architecture instead of an actual architecture.
 *
 * \param   [in]  prog       CUDA Runtime Compilation program.
 * \param   [out] cubinSizeRet Size of the generated cubin.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetCUBIN
 */
nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t *cubinSizeRet);


/**
 * \ingroup compilation
 * \brief   nvrtcGetCUBIN stores the cubin generated by the previous compilation
 *          of \p prog in the memory pointed by \p cubin. No cubin is available
 *          if the value specified to \c -arch is a virtual architecture instead
 *          of an actual architecture.
 *
 * \param   [in]  prog CUDA Runtime Compilation program.
 * \param   [out] cubin  Compiled and assembled result.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetCUBINSize
 */
nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char *cubin);

/**
 * \ingroup compilation
 * \brief   nvrtcGetNVVMSize sets \p nvvmSizeRet with the size of the NVVM
 *          generated by the previous compilation of \p prog. The value of
 *          nvvmSizeRet is set to 0 if the program was not compiled with 
 *          \c -dlto.
 *
 * \param   [in]  prog       CUDA Runtime Compilation program.
 * \param   [out] nvvmSizeRet Size of the generated NVVM.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetNVVM
 */
nvrtcResult nvrtcGetNVVMSize(nvrtcProgram prog, size_t *nvvmSizeRet);


/**
 * \ingroup compilation
 * \brief   nvrtcGetNVVM stores the NVVM generated by the previous compilation
 *          of \p prog in the memory pointed by \p nvvm.
 *          The program must have been compiled with -dlto,
 *          otherwise will return an error.
 *
 * \param   [in]  prog CUDA Runtime Compilation program.
 * \param   [out] nvvm Compiled result.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetNVVMSize
 */
nvrtcResult nvrtcGetNVVM(nvrtcProgram prog, char *nvvm);

/**
 * \ingroup compilation
 * \brief   nvrtcGetProgramLogSize sets \p logSizeRet with the size of the
 *          log generated by the previous compilation of \p prog (including the
 *          trailing \c NULL).
 *
 * Note that compilation log may be generated with warnings and informative
 * messages, even when the compilation of \p prog succeeds.
 *
 * \param   [in]  prog       CUDA Runtime Compilation program.
 * \param   [out] logSizeRet Size of the compilation log
 *                           (including the trailing \c NULL).
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetProgramLog
 */
nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t *logSizeRet);


/**
 * \ingroup compilation
 * \brief   nvrtcGetProgramLog stores the log generated by the previous
 *          compilation of \p prog in the memory pointed by \p log.
 *
 * \param   [in]  prog CUDA Runtime Compilation program.
 * \param   [out] log  Compilation log.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetProgramLogSize
 */
nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char *log);


/**
 * \ingroup compilation
 * \brief   nvrtcAddNameExpression notes the given name expression
 *          denoting the address of a __global__ function 
 *          or __device__/__constant__ variable.
 *
 * The identical name expression string must be provided on a subsequent
 * call to nvrtcGetLoweredName to extract the lowered name.
 * \param   [in]  prog CUDA Runtime Compilation program.
 * \param   [in] name_expression constant expression denoting the address of
 *               a __global__ function or __device__/__constant__ variable.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION \endlink
 *
 * \see     ::nvrtcGetLoweredName
 */
nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog,
                                   const char * const name_expression);

/**
 * \ingroup compilation
 * \brief   nvrtcGetLoweredName extracts the lowered (mangled) name
 *          for a __global__ function or __device__/__constant__ variable,
 *          and updates *lowered_name to point to it. The memory containing
 *          the name is released when the NVRTC program is destroyed by 
 *          nvrtcDestroyProgram.
 *          The identical name expression must have been previously
 *          provided to nvrtcAddNameExpression.
 *
 * \param   [in]  prog CUDA Runtime Compilation program.
 * \param   [in] name_expression constant expression denoting the address of 
 *               a __global__ function or __device__/__constant__ variable.
 * \param   [out] lowered_name initialized by the function to point to a
 *               C string containing the lowered (mangled)
 *               name corresponding to the provided name expression.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID \endlink
 *
 * \see     ::nvrtcAddNameExpression
 */
nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog,
                                const char *const name_expression,
                                const char** lowered_name);


/**
 * \defgroup options Supported Compile Options
 *
 * NVRTC supports the compile options below.
 * Option names with two preceding dashs (\c --) are long option names and
 * option names with one preceding dash (\c -) are short option names.
 * Short option names can be used instead of long option names.
 * When a compile option takes an argument, an assignment operator (\c =)
 * is used to separate the compile option argument from the compile option
 * name, e.g., \c "--gpu-architecture=compute_60".
 * Alternatively, the compile option name and the argument can be specified in
 * separate strings without an assignment operator, .e.g,
 * \c "--gpu-architecture" \c "compute_60".
 * Single-character short option names, such as \c -D, \c -U, and \c -I, do
 * not require an assignment operator, and the compile option name and the
 * argument can be present in the same string with or without spaces between
 * them.
 * For instance, \c "-D=<def>", \c "-D<def>", and \c "-D <def>" are all
 * supported.
 *
 * The valid compiler options are:
 *
 *   - Compilation targets
 *     - \c --gpu-architecture=\<arch\> (\c -arch)\n
 *       Specify the name of the class of GPU architectures for which the
 *       input must be compiled.\n
 *       - Valid <c>\<arch\></c>s:
 *         - \c compute_35
 *         - \c compute_37
 *         - \c compute_50
 *         - \c compute_52
 *         - \c compute_53
 *         - \c compute_60
 *         - \c compute_61
 *         - \c compute_62
 *         - \c compute_70
 *         - \c compute_72
 *         - \c compute_75
 *         - \c compute_80
 *         - \c sm_35
 *         - \c sm_37
 *         - \c sm_50
 *         - \c sm_52
 *         - \c sm_53
 *         - \c sm_60
 *         - \c sm_61
 *         - \c sm_62
 *         - \c sm_70
 *         - \c sm_72
 *         - \c sm_75
 *         - \c sm_80
 *       - Default: \c compute_52
 *   - Separate compilation / whole-program compilation
 *     - \c --device-c (\c -dc)\n
 *       Generate relocatable code that can be linked with other relocatable
 *       device code.  It is equivalent to --relocatable-device-code=true.
 *     - \c --device-w (\c -dw)\n
 *       Generate non-relocatable code.  It is equivalent to
 *       \c --relocatable-device-code=false.
 *     - \c --relocatable-device-code={true|false} (\c -rdc)\n
 *       Enable (disable) the generation of relocatable device code.
 *       - Default: \c false
 *     - \c --extensible-whole-program (\c -ewp)\n
 *       Do extensible whole program compilation of device code.
 *       - Default: \c false
 *   - Debugging support
 *     - \c --device-debug (\c -G)\n
 *       Generate debug information.
 *     - \c --generate-line-info (\c -lineinfo)\n
 *       Generate line-number information.
 *   - Code generation
 *     - \c --ptxas-options \<options\> (\c -Xptxas)\n
 *     - \c --ptxas-options=\<options\> \n
 *       Specify options directly to ptxas, the PTX optimizing assembler.
 *     - \c --maxrregcount=\<N\> (\c -maxrregcount)\n
 *       Specify the maximum amount of registers that GPU functions can use.
 *       Until a function-specific limit, a higher value will generally
 *       increase the performance of individual GPU threads that execute this
 *       function.  However, because thread registers are allocated from a
 *       global register pool on each GPU, a higher value of this option will
 *       also reduce the maximum thread block size, thereby reducing the amount
 *       of thread parallelism.  Hence, a good maxrregcount value is the result
 *       of a trade-off.  If this option is not specified, then no maximum is
 *       assumed.  Value less than the minimum registers required by ABI will
 *       be bumped up by the compiler to ABI minimum limit.
 *     - \c --ftz={true|false} (\c -ftz)\n
 *       When performing single-precision floating-point operations, flush
 *       denormal values to zero or preserve denormal values.
 *       \c --use_fast_math implies \c --ftz=true.
 *       - Default: \c false
 *     - \c --prec-sqrt={true|false} (\c -prec-sqrt)\n
 *       For single-precision floating-point square root, use IEEE
 *       round-to-nearest mode or use a faster approximation.
 *       \c --use_fast_math implies \c --prec-sqrt=false.
 *       - Default: \c true
 *     - \c --prec-div={true|false} (\c -prec-div)\n
 *       For single-precision floating-point division and reciprocals, use IEEE
 *       round-to-nearest mode or use a faster approximation.
 *       \c --use_fast_math implies \c --prec-div=false.
 *       - Default: \c true
 *     - \c --fmad={true|false} (\c -fmad)\n
 *       Enables (disables) the contraction of floating-point multiplies and
 *       adds/subtracts into floating-point multiply-add operations (FMAD,
 *       FFMA, or DFMA).  \c --use_fast_math implies \c --fmad=true.
 *       - Default: \c true
 *     - \c --use_fast_math (\c -use_fast_math)\n
 *       Make use of fast math operations.
 *       \c --use_fast_math implies \c --ftz=true \c --prec-div=false
 *       \c --prec-sqrt=false \c --fmad=true.
 *     - \c --extra-device-vectorization (\c -extra-device-vectorization)\n
 *       Enables more aggressive device code vectorization in the NVVM optimizer.
 *     - \c --modify-stack-limit={true|false} (\c -modify-stack-limit)\n
 *       On Linux, during compilation, use \c setrlimit() to increase stack size 
 *       to maximum allowed. The limit is reset to the previous value at the
 *       end of compilation.
 *       Note: \c setrlimit() changes the value for the entire process.
 *       - Default: \c true
 *     - \c --dlink-time-opt (\c -dlto)\n
 *       Generate intermediate code for later link-time optimization.
 *       It implies \c -rdc=true.  
 *       Note: when this is used the nvvmGetNVVM API should be used, 
 *       as PTX or Cubin will not be generated.
 *   - Preprocessing
 *     - \c --define-macro=\<def\> (\c -D)\n
 *       \c \<def\> can be either \c \<name\> or \c \<name=definitions\>.
 *       - \c \<name\> \n
 *         Predefine \c \<name\> as a macro with definition \c 1.
 *       - \c \<name\>=\<definition\> \n
 *         The contents of \c \<definition\> are tokenized and preprocessed
 *         as if they appeared during translation phase three in a \c \#define
 *         directive.  In particular, the definition will be truncated by
 *         embedded new line characters.
 *     - \c --undefine-macro=\<def\> (\c -U)\n
 *       Cancel any previous definition of \c \<def\>.
 *     - \c --include-path=\<dir\> (\c -I)\n
 *       Add the directory \c \<dir\> to the list of directories to be
 *       searched for headers.  These paths are searched after the list of
 *       headers given to ::nvrtcCreateProgram.
 *     - \c --pre-include=\<header\> (\c -include)\n
 *       Preinclude \c \<header\> during preprocessing.
 *     - \c --no-source-include (\c -no-source-include)
 *       The preprocessor by default adds the directory of each input sources
 *       to the include path. This option disables this feature and only
 *       considers the path specified explicitly.
 *   - Language Dialect
 *     - \c --std={c++03|c++11|c++14|c++17} (\c -std={c++11|c++14|c++17})\n
 *       Set language dialect to C++03, C++11, C++14 or C++17
 *     - \c --builtin-move-forward={true|false} (\c -builtin-move-forward)\n
 *       Provide builtin definitions of \c std::move and \c std::forward,
 *       when C++11 language dialect is selected.
 *       - Default: \c true
 *     - \c --builtin-initializer-list={true|false}
 *       (\c -builtin-initializer-list)\n
 *       Provide builtin definitions of \c std::initializer_list class and
 *       member functions when C++11 language dialect is selected.
 *       - Default: \c true
 *   - Misc.
 *     - \c --disable-warnings (\c -w)\n
 *       Inhibit all warning messages.
 *     - \c --restrict (\c -restrict)\n
 *       Programmer assertion that all kernel pointer parameters are restrict
 *       pointers.
 *     - \c --device-as-default-execution-space
 *       (\c -default-device)\n
 *       Treat entities with no execution space annotation as \c __device__
 *       entities.
 *     - \c --device-int128 (\c -device-int128)\n
 *       Allow the \c __int128 type in device code. Also causes the macro \c __CUDACC_RTC_INT128__
 *       to be defined.
 *     - \c --optimization-info=\<kind\> (\c -opt-info)\n
 *       Provide optimization reports for the specified kind of optimization.
 *       The following kind tags are supported:
 *         - \c inline : emit a remark when a function is inlined.
 *     - \c --version-ident={true|false} (\c -dQ)\n
 *       Embed used compiler's version info into generated PTX/CUBIN 
 *       - Default: \c false
 *     - \c --display-error-number (\c -err-no)\n
 *       Display diagnostic number for warning messages. (Default)
 *     - \c --no-display-error-number (\c -no-err-no)\n
 *       Disables the display of a diagnostic number for warning messages.
 *     - \c --diag-error=<error-number>,... (\c -diag-error)\n
 *       Emit error for specified diagnostic message number(s). Message numbers can be separated by comma.
 *     - \c --diag-suppress=<error-number>,... (\c -diag-suppress)\n
 *       Suppress specified diagnostic message number(s). Message numbers can be separated by comma.
 *     - \c --diag-warn=<error-number>,... (\c -diag-warn)\n
 *       Emit warning for specified diagnostic message number(s). Message numbers can be separated by comma.
 *
 */


#ifdef __cplusplus
}
#endif /* __cplusplus */


/* The utility function 'nvrtcGetTypeName' is not available by default. Define
   the macro 'NVRTC_GET_TYPE_NAME' to a non-zero value to make it available.
*/
   
#if NVRTC_GET_TYPE_NAME || __DOXYGEN_ONLY__

#if NVRTC_USE_CXXABI || __clang__ || __GNUC__ || __DOXYGEN_ONLY__
#include <cxxabi.h>
#include <cstdlib>

#elif defined(_WIN32)
#include <Windows.h>
#include <DbgHelp.h>
#endif /* NVRTC_USE_CXXABI || __clang__ || __GNUC__ */


#include <string>
#include <typeinfo>

template <typename T> struct __nvrtcGetTypeName_helper_t { };

/*************************************************************************//**
 *
 * \defgroup hosthelper Host Helper
 *
 * NVRTC defines the following functions for easier interaction with host code.
 *
 ****************************************************************************/

/**
 * \ingroup hosthelper
 * \brief   nvrtcGetTypeName stores the source level name of a type in the given 
 *          std::string location. 
 *
 * This function is only provided when the macro NVRTC_GET_TYPE_NAME is
 * defined with a non-zero value. It uses abi::__cxa_demangle or UnDecorateSymbolName
 * function calls to extract the type name, when using gcc/clang or cl.exe compilers,
 * respectively. If the name extraction fails, it will return NVRTC_INTERNAL_ERROR,
 * otherwise *result is initialized with the extracted name.
 * 
 * Windows-specific notes:
 * - nvrtcGetTypeName() is not multi-thread safe because it calls UnDecorateSymbolName(), 
 *   which is not multi-thread safe.
 * - The returned string may contain Microsoft-specific keywords such as __ptr64 and __cdecl.
 *
 * \param   [in] tinfo: reference to object of type std::type_info for a given type.
 * \param   [in] result: pointer to std::string in which to store the type name.
 * \return
 *  - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *  - \link #nvrtcResult NVRTC_ERROR_INTERNAL_ERROR \endlink
 *
 */
inline nvrtcResult nvrtcGetTypeName(const std::type_info &tinfo, std::string *result)
{
#if USE_CXXABI || __clang__ || __GNUC__
  const char *name = tinfo.name();
  int status;
  char *undecorated_name = abi::__cxa_demangle(name, 0, 0, &status);
  if (status == 0) {
    *result = undecorated_name;
    free(undecorated_name);
    return NVRTC_SUCCESS;
  }
#elif defined(_WIN32)
  const char *name = tinfo.raw_name();
  if (!name || *name != '.') {
    return NVRTC_ERROR_INTERNAL_ERROR;
  }
  char undecorated_name[4096];
  //name+1 skips over the '.' prefix
  if(UnDecorateSymbolName(name+1, undecorated_name,
                          sizeof(undecorated_name) / sizeof(*undecorated_name),
                           //note: doesn't seem to work correctly without UNDNAME_NO_ARGUMENTS.
                           UNDNAME_NO_ARGUMENTS | UNDNAME_NAME_ONLY ) ) {
    *result = undecorated_name;
    return NVRTC_SUCCESS;
  }
#endif  /* USE_CXXABI || __clang__ || __GNUC__ */

  return NVRTC_ERROR_INTERNAL_ERROR;
}

/**
 * \ingroup hosthelper
 * \brief   nvrtcGetTypeName stores the source level name of the template type argument
 *          T in the given std::string location.
 *
 * This function is only provided when the macro NVRTC_GET_TYPE_NAME is
 * defined with a non-zero value. It uses abi::__cxa_demangle or UnDecorateSymbolName
 * function calls to extract the type name, when using gcc/clang or cl.exe compilers,
 * respectively. If the name extraction fails, it will return NVRTC_INTERNAL_ERROR,
 * otherwise *result is initialized with the extracted name.
 * 
 * Windows-specific notes:
 * - nvrtcGetTypeName() is not multi-thread safe because it calls UnDecorateSymbolName(), 
 *   which is not multi-thread safe.
 * - The returned string may contain Microsoft-specific keywords such as __ptr64 and __cdecl.
 *
 * \param   [in] result: pointer to std::string in which to store the type name.
 * \return
 *  - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *  - \link #nvrtcResult NVRTC_ERROR_INTERNAL_ERROR \endlink
 *
 */
 
template <typename T>
nvrtcResult nvrtcGetTypeName(std::string *result)
{
  nvrtcResult res = nvrtcGetTypeName(typeid(__nvrtcGetTypeName_helper_t<T>), 
                                     result);
  if (res != NVRTC_SUCCESS) 
    return res;

  std::string repr = *result;
  std::size_t idx = repr.find("__nvrtcGetTypeName_helper_t");
  idx = (idx != std::string::npos) ? repr.find("<", idx) : idx;
  std::size_t last_idx = repr.find_last_of('>');
  if (idx == std::string::npos || last_idx == std::string::npos) {
    return NVRTC_ERROR_INTERNAL_ERROR;
  }
  ++idx;
  *result = repr.substr(idx, last_idx - idx);
  return NVRTC_SUCCESS;
}

#endif  /* NVRTC_GET_TYPE_NAME */

#endif /* __NVRTC_H__ */
