cmake_minimum_required(VERSION 3.14)  # for add_link_options and implicit target directories.
project("llama.cpp" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LLAMA_STANDALONE ON)

    # configure project version
    # TODO
else()
    set(LLAMA_STANDALONE OFF)
endif()

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    option(LLAMA_WASM_SINGLE_FILE "llama: embed WASM inside the generated llama.js" ON)
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()


#
# Option list
#

if (APPLE)
    set(LLAMA_METAL_DEFAULT ON)
else()
    set(LLAMA_METAL_DEFAULT OFF)
endif()

# general
option(BUILD_SHARED_LIBS                "build shared libraries"                                OFF)
option(LLAMA_STATIC                     "llama: static link libraries"                          OFF)
option(LLAMA_NATIVE                     "llama: enable -march=native flag"                      ON)
option(LLAMA_LTO                        "llama: enable link time optimization"                  OFF)

# debug
option(LLAMA_ALL_WARNINGS               "llama: enable all compiler warnings"                   ON)
option(LLAMA_ALL_WARNINGS_3RD_PARTY     "llama: enable all compiler warnings in 3rd party libs" OFF)
option(LLAMA_GPROF                      "llama: enable gprof"                                   OFF)

# sanitizers
option(LLAMA_SANITIZE_THREAD            "llama: enable thread sanitizer"                        OFF)
option(LLAMA_SANITIZE_ADDRESS           "llama: enable address sanitizer"                       OFF)
option(LLAMA_SANITIZE_UNDEFINED         "llama: enable undefined sanitizer"                     OFF)

# instruction set specific
if (LLAMA_NATIVE)
    set(INS_ENB OFF)
else()
    set(INS_ENB ON)
endif()

option(LLAMA_AVX                             "llama: enable AVX"                                ${INS_ENB})
option(LLAMA_AVX2                            "llama: enable AVX2"                               ${INS_ENB})
option(LLAMA_AVX512                          "llama: enable AVX512"                             OFF)
option(LLAMA_AVX512_VBMI                     "llama: enable AVX512-VBMI"                        OFF)
option(LLAMA_AVX512_VNNI                     "llama: enable AVX512-VNNI"                        OFF)
option(LLAMA_FMA                             "llama: enable FMA"                                ${INS_ENB})
# in MSVC F16C is implied with AVX2/AVX512
if (NOT MSVC)
    option(LLAMA_F16C                        "llama: enable F16C"                               ${INS_ENB})
endif()

if (WIN32)
    option(LLAMA_WIN_VER                     "llama: Windows Version"                           0x602)
endif()

# 3rd party libs
option(LLAMA_ACCELERATE                      "llama: enable Accelerate framework"               ON)
option(LLAMA_BLAS                            "llama: use BLAS"                                  OFF)
set(LLAMA_BLAS_VENDOR "Generic" CACHE STRING "llama: BLAS library vendor")
option(LLAMA_CUBLAS                          "llama: use CUDA"                                  OFF)
#option(LLAMA_CUDA_CUBLAS                     "llama: use cuBLAS for prompt processing"          OFF)
option(LLAMA_CUDA_FORCE_DMMV                 "llama: use dmmv instead of mmvq CUDA kernels"     OFF)
option(LLAMA_CUDA_FORCE_MMQ                  "llama: use mmq kernels instead of cuBLAS"         OFF)
set(LLAMA_CUDA_DMMV_X      "32" CACHE STRING "llama: x stride for dmmv CUDA kernels")
set(LLAMA_CUDA_MMV_Y        "1" CACHE STRING "llama: y block size for mmv CUDA kernels")
option(LLAMA_CUDA_F16                        "llama: use 16 bit floats for some calculations"   OFF)
set(LLAMA_CUDA_KQUANTS_ITER "2" CACHE STRING "llama: iters./thread per block for Q2_K/Q6_K")
set(LLAMA_CUDA_PEER_MAX_BATCH_SIZE "128" CACHE STRING
                                             "llama: max. batch size for using peer access")
option(LLAMA_HIPBLAS                         "llama: use hipBLAS"                               OFF)
option(LLAMA_HIP_UMA                         "llama: use HIP unified memory architecture"       OFF)
option(LLAMA_CLBLAST                         "llama: use CLBlast"                               OFF)
option(LLAMA_METAL                           "llama: use Metal"                                 ${LLAMA_METAL_DEFAULT})
option(LLAMA_METAL_NDEBUG                    "llama: disable Metal debugging"                   OFF)
option(LLAMA_METAL_SHADER_DEBUG              "llama: compile Metal with -fno-fast-math"         OFF)
option(LLAMA_MPI                             "llama: use MPI"                                   OFF)
option(LLAMA_QKK_64                          "llama: use super-block size of 64 for k-quants"   OFF)

option(LLAMA_BUILD_TESTS                     "llama: build tests"    ${LLAMA_STANDALONE})
option(LLAMA_BUILD_EXAMPLES                  "llama: build examples" ${LLAMA_STANDALONE})
option(LLAMA_BUILD_SERVER                    "llama: build server example"                      ON)

# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/scripts/build-info.cmake)

#
# Compile flags
#

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
include(CheckCXXCompilerFlag)

# enable libstdc++ assertions for debug builds
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions($<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
endif()

if (NOT MSVC)
    if (LLAMA_SANITIZE_THREAD)
        add_compile_options(-fsanitize=thread)
        link_libraries(-fsanitize=thread)
    endif()

    if (LLAMA_SANITIZE_ADDRESS)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries(-fsanitize=address)
    endif()

    if (LLAMA_SANITIZE_UNDEFINED)
        add_compile_options(-fsanitize=undefined)
        link_libraries(-fsanitize=undefined)
    endif()
endif()

if (APPLE AND LLAMA_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        add_compile_definitions(GGML_USE_ACCELERATE)
        add_compile_definitions(ACCELERATE_NEW_LAPACK)
        add_compile_definitions(ACCELERATE_LAPACK_ILP64)
        set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} ${ACCELERATE_FRAMEWORK})
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()

if (LLAMA_METAL)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(METAL_FRAMEWORK    Metal      REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit   REQUIRED)

    message(STATUS "Metal framework found")
    set(GGML_HEADERS_METAL ggml-metal.h)
    set(GGML_SOURCES_METAL ggml-metal.m)

    add_compile_definitions(GGML_USE_METAL)
    if (LLAMA_METAL_NDEBUG)
        add_compile_definitions(GGML_METAL_NDEBUG)
    endif()

    # get full path to the file
    #add_compile_definitions(GGML_METAL_DIR_KERNELS="${CMAKE_CURRENT_SOURCE_DIR}/")

    # copy ggml-metal.metal to bin directory
    configure_file(ggml-metal.metal ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-metal.metal COPYONLY)

    if (LLAMA_METAL_SHADER_DEBUG)
        # custom command to do the following:
        #   xcrun -sdk macosx metal    -fno-fast-math -c ggml-metal.metal -o ggml-metal.air
        #   xcrun -sdk macosx metallib                   ggml-metal.air   -o default.metallib
        #
        # note: this is the only way I found to disable fast-math in Metal. it's ugly, but at least it works
        #       disabling fast math is needed in order to pass tests/test-backend-ops
        # note: adding -fno-inline fixes the tests when using MTL_SHADER_VALIDATION=1
        # note: unfortunately, we have to call it default.metallib instead of ggml.metallib
        #       ref: https://github.com/ggerganov/whisper.cpp/issues/1720
        set(XC_FLAGS -fno-fast-math -fno-inline -g)
        if (LLAMA_QKK_64)
            set(XC_FLAGS ${XC_FLAGS} -DQK_K=64)
        endif()

        add_custom_command(
            OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
            COMMAND xcrun -sdk macosx metal    ${XC_FLAGS} -c ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-metal.metal -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-metal.air
            COMMAND xcrun -sdk macosx metallib                ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-metal.air   -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
            DEPENDS ggml-metal.metal
            COMMENT "Compiling Metal kernels"
        )

        add_custom_target(
            ggml-metal ALL
            DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
        )
    endif()

    set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS}
        ${FOUNDATION_LIBRARY}
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
        )
endif()
if (LLAMA_BLAS)
    if (LLAMA_STATIC)
        set(BLA_STATIC ON)
    endif()
    if ($(CMAKE_VERSION) VERSION_GREATER_EQUAL 3.22)
        set(BLA_SIZEOF_INTEGER 8)
    endif()

    set(BLA_VENDOR ${LLAMA_BLAS_VENDOR})
    find_package(BLAS)

    if (BLAS_FOUND)
        message(STATUS "BLAS found, Libraries: ${BLAS_LIBRARIES}")

        if ("${BLAS_INCLUDE_DIRS}" STREQUAL "")
            # BLAS_INCLUDE_DIRS is missing in FindBLAS.cmake.
            # see https://gitlab.kitware.com/cmake/cmake/-/issues/20268
            find_package(PkgConfig REQUIRED)
            if (${LLAMA_BLAS_VENDOR} MATCHES "Generic")
                pkg_check_modules(DepBLAS REQUIRED blas)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "OpenBLAS")
                # As of openblas v0.3.22, the 64-bit is named openblas64.pc
                pkg_check_modules(DepBLAS openblas64)
                if (NOT DepBLAS_FOUND)
                    pkg_check_modules(DepBLAS REQUIRED openblas)
                endif()
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "FLAME")
                pkg_check_modules(DepBLAS REQUIRED blis)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "ATLAS")
                pkg_check_modules(DepBLAS REQUIRED blas-atlas)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "FlexiBLAS")
                pkg_check_modules(DepBLAS REQUIRED flexiblas_api)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "Intel")
                # all Intel* libraries share the same include path
                pkg_check_modules(DepBLAS REQUIRED mkl-sdl)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "NVHPC")
                # this doesn't provide pkg-config
                # suggest to assign BLAS_INCLUDE_DIRS on your own
                if ("${NVHPC_VERSION}" STREQUAL "")
                    message(WARNING "Better to set NVHPC_VERSION")
                else()
                    set(DepBLAS_FOUND ON)
                    set(DepBLAS_INCLUDE_DIRS "/opt/nvidia/hpc_sdk/${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}/${NVHPC_VERSION}/math_libs/include")
                endif()
            endif()
            if (DepBLAS_FOUND)
                set(BLAS_INCLUDE_DIRS ${DepBLAS_INCLUDE_DIRS})
            else()
                message(WARNING "BLAS_INCLUDE_DIRS neither been provided nor been automatically"
                " detected by pkgconfig, trying to find cblas.h from possible paths...")
                find_path(BLAS_INCLUDE_DIRS
                    NAMES cblas.h
                    HINTS
                        /usr/include
                        /usr/local/include
                        /usr/include/openblas
                        /opt/homebrew/opt/openblas/include
                        /usr/local/opt/openblas/include
                        /usr/include/x86_64-linux-gnu/openblas/include
                )
            endif()
        endif()

        message(STATUS "BLAS found, Includes: ${BLAS_INCLUDE_DIRS}")
        add_compile_options(${BLAS_LINKER_FLAGS})
        add_compile_definitions(GGML_USE_OPENBLAS)
        if (${BLAS_INCLUDE_DIRS} MATCHES "mkl" AND (${LLAMA_BLAS_VENDOR} MATCHES "Generic" OR ${LLAMA_BLAS_VENDOR} MATCHES "Intel"))
            add_compile_definitions(GGML_BLAS_USE_MKL)
        endif()
        set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} ${BLAS_LIBRARIES})
        set(LLAMA_EXTRA_INCLUDES ${LLAMA_EXTRA_INCLUDES} ${BLAS_INCLUDE_DIRS})

    else()
        message(WARNING "BLAS not found, please refer to "
        "https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors"
        " to set correct LLAMA_BLAS_VENDOR")
    endif()
endif()

if (LLAMA_QKK_64)
    add_compile_definitions(GGML_QKK_64)
endif()

if (LLAMA_CUBLAS)
    cmake_minimum_required(VERSION 3.17)

    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        message(STATUS "cuBLAS found")

        enable_language(CUDA)

        set(GGML_HEADERS_CUDA ggml-cuda.h)
        set(GGML_SOURCES_CUDA ggml-cuda.cu)

        add_compile_definitions(GGML_USE_CUBLAS)
#        if (LLAMA_CUDA_CUBLAS)
#            add_compile_definitions(GGML_CUDA_CUBLAS)
#        endif()
        if (LLAMA_CUDA_FORCE_DMMV)
            add_compile_definitions(GGML_CUDA_FORCE_DMMV)
        endif()
        if (LLAMA_CUDA_FORCE_MMQ)
            add_compile_definitions(GGML_CUDA_FORCE_MMQ)
        endif()
        add_compile_definitions(GGML_CUDA_DMMV_X=${LLAMA_CUDA_DMMV_X})
        add_compile_definitions(GGML_CUDA_MMV_Y=${LLAMA_CUDA_MMV_Y})
        if (DEFINED LLAMA_CUDA_DMMV_Y)
            add_compile_definitions(GGML_CUDA_MMV_Y=${LLAMA_CUDA_DMMV_Y}) # for backwards compatibility
        endif()
        if (LLAMA_CUDA_F16 OR LLAMA_CUDA_DMMV_F16)
            add_compile_definitions(GGML_CUDA_F16)
        endif()
        add_compile_definitions(K_QUANTS_PER_ITERATION=${LLAMA_CUDA_KQUANTS_ITER})
        add_compile_definitions(GGML_CUDA_PEER_MAX_BATCH_SIZE=${LLAMA_CUDA_PEER_MAX_BATCH_SIZE})

        if (LLAMA_STATIC)
            if (WIN32)
                # As of 12.3.1 CUDA Tookit for Windows does not offer a static cublas library
                set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas CUDA::cublasLt)
            else ()
                set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
            endif()
        else()
            set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt)
        endif()

        set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cuda_driver)

    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # 52 == lowest CUDA 12 standard
        # 60 == f16 CUDA intrinsics
        # 61 == integer CUDA intrinsics
        # 70 == compute capability at which unrolling a loop in mul_mat_q kernels is faster
        if (LLAMA_CUDA_F16 OR LLAMA_CUDA_DMMV_F16)
            set(CMAKE_CUDA_ARCHITECTURES "60;61;70") # needed for f16 CUDA intrinsics
        else()
            set(CMAKE_CUDA_ARCHITECTURES "52;61;70") # lowest CUDA 12 standard + lowest for integer intrinsics
            #set(CMAKE_CUDA_ARCHITECTURES "") # use this to compile much faster, but only F16 models work
        endif()
    endif()
    message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    else()
        message(WARNING "cuBLAS not found")
    endif()
endif()

if (LLAMA_MPI)
    cmake_minimum_required(VERSION 3.10)
    find_package(MPI)
    if (MPI_C_FOUND)
        message(STATUS "MPI found")
        set(GGML_HEADERS_MPI ggml-mpi.h)
        set(GGML_SOURCES_MPI ggml-mpi.c ggml-mpi.h)
        add_compile_definitions(GGML_USE_MPI)
        add_compile_definitions(${MPI_C_COMPILE_DEFINITIONS})
        if (NOT MSVC)
            add_compile_options(-Wno-cast-qual)
        endif()
        set(LLAMA_EXTRA_LIBS     ${LLAMA_EXTRA_LIBS}     ${MPI_C_LIBRARIES})
        set(LLAMA_EXTRA_INCLUDES ${LLAMA_EXTRA_INCLUDES} ${MPI_C_INCLUDE_DIRS})
        # Even if you're only using the C header, C++ programs may bring in MPI
        # C++ functions, so more linkage is needed
        if (MPI_CXX_FOUND)
            set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS}     ${MPI_CXX_LIBRARIES})
        endif()
    else()
        message(WARNING "MPI not found")
    endif()
endif()

if (LLAMA_CLBLAST)
    find_package(CLBlast)
    if (CLBlast_FOUND)
        message(STATUS "CLBlast found")

        set(GGML_HEADERS_OPENCL ggml-opencl.h)
        set(GGML_SOURCES_OPENCL ggml-opencl.cpp)

        add_compile_definitions(GGML_USE_CLBLAST)

        set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} clblast)
    else()
        message(WARNING "CLBlast not found")
    endif()
endif()

if (LLAMA_HIPBLAS)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)

    if (NOT ${CMAKE_C_COMPILER_ID} MATCHES "Clang")
        message(WARNING "Only LLVM is supported for HIP, hint: CC=/opt/rocm/llvm/bin/clang")
    endif()
    if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
        message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
    endif()

    find_package(hip)
    find_package(hipblas)
    find_package(rocblas)

    if (${hipblas_FOUND} AND ${hip_FOUND})
        message(STATUS "HIP and hipBLAS found")
        add_compile_definitions(GGML_USE_HIPBLAS GGML_USE_CUBLAS)
        if (LLAMA_HIP_UMA)
            add_compile_definitions(GGML_HIP_UMA)
        endif()
        add_library(ggml-rocm OBJECT ggml-cuda.cu ggml-cuda.h)
        if (BUILD_SHARED_LIBS)
            set_target_properties(ggml-rocm PROPERTIES POSITION_INDEPENDENT_CODE ON)
        endif()
        if (LLAMA_CUDA_FORCE_DMMV)
            target_compile_definitions(ggml-rocm PRIVATE GGML_CUDA_FORCE_DMMV)
        endif()
        if (LLAMA_CUDA_FORCE_MMQ)
            target_compile_definitions(ggml-rocm PRIVATE GGML_CUDA_FORCE_MMQ)
        endif()
        target_compile_definitions(ggml-rocm PRIVATE GGML_CUDA_DMMV_X=${LLAMA_CUDA_DMMV_X})
        target_compile_definitions(ggml-rocm PRIVATE GGML_CUDA_MMV_Y=${LLAMA_CUDA_MMV_Y})
        target_compile_definitions(ggml-rocm PRIVATE K_QUANTS_PER_ITERATION=${LLAMA_CUDA_KQUANTS_ITER})
        set_source_files_properties(ggml-cuda.cu PROPERTIES LANGUAGE CXX)
        target_link_libraries(ggml-rocm PRIVATE hip::device PUBLIC hip::host roc::rocblas roc::hipblas)

        if (LLAMA_STATIC)
            message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
        endif()
        set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} ggml-rocm)
    else()
        message(WARNING "hipBLAS or HIP not found. Try setting CMAKE_PREFIX_PATH=/opt/rocm")
    endif()
endif()

function(get_flags CCID CCVER)
    set(C_FLAGS "")
    set(CXX_FLAGS "")

    if (CCID MATCHES "Clang")
        set(C_FLAGS   -Wunreachable-code-break -Wunreachable-code-return)
        set(CXX_FLAGS -Wunreachable-code-break -Wunreachable-code-return -Wmissing-prototypes -Wextra-semi)

        if (
            (CCID STREQUAL "Clang"      AND CCVER VERSION_GREATER_EQUAL 3.8.0) OR
            (CCID STREQUAL "AppleClang" AND CCVER VERSION_GREATER_EQUAL 7.3.0)
        )
            set(C_FLAGS ${C_FLAGS} -Wdouble-promotion)
        endif()
    elseif (CCID STREQUAL "GNU")
        set(C_FLAGS   -Wdouble-promotion)
        set(CXX_FLAGS -Wno-array-bounds)

        if (CCVER VERSION_GREATER_EQUAL 7.1.0)
            set(CXX_FLAGS ${CXX_FLAGS} -Wno-format-truncation)
        endif()
        if (CCVER VERSION_GREATER_EQUAL 8.1.0)
            set(CXX_FLAGS ${CXX_FLAGS} -Wextra-semi)
        endif()
    endif()

    set(GF_C_FLAGS   ${C_FLAGS}   PARENT_SCOPE)
    set(GF_CXX_FLAGS ${CXX_FLAGS} PARENT_SCOPE)
endfunction()

if (LLAMA_ALL_WARNINGS)
    if (NOT MSVC)
        set(WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)
        set(C_FLAGS       -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                          -Werror=implicit-int -Werror=implicit-function-declaration)
        set(CXX_FLAGS     -Wmissing-declarations -Wmissing-noreturn)

        set(C_FLAGS   ${WARNING_FLAGS} ${C_FLAGS})
        set(CXX_FLAGS ${WARNING_FLAGS} ${CXX_FLAGS})

        get_flags(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})

        add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS};${GF_C_FLAGS}>"
                            "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS};${GF_CXX_FLAGS}>")
    else()
        # todo : msvc
        set(C_FLAGS   "")
        set(CXX_FLAGS "")
    endif()
endif()

if (LLAMA_CUBLAS)
    set(CUDA_FLAGS ${CXX_FLAGS} -use_fast_math)
    if (NOT MSVC)
        set(CUDA_FLAGS ${CUDA_FLAGS} -Wno-pedantic)
    endif()

    if (LLAMA_ALL_WARNINGS AND NOT MSVC)
        set(NVCC_CMD ${CMAKE_CUDA_COMPILER} .c)
        if (NOT CMAKE_CUDA_HOST_COMPILER STREQUAL "")
            set(NVCC_CMD ${NVCC_CMD} -ccbin ${CMAKE_CUDA_HOST_COMPILER})
        endif()

        execute_process(
            COMMAND ${NVCC_CMD} -Xcompiler --version
            OUTPUT_VARIABLE CUDA_CCFULLVER
            ERROR_QUIET
        )

        if (NOT CUDA_CCFULLVER MATCHES clang)
            set(CUDA_CCID "GNU")
            execute_process(
                COMMAND ${NVCC_CMD} -Xcompiler "-dumpfullversion -dumpversion"
                OUTPUT_VARIABLE CUDA_CCVER
                ERROR_QUIET
            )
        else()
            if (CUDA_CCFULLVER MATCHES Apple)
                set(CUDA_CCID "AppleClang")
            else()
                set(CUDA_CCID "Clang")
            endif()
            string(REGEX REPLACE "^.* version ([0-9.]*).*$" "\\1" CUDA_CCVER ${CUDA_CCFULLVER})
        endif()

        message("-- CUDA host compiler is ${CUDA_CCID} ${CUDA_CCVER}")

        get_flags(${CUDA_CCID} ${CUDA_CCVER})
        list(JOIN GF_CXX_FLAGS " " CUDA_CXX_FLAGS)  # pass host compiler flags as a single argument
        if (NOT CUDA_CXX_FLAGS STREQUAL "")
            set(CUDA_FLAGS ${CUDA_FLAGS} -Xcompiler ${CUDA_CXX_FLAGS})
        endif()
    endif()

    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
endif()

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

if (LLAMA_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

# this version of Apple ld64 is buggy
execute_process(
    COMMAND ${CMAKE_C_COMPILER} ${CMAKE_EXE_LINKER_FLAGS} -Wl,-v
    ERROR_VARIABLE output
    OUTPUT_QUIET
)
if (output MATCHES "dyld-1015\.7")
    add_compile_definitions(HAVE_BUGGY_APPLE_LINKER)
endif()

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (MSVC)
  string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" CMAKE_GENERATOR_PLATFORM_LWR)
  message(STATUS "CMAKE_GENERATOR_PLATFORM: ${CMAKE_GENERATOR_PLATFORM}")
else ()
  set(CMAKE_GENERATOR_PLATFORM_LWR "")
endif ()

if (NOT MSVC)
    if (LLAMA_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (LLAMA_GPROF)
        add_compile_options(-pg)
    endif()
endif()

function(add_compile_option_cpp ARG)
    # Adds a compile option to C/C++ only, but not for Cuda.
    # Use, e.g., for CPU-architecture flags.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${ARG}>)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:${ARG}>)
endfunction()

if ((${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm") OR (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64") OR ("${CMAKE_GENERATOR_PLATFORM_LWR}" MATCHES "arm64"))
    message(STATUS "ARM detected")
    if (MSVC)
        add_compile_definitions(__ARM_NEON)
        add_compile_definitions(__ARM_FEATURE_FMA)
        add_compile_definitions(__ARM_FEATURE_DOTPROD)
        # add_compile_definitions(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) # MSVC doesn't support vdupq_n_f16, vld1q_f16, vst1q_f16
        add_compile_definitions(__aarch64__) # MSVC defines _M_ARM64 instead
    else()
        check_cxx_compiler_flag(-mfp16-format=ieee COMPILER_SUPPORTS_FP16_FORMAT_I3E)
        if (NOT "${COMPILER_SUPPORTS_FP16_FORMAT_I3E}" STREQUAL "")
            add_compile_options(-mfp16-format=ieee)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv6")
            # Raspberry Pi 1, Zero
            add_compile_options(-mfpu=neon-fp-armv8 -mno-unaligned-access)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
            # Raspberry Pi 2
            add_compile_options(-mfpu=neon-fp-armv8 -mno-unaligned-access -funsafe-math-optimizations)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv8")
            # Raspberry Pi 3, 4, Zero 2 (32-bit)
            add_compile_options(-mno-unaligned-access)
        endif()
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$" OR "${CMAKE_GENERATOR_PLATFORM_LWR}" MATCHES "^(x86_64|i686|amd64|x64)$" )
    message(STATUS "x86 detected")
    if (MSVC)
        # instruction set detection for MSVC only
        if (LLAMA_NATIVE)
            include(cmake/FindSIMD.cmake)
        endif ()
        if (LLAMA_AVX512)
            add_compile_option_cpp(/arch:AVX512)
            # MSVC has no compile-time flags enabling specific
            # AVX512 extensions, neither it defines the
            # macros corresponding to the extensions.
            # Do it manually.
            if (LLAMA_AVX512_VBMI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VBMI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VBMI__>)
            endif()
            if (LLAMA_AVX512_VNNI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VNNI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VNNI__>)
            endif()
        elseif (LLAMA_AVX2)
            add_compile_option_cpp(/arch:AVX2)
        elseif (LLAMA_AVX)
            add_compile_option_cpp(/arch:AVX)
        endif()
    else()
        if (LLAMA_NATIVE)
            add_compile_option_cpp(-march=native)
        endif()
        if (LLAMA_F16C)
            add_compile_option_cpp(-mf16c)
        endif()
        if (LLAMA_FMA)
            add_compile_option_cpp(-mfma)
        endif()
        if (LLAMA_AVX)
            add_compile_option_cpp(-mavx)
        endif()
        if (LLAMA_AVX2)
            add_compile_option_cpp(-mavx2)
        endif()
        if (LLAMA_AVX512)
            add_compile_option_cpp(-mavx512f)
            add_compile_option_cpp(-mavx512bw)
        endif()
        if (LLAMA_AVX512_VBMI)
            add_compile_option_cpp(-mavx512vbmi)
        endif()
        if (LLAMA_AVX512_VNNI)
            add_compile_option_cpp(-mavx512vnni)
        endif()
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
    message(STATUS "PowerPC detected")
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64le")
        add_compile_options(-mcpu=powerpc64le)
    else()
        add_compile_options(-mcpu=native -mtune=native)
        #TODO: Add  targets for Power8/Power9 (Altivec/VSX) and Power10(MMA) and query for big endian systems (ppc64/le/be)
    endif()
else()
    message(STATUS "Unknown architecture")
endif()

if (MINGW)
    # Target Windows 8 for PrefetchVirtualMemory
    add_compile_definitions(_WIN32_WINNT=${LLAMA_WIN_VER})
endif()

#
# POSIX conformance
#

# clock_gettime came in POSIX.1b (1993)
# CLOCK_MONOTONIC came in POSIX.1-2001 / SUSv3 as optional
# posix_memalign came in POSIX.1-2001 / SUSv3
# M_PI is an XSI extension since POSIX.1-2001 / SUSv3, came in XPG1 (1985)
add_compile_definitions(_XOPEN_SOURCE=600)

# Somehow in OpenBSD whenever POSIX conformance is specified
# some string functions rely on locale_t availability,
# which was introduced in POSIX.1-2008, forcing us to go higher
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    remove_definitions(-D_XOPEN_SOURCE=600)
    add_compile_definitions(_XOPEN_SOURCE=700)
endif()

# Data types, macros and functions related to controlling CPU affinity and
# some memory allocation are available on Linux through GNU extensions in libc
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions(_GNU_SOURCE)
endif()

# RLIMIT_MEMLOCK came in BSD, is not specified in POSIX.1,
# and on macOS its availability depends on enabling Darwin extensions
# similarly on DragonFly, enabling BSD extensions is necessary
if (
    CMAKE_SYSTEM_NAME MATCHES "Darwin" OR
    CMAKE_SYSTEM_NAME MATCHES "iOS" OR
    CMAKE_SYSTEM_NAME MATCHES "tvOS" OR
    CMAKE_SYSTEM_NAME MATCHES "DragonFly"
)
    add_compile_definitions(_DARWIN_C_SOURCE)
endif()

# alloca is a non-standard interface that is not visible on BSDs when
# POSIX conformance is specified, but not all of them provide a clean way
# to enable it in such cases
if (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    add_compile_definitions(__BSD_VISIBLE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
    add_compile_definitions(_NETBSD_SOURCE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    add_compile_definitions(_BSD_SOURCE)
endif()

#
# libraries
#

# ggml

if (GGML_USE_CPU_HBM)
    add_definitions(-DGGML_USE_CPU_HBM)
    find_library(memkind memkind REQUIRED)
endif()

add_library(ggml OBJECT
            ggml.c
            ggml.h
            ggml-alloc.c
            ggml-alloc.h
            ggml-backend.c
            ggml-backend.h
            ggml-quants.c
            ggml-quants.h
            ${GGML_SOURCES_CUDA}   ${GGML_HEADERS_CUDA}
            ${GGML_SOURCES_OPENCL} ${GGML_HEADERS_OPENCL}
            ${GGML_SOURCES_METAL}  ${GGML_HEADERS_METAL}
            ${GGML_SOURCES_MPI}    ${GGML_HEADERS_MPI}
            ${GGML_SOURCES_EXTRA}  ${GGML_HEADERS_EXTRA}
            )

target_include_directories(ggml PUBLIC . ${LLAMA_EXTRA_INCLUDES})
target_compile_features(ggml PUBLIC c_std_11) # don't bump
target_link_libraries(ggml PUBLIC Threads::Threads ${LLAMA_EXTRA_LIBS})
if (GGML_USE_CPU_HBM)
    target_link_libraries(ggml PUBLIC memkind)
endif()

add_library(ggml_static STATIC $<TARGET_OBJECTS:ggml>)
if (BUILD_SHARED_LIBS)
    set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(ggml_shared SHARED $<TARGET_OBJECTS:ggml>)
    target_link_libraries(ggml_shared PUBLIC Threads::Threads ${LLAMA_EXTRA_LIBS})
    install(TARGETS ggml_shared LIBRARY)
endif()

# llama

add_library(llama
            llama.cpp
            llama.h
            )

target_include_directories(llama PUBLIC .)
target_compile_features(llama PUBLIC cxx_std_11) # don't bump
target_link_libraries(llama PRIVATE
    ggml
    ${LLAMA_EXTRA_LIBS}
    )

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_SHARED LLAMA_BUILD)
    if (LLAMA_METAL)
        set_target_properties(llama PROPERTIES RESOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ggml-metal.metal")
    endif()
endif()


#
# install
#

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(LLAMA_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}
    CACHE PATH "Location of header files")
set(LLAMA_LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}
    CACHE PATH "Location of library files")
set(LLAMA_BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR}
    CACHE PATH "Location of binary files")
set(LLAMA_BUILD_NUMBER ${BUILD_NUMBER})
set(LLAMA_BUILD_COMMIT ${BUILD_COMMIT})
set(LLAMA_INSTALL_VERSION 0.0.${BUILD_NUMBER})
get_directory_property(LLAMA_TRANSIENT_DEFINES COMPILE_DEFINITIONS)

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/LlamaConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/LlamaConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Llama
    PATH_VARS LLAMA_INCLUDE_INSTALL_DIR
              LLAMA_LIB_INSTALL_DIR
              LLAMA_BIN_INSTALL_DIR )

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/LlamaConfigVersion.cmake
    VERSION ${LLAMA_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/LlamaConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/LlamaConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Llama)

set(GGML_PUBLIC_HEADERS "ggml.h" "ggml-alloc.h" "ggml-backend.h"
        "${GGML_HEADERS_CUDA}" "${GGML_HEADERS_OPENCL}"
        "${GGML_HEADERS_METAL}" "${GGML_HEADERS_MPI}" "${GGML_HEADERS_EXTRA}")

set_target_properties(ggml PROPERTIES PUBLIC_HEADER "${GGML_PUBLIC_HEADERS}")
install(TARGETS ggml PUBLIC_HEADER)

set_target_properties(llama PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/llama.h)
install(TARGETS llama LIBRARY PUBLIC_HEADER)

install(
    FILES convert.py
    PERMISSIONS
        OWNER_READ
        OWNER_WRITE
        OWNER_EXECUTE
        GROUP_READ
        GROUP_EXECUTE
        WORLD_READ
        WORLD_EXECUTE
    DESTINATION ${CMAKE_INSTALL_BINDIR})
install(
    FILES convert-lora-to-ggml.py
    PERMISSIONS
        OWNER_READ
        OWNER_WRITE
        OWNER_EXECUTE
        GROUP_READ
        GROUP_EXECUTE
        WORLD_READ
        WORLD_EXECUTE
    DESTINATION ${CMAKE_INSTALL_BINDIR})
if (LLAMA_METAL)
    install(
        FILES ggml-metal.metal
        PERMISSIONS
            OWNER_READ
            OWNER_WRITE
            GROUP_READ
            WORLD_READ
        DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

#
# programs, examples and tests
#

add_subdirectory(common)

if (LLAMA_BUILD_TESTS AND NOT CMAKE_JS_VERSION)
    include(CTest)
    add_subdirectory(tests)
endif ()

if (LLAMA_BUILD_EXAMPLES)
    add_subdirectory(examples)
    add_subdirectory(pocs)
endif()
