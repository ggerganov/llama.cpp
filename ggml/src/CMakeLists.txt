include(CheckCXXCompilerFlag)

unset(GGML_CDEF_PUBLIC)

add_compile_definitions(GGML_SCHED_MAX_COPIES=${GGML_SCHED_MAX_COPIES})

# enable libstdc++ assertions for debug builds
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions($<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
endif()

if (NOT MSVC)
    if (GGML_SANITIZE_THREAD)
        add_compile_options(-fsanitize=thread)
        link_libraries     (-fsanitize=thread)
    endif()

    if (GGML_SANITIZE_ADDRESS)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries     (-fsanitize=address)
    endif()

    if (GGML_SANITIZE_UNDEFINED)
        add_compile_options(-fsanitize=undefined)
        link_libraries     (-fsanitize=undefined)
    endif()
endif()

unset(GGML_EXTRA_LIBS_PRIVATE)
unset(GGML_EXTRA_LIBS_PUBLIC)

if (GGML_HIPBLAS)
    if (NOT EXISTS $ENV{ROCM_PATH})
        if (NOT EXISTS /opt/rocm)
            set(ROCM_PATH /usr)
        else()
            set(ROCM_PATH /opt/rocm)
        endif()
    else()
        set(ROCM_PATH $ENV{ROCM_PATH})
    endif()

    list(APPEND CMAKE_PREFIX_PATH  ${ROCM_PATH})
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

    # CMake on Windows doesn't support the HIP language yet
    if (WIN32)
        set(CXX_IS_HIPCC TRUE)
    else()
        string(REGEX MATCH "hipcc(\.bat)?$" CXX_IS_HIPCC "${CMAKE_CXX_COMPILER}")
    endif()

    if (CXX_IS_HIPCC)
        if (LINUX)
            if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
                message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
            endif()

            message(WARNING "Setting hipcc as the C++ compiler is legacy behavior."
                    " Prefer setting the HIP compiler directly. See README for details.")
        endif()
    else()
        # Forward AMDGPU_TARGETS to CMAKE_HIP_ARCHITECTURES.
        if (AMDGPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
            set(CMAKE_HIP_ARCHITECTURES ${AMDGPU_TARGETS})
        endif()
        cmake_minimum_required(VERSION 3.21)
        enable_language(HIP)
    endif()

    find_package(hip     REQUIRED)
    find_package(hipblas REQUIRED)
    find_package(rocblas REQUIRED)

    message(STATUS "HIP and hipBLAS found")

    file(GLOB   GGML_HEADERS_ROCM "ggml-cuda/*.cuh")
    list(APPEND GGML_HEADERS_ROCM "../include/ggml-cuda.h")

    file(GLOB   GGML_SOURCES_ROCM "ggml-cuda/*.cu")
    list(APPEND GGML_SOURCES_ROCM "ggml-cuda.cu")
    file(GLOB   SRCS "ggml-cuda/template-instances/fattn-wmma*.cu")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})
    file(GLOB   SRCS "ggml-cuda/template-instances/mmq*.cu")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})

    if (GGML_CUDA_FA_ALL_QUANTS)
        file(GLOB   SRCS "ggml-cuda/template-instances/fattn-vec*.cu")
        list(APPEND GGML_SOURCES_ROCM ${SRCS})
        add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
    else()
        file(GLOB   SRCS "ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu")
        list(APPEND GGML_SOURCES_ROCM ${SRCS})
        file(GLOB   SRCS "ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu")
        list(APPEND GGML_SOURCES_ROCM ${SRCS})
        file(GLOB   SRCS "ggml-cuda/template-instances/fattn-vec*f16-f16.cu")
        list(APPEND GGML_SOURCES_ROCM ${SRCS})
    endif()

    list(APPEND GGML_CDEF_PUBLIC GGML_USE_CUDA)

    add_compile_definitions(GGML_USE_HIPBLAS)
    add_compile_definitions(GGML_CUDA_DMMV_X=${GGML_CUDA_DMMV_X})
    add_compile_definitions(GGML_CUDA_MMV_Y=${GGML_CUDA_MMV_Y})
    add_compile_definitions(K_QUANTS_PER_ITERATION=${GGML_CUDA_KQUANTS_ITER})

    if (GGML_HIP_UMA)
        add_compile_definitions(GGML_HIP_UMA)
    endif()

    if (GGML_CUDA_FORCE_DMMV)
        add_compile_definitions(GGML_CUDA_FORCE_DMMV)
    endif()

    if (GGML_CUDA_FORCE_MMQ)
        add_compile_definitions(GGML_CUDA_FORCE_MMQ)
    endif()

    if (GGML_CUDA_FORCE_CUBLAS)
        add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
    endif()

    if (GGML_CUDA_NO_PEER_COPY)
        add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
    endif()

    if (CXX_IS_HIPCC)
        set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE CXX)
        list(APPEND GGML_EXTRA_LIBS_PRIVATE hip::device)
    else()
        set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE HIP)
    endif()

    if (GGML_STATIC)
        message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
    endif()

    list(APPEND GGML_EXTRA_LIBS_PUBLIC hip::host roc::rocblas roc::hipblas)
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
            list(APPEND C_FLAGS -Wdouble-promotion)
        endif()
    elseif (CCID STREQUAL "GNU")
        set(C_FLAGS   -Wdouble-promotion)
        set(CXX_FLAGS -Wno-array-bounds)
        if (CCVER VERSION_GREATER_EQUAL 8.1.0)
            list(APPEND CXX_FLAGS -Wextra-semi)
        endif()
    endif()

    set(GF_C_FLAGS   ${C_FLAGS}   PARENT_SCOPE)
    set(GF_CXX_FLAGS ${CXX_FLAGS} PARENT_SCOPE)
endfunction()

if (GGML_FATAL_WARNINGS)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND C_FLAGS   -Werror)
        list(APPEND CXX_FLAGS -Werror)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(/WX)
    endif()
endif()

if (GGML_ALL_WARNINGS)
    if (NOT MSVC)
        list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)
        list(APPEND C_FLAGS       -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                                  -Werror=implicit-int -Werror=implicit-function-declaration)
        list(APPEND CXX_FLAGS     -Wmissing-declarations -Wmissing-noreturn)

        list(APPEND C_FLAGS   ${WARNING_FLAGS})
        list(APPEND CXX_FLAGS ${WARNING_FLAGS})

        get_flags(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})

        add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS};${GF_C_FLAGS}>"
                            "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS};${GF_CXX_FLAGS}>")
    else()
        # todo : msvc
        set(C_FLAGS   "")
        set(CXX_FLAGS "")
    endif()
endif()

if (GGML_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

if (GGML_CCACHE)
    find_program(GGML_CCACHE_FOUND ccache)

    if (GGML_CCACHE_FOUND)
        # TODO: should not be set globally
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
        set(ENV{CCACHE_SLOPPINESS} time_macros)
        message(STATUS "ccache found, compilation results will be cached. Disable with GGML_CCACHE=OFF.")
    else()
        message(STATUS "Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF")
    endif ()
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

# architecture specific
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
    if (GGML_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (GGML_GPROF)
        add_compile_options(-pg)
    endif()
endif()

if (MINGW)
    # Target Windows 8 for PrefetchVirtualMemory
    add_compile_definitions(_WIN32_WINNT=${GGML_WIN_VER})
endif()

#
# POSIX conformance
#

# clock_gettime came in POSIX.1b (1993)
# CLOCK_MONOTONIC came in POSIX.1-2001 / SUSv3 as optional
# posix_memalign came in POSIX.1-2001 / SUSv3
# M_PI is an XSI extension since POSIX.1-2001 / SUSv3, came in XPG1 (1985)

# Somehow in OpenBSD whenever POSIX conformance is specified
# some string functions rely on locale_t availability,
# which was introduced in POSIX.1-2008, forcing us to go higher
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    add_compile_definitions(_XOPEN_SOURCE=700)
else()
    add_compile_definitions(_XOPEN_SOURCE=600)
endif()

# Data types, macros and functions related to controlling CPU affinity and
# some memory allocation are available on Linux through GNU extensions in libc
if (CMAKE_SYSTEM_NAME MATCHES "Linux" OR CMAKE_SYSTEM_NAME MATCHES "Android")
    add_compile_definitions(_GNU_SOURCE)
endif()

# RLIMIT_MEMLOCK came in BSD, is not specified in POSIX.1,
# and on macOS its availability depends on enabling Darwin extensions
# similarly on DragonFly, enabling BSD extensions is necessary
if (
    CMAKE_SYSTEM_NAME MATCHES "Darwin" OR
    CMAKE_SYSTEM_NAME MATCHES "iOS"    OR
    CMAKE_SYSTEM_NAME MATCHES "tvOS"   OR
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

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        # TODO: should not use this
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

# ggml

add_library(ggml-base
            ../include/ggml.h
            ../include/ggml-alloc.h
            ../include/ggml-backend.h
            ../include/ggml-cpp.h
            ggml.c
            ggml-alloc.c
            ggml-backend.cpp
            ggml-threading.cpp
            ggml-threading.h
            ggml-quants.c
            ggml-quants.h
            ggml-aarch64.c
            ggml-aarch64.h)

add_library(ggml
            ggml-backend-reg.cpp)

target_link_libraries(ggml PUBLIC ggml-base)

function(ggml_add_backend backend)
    string(TOUPPER "GGML_${backend}" backend_id)
    if (${backend_id})
        string(TOLOWER "ggml-${backend}" backend_target)
        add_subdirectory(${backend_target})
        if (${backend_id})
            message(STATUS "Including ${backend} backend")
            if (${BUILD_SHARED_LIBS})
                target_compile_definitions(${backend_target} PRIVATE GGML_BACKEND_BUILD)
                target_compile_definitions(${backend_target} PUBLIC  GGML_BACKEND_SHARED)
            endif()
            target_link_libraries(ggml PUBLIC ${backend_target})
            string(TOUPPER "GGML_USE_${backend}" backend_use)
            target_compile_definitions(ggml PUBLIC ${backend_use})
        endif()
    endif()
endfunction()

set(GGML_CPU ON)
ggml_add_backend(CPU)
ggml_add_backend(AMX)
ggml_add_backend(BLAS)
ggml_add_backend(CANN)
ggml_add_backend(CUDA)
ggml_add_backend(Kompute)
ggml_add_backend(METAL)
ggml_add_backend(RPC)
ggml_add_backend(SYCL)
ggml_add_backend(Vulkan)

target_compile_definitions(ggml-base PUBLIC    ${GGML_CDEF_PUBLIC})
target_include_directories(ggml-base PUBLIC    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include> $<INSTALL_INTERFACE:include>)
target_compile_features   (ggml-base PRIVATE c_std_11) # don't bump

target_compile_definitions(ggml      PUBLIC    ${GGML_CDEF_PUBLIC})
target_include_directories(ggml      PUBLIC    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include> $<INSTALL_INTERFACE:include>)
target_compile_features   (ggml      PRIVATE c_std_11) # don't bump

# this probably should be removed
target_include_directories(ggml-base PRIVATE . ${GGML_EXTRA_INCLUDES})
target_link_directories   (ggml-base PRIVATE   ${GGML_EXTRA_LIBDIRS})

list(APPEND GGML_EXTRA_LIBS_PRIVATE Threads::Threads)

find_library(MATH_LIBRARY m)
if (MATH_LIBRARY)
    if (NOT WIN32 OR NOT DEFINED ENV{ONEAPI_ROOT})
        list(APPEND GGML_EXTRA_LIBS_PRIVATE m)
    endif()
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Android")
    list(APPEND GGML_EXTRA_LIBS_PRIVATE dl) # Must be linked explicitly
endif()

list(REMOVE_DUPLICATES GGML_EXTRA_LIBS_PRIVATE)
list(REMOVE_DUPLICATES GGML_EXTRA_LIBS_PUBLIC)
target_link_libraries(ggml PRIVATE ${GGML_EXTRA_LIBS_PRIVATE} PUBLIC ${GGML_EXTRA_LIBS_PUBLIC})

if (BUILD_SHARED_LIBS)
    set_target_properties(ggml-base PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(ggml      PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(ggml-base PRIVATE GGML_BUILD)
    target_compile_definitions(ggml      PRIVATE GGML_BUILD)
    target_compile_definitions(ggml-base PUBLIC  GGML_SHARED)
    target_compile_definitions(ggml      PUBLIC  GGML_SHARED)
endif()
