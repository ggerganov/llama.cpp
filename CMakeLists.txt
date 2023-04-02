cmake_minimum_required(VERSION 3.12) # Don't bump this version for no reason
project("rwkv.cpp" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(RWKV_STANDALONE ON)

    # configure project version
    # TODO
else()
    set(RWKV_STANDALONE OFF)
endif()

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    option(RWKV_WASM_SINGLE_FILE "rwkv: embed WASM inside the generated rwkv.js" ON)
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

# general
option(RWKV_STATIC                 "rwkv: static link libraries"                          OFF)
option(RWKV_NATIVE                 "rwkv: enable -march=native flag"                      OFF)
option(RWKV_LTO                    "rwkv: enable link time optimization"                  OFF)

# debug
option(RWKV_ALL_WARNINGS           "rwkv: enable all compiler warnings"                   ON)
option(RWKV_ALL_WARNINGS_3RD_PARTY "rwkv: enable all compiler warnings in 3rd party libs" OFF)
option(RWKV_GPROF                  "rwkv: enable gprof"                                   OFF)

# sanitizers
option(RWKV_SANITIZE_THREAD        "rwkv: enable thread sanitizer"                        OFF)
option(RWKV_SANITIZE_ADDRESS       "rwkv: enable address sanitizer"                       OFF)
option(RWKV_SANITIZE_UNDEFINED     "rwkv: enable undefined sanitizer"                     OFF)

# instruction set specific
option(RWKV_AVX                    "rwkv: enable AVX"                                     ON)
option(RWKV_AVX2                   "rwkv: enable AVX2"                                    ON)
option(RWKV_AVX512                 "rwkv: enable AVX512"                                  OFF)
option(RWKV_FMA                    "rwkv: enable FMA"                                     ON)

# 3rd party libs
option(RWKV_ACCELERATE             "rwkv: enable Accelerate framework"                    ON)
option(RWKV_OPENBLAS               "rwkv: use OpenBLAS"                                   OFF)

#
# Compile flags
#

set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (NOT MSVC)
    if (RWKV_SANITIZE_THREAD)
        add_compile_options(-fsanitize=thread)
        link_libraries(-fsanitize=thread)
    endif()

    if (RWKV_SANITIZE_ADDRESS)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries(-fsanitize=address)
    endif()

    if (RWKV_SANITIZE_UNDEFINED)
        add_compile_options(-fsanitize=undefined)
        link_libraries(-fsanitize=undefined)
    endif()
endif()

if (APPLE AND RWKV_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        add_compile_definitions(GGML_USE_ACCELERATE)
        set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} ${ACCELERATE_FRAMEWORK})
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()
if (RWKV_OPENBLAS)
    if (RWKV_STATIC)
        set(BLA_STATIC ON)
    endif()

    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS)
    if (BLAS_FOUND)
        message(STATUS "OpenBLAS found")

        add_compile_definitions(GGML_USE_OPENBLAS)
        add_link_options(${BLAS_LIBRARIES})
    else()
        message(WARNING "OpenBLAS not found")
    endif()
endif()

if (RWKV_ALL_WARNINGS)
    if (NOT MSVC)
        set(c_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
            -Wdouble-promotion
            -Wshadow
            -Wstrict-prototypes
            -Wpointer-arith
            -Wno-unused-function
        )
        set(cxx_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
            -Wno-unused-function
        )
    else()
        # todo : msvc
    endif()

    add_compile_options(
            "$<$<COMPILE_LANGUAGE:C>:${c_flags}>"
            "$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>"
    )

endif()

if (RWKV_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (NOT MSVC)
    if (RWKV_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (RWKV_GPROF)
        add_compile_options(-pg)
    endif()
    if (RWKV_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    if (MSVC)
        # TODO: arm msvc?
    else()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
            add_compile_options(-mcpu=native)
        endif()
        # TODO: armv6,7,8 version specific flags
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86 detected")
    if (MSVC)
        if (RWKV_AVX512)
            add_compile_options(/arch:AVX512)
        elseif (RWKV_AVX2)
            add_compile_options(/arch:AVX2)
        elseif (RWKV_AVX)
            add_compile_options(/arch:AVX)
        endif()
    else()
        add_compile_options(-mf16c)
        if (RWKV_FMA)
            add_compile_options(-mfma)
        endif()
        if (RWKV_AVX)
            add_compile_options(-mavx)
        endif()
        if (RWKV_AVX2)
            add_compile_options(-mavx2)
        endif()
        if (RWKV_AVX512)
            add_compile_options(-mavx512f)
            # add_compile_options(-mavx512cd)
            # add_compile_options(-mavx512dq)
            # add_compile_options(-mavx512bw)
        endif()
    endif()
else()
    # TODO: support PowerPC
    message(STATUS "Unknown architecture")
endif()

#
# Build libraries
#

add_library(ggml OBJECT
            ggml.c
            ggml.h)

target_include_directories(ggml PUBLIC .)
target_compile_features(ggml PUBLIC c_std_11) # don't bump
target_link_libraries(ggml PRIVATE Threads::Threads ${RWKV_EXTRA_LIBS})
if (BUILD_SHARED_LIBS)
    set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_library(rwkv
            rwkv.cpp
            rwkv.h)

target_include_directories(rwkv PUBLIC .)
target_compile_features(rwkv PUBLIC cxx_std_11) # don't bump
target_link_libraries(rwkv PRIVATE ggml ${RWKV_EXTRA_LIBS})

if (BUILD_SHARED_LIBS)
    set_target_properties(rwkv PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(rwkv PRIVATE RWKV_SHARED RWKV_BUILD)
endif()
