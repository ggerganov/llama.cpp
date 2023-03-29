cmake_minimum_required(VERSION 3.12) # Don't bump this version for no reason
project("llama.cpp" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
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

# general
option(LLAMA_STATIC                 "llama: static link libraries"                          OFF)
option(LLAMA_NATIVE                 "llama: enable -march=native flag"                      OFF)
option(LLAMA_LTO                    "llama: enable link time optimization"                  OFF)

# debug
option(LLAMA_ALL_WARNINGS           "llama: enable all compiler warnings"                   ON)
option(LLAMA_ALL_WARNINGS_3RD_PARTY "llama: enable all compiler warnings in 3rd party libs" OFF)
option(LLAMA_GPROF                  "llama: enable gprof"                                   OFF)

# sanitizers
option(LLAMA_SANITIZE_THREAD        "llama: enable thread sanitizer"                        OFF)
option(LLAMA_SANITIZE_ADDRESS       "llama: enable address sanitizer"                       OFF)
option(LLAMA_SANITIZE_UNDEFINED     "llama: enable undefined sanitizer"                     OFF)

# instruction set specific
option(LLAMA_AVX                    "llama: enable AVX"                                     ON)
option(LLAMA_AVX2                   "llama: enable AVX2"                                    ON)
option(LLAMA_AVX512                 "llama: enable AVX512"                                  OFF)
option(LLAMA_FMA                    "llama: enable FMA"                                     ON)

# 3rd party libs
option(LLAMA_ACCELERATE             "llama: enable Accelerate framework"                    ON)
option(LLAMA_OPENBLAS               "llama: use OpenBLAS"                                   OFF)

option(LLAMA_BUILD_TESTS            "llama: build tests"    ${LLAMA_STANDALONE})
option(LLAMA_BUILD_EXAMPLES         "llama: build examples" ${LLAMA_STANDALONE})

#
# Compile flags
#

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

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
        set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} ${ACCELERATE_FRAMEWORK})
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()
if (LLAMA_OPENBLAS)
    if (LLAMA_STATIC)
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

if (LLAMA_ALL_WARNINGS)
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

if (LLAMA_LTO)
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
    if (LLAMA_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (LLAMA_GPROF)
        add_compile_options(-pg)
    endif()
    if (LLAMA_NATIVE)
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
        if (LLAMA_AVX512)
            add_compile_options(/arch:AVX512)
        elseif (LLAMA_AVX2)
            add_compile_options(/arch:AVX2)
        elseif (LLAMA_AVX)
            add_compile_options(/arch:AVX)
        endif()
    else()
        add_compile_options(-mf16c)
        if (LLAMA_FMA)
            add_compile_options(-mfma)
        endif()
        if (LLAMA_AVX)
            add_compile_options(-mavx)
        endif()
        if (LLAMA_AVX2)
            add_compile_options(-mavx2)
        endif()
        if (LLAMA_AVX512)
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
target_link_libraries(ggml PRIVATE Threads::Threads ${LLAMA_EXTRA_LIBS})
if (BUILD_SHARED_LIBS)
    set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_library(llama
            llama.cpp
            llama.h)

target_include_directories(llama PUBLIC .)
target_compile_features(llama PUBLIC cxx_std_11) # don't bump
target_link_libraries(llama PRIVATE ggml ${LLAMA_EXTRA_LIBS})
if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_SHARED LLAMA_BUILD)
endif()

#
# programs, examples and tests
#

if (LLAMA_BUILD_TESTS AND NOT CMAKE_JS_VERSION)
    include(CTest)
    add_subdirectory(tests)
endif ()

if (LLAMA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
