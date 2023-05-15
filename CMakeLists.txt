# DO NOT USE THIS FILE. 
# IT'S ONLY FOR CUBLAS BUILD PURPOSES ON WINDOWS VISUAL STUDIO. 
# IT WILL NOT BE UPDATED OR MAINTAINED !!!

message(STATUS "============== ============== ==============")
message(STATUS "WARNING! Do NOT use this file. It is UNSUPPORTED for normal users. Use MAKE instead.")
message(STATUS "It is ONLY for CUBLAS build testing on windows visual studio. IT WILL NOT BE UPDATED OR MAINTAINED !!!")
message(STATUS "IF YOU ARE SEEING THIS, you MUST ONLY be building AN EXPERIMENAL WINDOWS CUBLAS BUILD! NOTHING ELSE WILL BE SUPPORTED !!!")
message(STATUS "============== ============== ==============")

cmake_minimum_required(VERSION 3.12) # Don't bump this version for no reason
project("llama.cpp" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(LLAMA_STANDALONE ON)
set(BUILD_SHARED_LIBS_DEFAULT ON)
set(LLAMA_STATIC ON)
set(LLAMA_NATIVE OFF)
set(LLAMA_LTO OFF)
set(LLAMA_ALL_WARNINGS OFF)
set(LLAMA_ALL_WARNINGS_3RD_PARTY OFF)
set(LLAMA_GPROF OFF)
set(LLAMA_SANITIZE_THREAD OFF)
set(LLAMA_SANITIZE_ADDRESS OFF)
set(LLAMA_SANITIZE_UNDEFINED OFF)

# instruction set specific
option(LLAMA_AVX                    "llama: enable AVX"                                     ON)
option(LLAMA_AVX2                   "llama: enable AVX2"                                    ON)
option(LLAMA_AVX512                 "llama: enable AVX512"                                  OFF)
option(LLAMA_AVX512_VBMI            "llama: enable AVX512-VBMI"                             OFF)
option(LLAMA_AVX512_VNNI            "llama: enable AVX512-VNNI"                             OFF)
option(LLAMA_FMA                    "llama: enable FMA"                                     ON)
# in MSVC F16C is implied with AVX2/AVX512
if (NOT MSVC)
    option(LLAMA_F16C               "llama: enable F16C"                                    ON)
endif()

# 3rd party libs
option(LLAMA_CUBLAS                 "llama: use cuBLAS"                                     ON)

option(LLAMA_BUILD_TESTS            "llama: build tests"    OFF)
option(LLAMA_BUILD_EXAMPLES         "llama: build examples" OFF)

#
# Build info header
#

# Generate initial build-info.h
include(${CMAKE_CURRENT_SOURCE_DIR}/scripts/build-info.cmake)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    set(GIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.git")

    # Is git submodule
    if(NOT IS_DIRECTORY "${GIT_DIR}")
        file(READ ${GIT_DIR} REAL_GIT_DIR_LINK)
        string(REGEX REPLACE "gitdir: (.*)\n$" "\\1" REAL_GIT_DIR ${REAL_GIT_DIR_LINK})
        set(GIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${REAL_GIT_DIR}")
    endif()

    # Add a custom target for build-info.h
    add_custom_target(BUILD_INFO ALL DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/build-info.h")

    # Add a custom command to rebuild build-info.h when .git/index changes
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/build-info.h"
        COMMENT "Generating build details from Git"
        COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build-info.cmake"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS "${GIT_DIR}/index"
        VERBATIM
    )
else()
    message(WARNING "Git repository not found; to enable automatic generation of build info, make sure Git is installed and the project is a Git repository.")
endif()

#
# Compile flags
#

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (LLAMA_CUBLAS)
    cmake_minimum_required(VERSION 3.17)

    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        message(STATUS "cuBLAS found")

        enable_language(CUDA)

        set(GGML_CUDA_SOURCES ggml-cuda.cu ggml-cuda.h)

        add_compile_definitions(GGML_USE_CUBLAS)

        if (LLAMA_STATIC)
            set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
        else()
            set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt)
        endif()

    else()
        message(WARNING "cuBLAS not found")
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
        )
        set(cxx_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
            -Wno-unused-function
            -Wno-multichar
        )
    else()
        # todo : msvc
    endif()

    add_compile_options(
            "$<$<COMPILE_LANGUAGE:C>:${c_flags}>"
            "$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>"
    )

endif()

if (MSVC)
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
            # Apple M1, M2, etc.
            # Raspberry Pi 3, 4, Zero 2 (64-bit)
            add_compile_options(-mcpu=native)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv6")
            # Raspberry Pi 1, Zero
            add_compile_options(-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
            # Raspberry Pi 2
            add_compile_options(-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv8")
            # Raspberry Pi 3, 4, Zero 2 (32-bit)
            add_compile_options(-mfp16-format=ieee -mno-unaligned-access)
        endif()
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86 detected")
    if (MSVC)
        if (LLAMA_AVX512)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX512>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
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
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX2>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
        elseif (LLAMA_AVX)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>)
        endif()
    else()
        if (LLAMA_F16C)
            add_compile_options(-mf16c)
        endif()
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
            add_compile_options(-mavx512bw)
        endif()
        if (LLAMA_AVX512_VBMI)
            add_compile_options(-mavx512vbmi)
        endif()
        if (LLAMA_AVX512_VNNI)
            add_compile_options(-mavx512vnni)
        endif()
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
    message(STATUS "PowerPC detected")
    add_compile_options(-mcpu=native -mtune=native)
    #TODO: Add  targets for Power8/Power9 (Altivec/VSX) and Power10(MMA) and query for big endian systems (ppc64/le/be)
else()
    message(STATUS "Unknown architecture")
endif()

#
# Build libraries
#

add_library(ggml OBJECT
            ggml.c
            ggml.h
            ${GGML_CUDA_SOURCES})
target_include_directories(ggml PUBLIC . ./otherarch ./otherarch/tools)
target_compile_features(ggml PUBLIC c_std_11) # don't bump
target_link_libraries(ggml PUBLIC Threads::Threads ${LLAMA_EXTRA_LIBS})
set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(ggml_v1 OBJECT
            otherarch/ggml_v1.c
            otherarch/ggml_v1.h)
target_include_directories(ggml_v1 PUBLIC . ./otherarch ./otherarch/tools)
target_compile_features(ggml_v1 PUBLIC c_std_11) # don't bump
target_link_libraries(ggml_v1 PUBLIC Threads::Threads ${LLAMA_EXTRA_LIBS})
set_target_properties(ggml_v1 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(common2
            examples/common.cpp
            examples/common.h)
target_include_directories(common2 PUBLIC . ./otherarch ./otherarch/tools ./examples)
target_compile_features(common2 PUBLIC cxx_std_11) # don't bump
target_link_libraries(common2 PRIVATE ggml ${LLAMA_EXTRA_LIBS})
set_target_properties(common2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(gpttype_adapter 
            gpttype_adapter.cpp)
target_include_directories(gpttype_adapter PUBLIC . ./otherarch ./otherarch/tools ./examples)
target_compile_features(gpttype_adapter PUBLIC cxx_std_11) # don't bump
target_link_libraries(gpttype_adapter PRIVATE common2 ggml ${LLAMA_EXTRA_LIBS})
set_target_properties(gpttype_adapter PROPERTIES POSITION_INDEPENDENT_CODE ON)


if (GGML_CUDA_SOURCES)
    message(STATUS "GGML CUDA sources found, configuring CUDA architecture")
    set_property(TARGET ggml PROPERTY CUDA_ARCHITECTURES OFF)
    set_property(TARGET ggml PROPERTY CUDA_SELECT_NVCC_ARCH_FLAGS "Auto")
    set_property(TARGET llama PROPERTY CUDA_ARCHITECTURES OFF)
endif()

set(TARGET koboldcpp)
add_library(${TARGET} SHARED expose.cpp expose.h)
target_include_directories(${TARGET} PUBLIC . ./otherarch ./otherarch/tools ./examples)
target_compile_features(${TARGET} PUBLIC cxx_std_11) # don't bump
set_target_properties(${TARGET} PROPERTIES PREFIX "")
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME "koboldcpp")
set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(${TARGET} PUBLIC ggml ggml_v1 common2 gpttype_adapter ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

