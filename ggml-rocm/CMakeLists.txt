cmake_minimum_required(VERSION 3.12) # Don't bump this version for no reason
project(ggml-rocm CXX)

set(LLAMA_CUDA_DMMV_X      "32" CACHE STRING "llama: x stride for dmmv CUDA kernels")
set(LLAMA_CUDA_MMV_Y        "1" CACHE STRING "llama: y block size for mmv CUDA kernels")
set(LLAMA_CUDA_KQUANTS_ITER "2" CACHE STRING "llama: iters./thread per block for Q2_K/Q6_K")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm)

if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    message(FATAL_ERROR "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
endif()

find_package(hip)
find_package(hipblas)
find_package(rocblas)

if (${hipblas_FOUND} AND ${hip_FOUND})
    message(STATUS "HIP and hipBLAS found")
    add_compile_definitions(GGML_USE_HIPBLAS GGML_USE_CUBLAS)
    add_library(ggml-rocm SHARED ggml-cuda.cu ggml-cuda.h ggml.h)
    set_target_properties(ggml-rocm PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if (LLAMA_CUDA_FORCE_DMMV)
        target_compile_definitions(ggml-rocm PRIVATE GGML_CUDA_FORCE_DMMV)
    endif()
    target_compile_definitions(ggml-rocm PRIVATE GGML_CUDA_DMMV_X=${LLAMA_CUDA_DMMV_X})
    target_compile_definitions(ggml-rocm PRIVATE GGML_CUDA_MMV_Y=${LLAMA_CUDA_MMV_Y})
    target_compile_definitions(ggml-rocm PRIVATE K_QUANTS_PER_ITERATION=${LLAMA_CUDA_KQUANTS_ITER})
    target_compile_definitions(ggml-rocm PRIVATE CC_TURING=1000000000)
    set_source_files_properties(ggml-cuda.cu PROPERTIES LANGUAGE CXX)
    set_target_properties(ggml-rocm PROPERTIES LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../../../bin)
    target_link_libraries(ggml-rocm PUBLIC hip::device hip::host roc::rocblas roc::hipblas)
else()
    message(FATAL_ERROR "hipBLAS or HIP not found. Try setting CMAKE_PREFIX_PATH=/opt/rocm")
endif()

if (NOT MSVC)
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
        "$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>"
)
