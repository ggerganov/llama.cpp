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

# musa, hip: add directory with a CMakeLists.txt file, but no source files (use refer to ggml-cuda files as ../ggml-cuda)
if (GGML_MUSA)
    list(APPEND CMAKE_MODULE_PATH "/usr/local/musa/cmake/")
    find_package(MUSAToolkit)
    set(CUDAToolkit_FOUND ${MUSAToolkit_FOUND})
else()
    find_package(CUDAToolkit)
endif()

# if (GGML_MUSA)
# set(CMAKE_CUDA_COMPILER ${MUSAToolkit_MCC_EXECUTABLE})
# else()
# if (GGML_MUSA)
# set_source_files_properties(${GGML_SOURCES_CUDA} PROPERTIES LANGUAGE CXX)
# foreach(SOURCE ${GGML_SOURCES_CUDA})
#     set_property(SOURCE ${SOURCE} PROPERTY COMPILE_FLAGS "-x musa -mtgpu --cuda-gpu-arch=mp_21 --cuda-gpu-arch=mp_22")
# endforeach()
# endif()


if (GGML_MUSA)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_C_EXTENSIONS OFF)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_CXX_EXTENSIONS OFF)

    set(GGML_CUDA ON)

    list(APPEND GGML_CDEF_PUBLIC GGML_USE_MUSA)
endif()

if (GGML_AMX)
    if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 11.0)
    else()
        set(GGML_AMX OFF)
        message(WARNING "AMX requires gcc version > 11.0. Turning off GGML_AMX.")
    endif()

    if (GGML_AMX)
        message(STATUS "Using AMX")

        list(APPEND GGML_CDEF_PUBLIC GGML_USE_AMX)

        file(GLOB   GGML_HEADERS_AMX "ggml-amx/*.h")
        list(APPEND GGML_HEADERS_AMX "../include/ggml-amx.h")

        file(GLOB   GGML_SOURCES_AMX "ggml-amx/*.cpp")
        list(APPEND GGML_SOURCES_AMX "ggml-amx.cpp")
    endif()
endif()


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

if (GGML_SYCL)
    if (NOT GGML_SYCL_TARGET MATCHES "^(INTEL|NVIDIA|AMD)$")
        message(FATAL_ERROR "Invalid backend chosen, supported options are INTEL, NVIDIA, or AMD")
    endif()

    check_cxx_compiler_flag("-fsycl" SUPPORTS_SYCL)

    if (DEFINED ENV{ONEAPI_ROOT})
        message(STATUS "Using oneAPI Release SYCL compiler (icpx).")
    elseif(SUPPORTS_SYCL)
        message(WARNING "Using open-source SYCL compiler (clang++). Didn't detect ENV {ONEAPI_ROOT}.
         If you expected the oneAPI Release compiler, please install oneAPI & source it, like:
         source /opt/intel/oneapi/setvars.sh")
    else()
        message(FATAL_ERROR, "C++ compiler lacks SYCL support.")
    endif()
    message(STATUS "SYCL found")
    #todo: AOT

    list(APPEND GGML_CDEF_PUBLIC GGML_USE_SYCL)

    if (GGML_SYCL_F16)
        if (GGML_SYCL_TARGET STREQUAL "AMD")
            message(WARNING "AMD target does not entirely support FP16 in the SYCL backend.")
        endif()
        add_compile_definitions(GGML_SYCL_F16)
    endif()

    if (GGML_CUDA_FORCE_MMQ)
        add_compile_definitions(GGML_SYCL_FORCE_MMQ)
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -fsycl")

    if (GGML_SYCL_TARGET STREQUAL "NVIDIA")
        add_compile_definitions(GGML_SYCL_WARP_SIZE=32)
    elseif (GGML_SYCL_TARGET STREQUAL "AMD")
        # INFO: Allowed Sub_group_sizes are not consistent through all
        # hip targets. For example, 64 is used for certain models, but the backend
        # does not support it.
        # Target archs tested working: gfx1030, gfx1031, (Only tested sub_group_size = 32)
        add_compile_definitions(GGML_SYCL_WARP_SIZE=32)
    else()
        add_compile_definitions(GGML_SYCL_WARP_SIZE=16)
    endif()

    file(GLOB   GGML_HEADERS_SYCL "ggml-sycl/*.hpp")
    list(APPEND GGML_HEADERS_SYCL "../include/ggml-sycl.h")

    file(GLOB   GGML_SOURCES_SYCL "ggml-sycl/*.cpp")
    list(APPEND GGML_SOURCES_SYCL "ggml-sycl.cpp")

    find_package(DNNL)
    message("-- DNNL found:" ${DNNL_FOUND})

    if (GGML_SYCL_TARGET STREQUAL "INTEL")
        add_compile_definitions(GGML_SYCL_DNNL=${DNNL_FOUND})
    else()
        add_compile_definitions(GGML_SYCL_DNNL=0)
    endif()

    if (${DNNL_FOUND} AND GGML_SYCL_TARGET STREQUAL "INTEL")
        list(APPEND GGML_EXTRA_LIBS_PRIVATE DNNL::dnnl)
    endif()

    if (WIN32)
        find_package(IntelSYCL REQUIRED)
        find_package(MKL REQUIRED)
        list(APPEND GGML_EXTRA_LIBS_PRIVATE IntelSYCL::SYCL_CXX MKL::MKL MKL::MKL_SYCL)
    else()
        if (GGML_SYCL_TARGET STREQUAL "INTEL")
            list(APPEND GGML_EXTRA_LIBS_PRIVATE sycl OpenCL mkl_core pthread m dl mkl_sycl_blas mkl_intel_ilp64 mkl_tbb_thread)
        elseif (GGML_SYCL_TARGET STREQUAL "NVIDIA")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda")
            list(APPEND GGML_EXTRA_LIBS_PRIVATE sycl pthread m dl onemkl)
        elseif (GGML_SYCL_TARGET STREQUAL "AMD")
            if (GGML_SYCL_HIP_TARGET STREQUAL "")
                message(ERROR "Can't enable SYCL hip backend, GGML_SYCL_HIP_TARGET has not been set.")
            endif()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${GGML_SYCL_HIP_TARGET}")
            list(APPEND GGML_EXTRA_LIBS_PRIVATE sycl pthread m dl onemkl)
        endif()
    endif()
endif()

if (GGML_VULKAN)
    find_package(Vulkan COMPONENTS glslc REQUIRED)

    if (Vulkan_FOUND)
        message(STATUS "Vulkan found")

        list(APPEND GGML_CDEF_PUBLIC GGML_USE_VULKAN)

        # Workaround to the "can't dereference invalidated vector iterator" bug in clang-cl debug build
        # Posssibly relevant: https://stackoverflow.com/questions/74748276/visual-studio-no-displays-the-correct-length-of-stdvector
        if (MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            add_compile_definitions(_ITERATOR_DEBUG_LEVEL=0)
        endif()

        if (GGML_VULKAN_CHECK_RESULTS)
            add_compile_definitions(GGML_VULKAN_CHECK_RESULTS)
        endif()

        if (GGML_VULKAN_DEBUG)
            add_compile_definitions(GGML_VULKAN_DEBUG)
        endif()

        if (GGML_VULKAN_MEMORY_DEBUG)
            add_compile_definitions(GGML_VULKAN_MEMORY_DEBUG)
        endif()

        if (GGML_VULKAN_SHADER_DEBUG_INFO)
            add_compile_definitions(GGML_VULKAN_SHADER_DEBUG_INFO)
        endif()

        if (GGML_VULKAN_PERF)
            add_compile_definitions(GGML_VULKAN_PERF)
        endif()

        if (GGML_VULKAN_VALIDATE)
            add_compile_definitions(GGML_VULKAN_VALIDATE)
        endif()

        if (GGML_VULKAN_RUN_TESTS)
            add_compile_definitions(GGML_VULKAN_RUN_TESTS)
        endif()

        add_subdirectory(vulkan-shaders)

        set (_ggml_vk_genshaders_cmd vulkan-shaders-gen)
        set (_ggml_vk_header     ${CMAKE_CURRENT_BINARY_DIR}/ggml-vulkan-shaders.hpp)
        set (_ggml_vk_source     ${CMAKE_CURRENT_BINARY_DIR}/ggml-vulkan-shaders.cpp)
        set (_ggml_vk_input_dir  ${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders)
        set (_ggml_vk_output_dir ${CMAKE_CURRENT_BINARY_DIR}/vulkan-shaders.spv)

        file(GLOB _ggml_vk_shader_deps "${_ggml_vk_input_dir}/*.comp")

        add_custom_command(
            OUTPUT ${_ggml_vk_header}
                   ${_ggml_vk_source}

            COMMAND ${_ggml_vk_genshaders_cmd}
                --glslc      ${Vulkan_GLSLC_EXECUTABLE}
                --input-dir  ${_ggml_vk_input_dir}
                --output-dir ${_ggml_vk_output_dir}
                --target-hpp ${_ggml_vk_header}
                --target-cpp ${_ggml_vk_source}
                --no-clean

            DEPENDS ${_ggml_vk_shader_deps}
            COMMENT "Generate vulkan shaders"
        )

        set(GGML_HEADERS_VULKAN ${CMAKE_CURRENT_SOURCE_DIR}/../include/ggml-vulkan.h ${_ggml_vk_header})
        set(GGML_SOURCES_VULKAN ggml-vulkan.cpp ${_ggml_vk_source})

        list(APPEND GGML_EXTRA_LIBS_PRIVATE Vulkan::Vulkan)
        list(APPEND GGML_EXTRA_INCLUDES     ${CMAKE_CURRENT_BINARY_DIR})
    else()
        message(WARNING "Vulkan not found")
    endif()
endif()

if (GGML_KOMPUTE)
    add_compile_definitions(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

    find_package(Vulkan COMPONENTS glslc REQUIRED)
    find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)

    if (NOT glslc_executable)
        message(FATAL_ERROR "glslc not found")
    endif()

    function(compile_shader)
        set(options)
        set(oneValueArgs)
        set(multiValueArgs SOURCES)
        cmake_parse_arguments(compile_shader "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
        foreach(source ${compile_shader_SOURCES})
            get_filename_component(filename ${source} NAME)
            set(spv_file ${filename}.spv)
            add_custom_command(
                OUTPUT ${spv_file}
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${source}
                ${CMAKE_CURRENT_SOURCE_DIR}/kompute-shaders/common.comp
                ${CMAKE_CURRENT_SOURCE_DIR}/kompute-shaders/op_getrows.comp
                ${CMAKE_CURRENT_SOURCE_DIR}/kompute-shaders/op_mul_mv_q_n_pre.comp
                ${CMAKE_CURRENT_SOURCE_DIR}/kompute-shaders/op_mul_mv_q_n.comp
                COMMAND ${glslc_executable} --target-env=vulkan1.2 -o ${spv_file} ${CMAKE_CURRENT_SOURCE_DIR}/${source}
                COMMENT "Compiling ${source} to ${spv_file}"
                )

            get_filename_component(RAW_FILE_NAME ${spv_file} NAME)
            set(FILE_NAME "shader${RAW_FILE_NAME}")
            string(REPLACE ".comp.spv" ".h" HEADER_FILE ${FILE_NAME})
            string(TOUPPER ${HEADER_FILE} HEADER_FILE_DEFINE)
            string(REPLACE "." "_" HEADER_FILE_DEFINE "${HEADER_FILE_DEFINE}")
            set(OUTPUT_HEADER_FILE "${HEADER_FILE}")
            message(STATUS "${HEADER_FILE} generating ${HEADER_FILE_DEFINE}")
            if(CMAKE_GENERATOR MATCHES "Visual Studio")
                add_custom_command(
                    OUTPUT ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo "/*THIS FILE HAS BEEN AUTOMATICALLY GENERATED - DO NOT EDIT*/" > ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo \"\#ifndef ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo \"\#define ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo "namespace kp {" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo "namespace shader_data {" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/xxd -i ${RAW_FILE_NAME} >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo "}}" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo \"\#endif // define ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
                    DEPENDS ${spv_file} xxd
                    COMMENT "Converting to hpp: ${FILE_NAME} ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/xxd"
                    )
            else()
                add_custom_command(
                    OUTPUT ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo "/*THIS FILE HAS BEEN AUTOMATICALLY GENERATED - DO NOT EDIT*/" > ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo \"\#ifndef ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo \"\#define ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo "namespace kp {" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo "namespace shader_data {" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_BINARY_DIR}/bin/xxd -i ${RAW_FILE_NAME} >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo "}}" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo \"\#endif // define ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
                    DEPENDS ${spv_file} xxd
                    COMMENT "Converting to hpp: ${FILE_NAME} ${CMAKE_BINARY_DIR}/bin/xxd"
                    )
            endif()
        endforeach()
    endfunction()

    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kompute/CMakeLists.txt")
        message(STATUS "Kompute found")
        set(KOMPUTE_OPT_LOG_LEVEL Error CACHE STRING "Kompute log level")
        add_subdirectory(kompute)

        # Compile our shaders
        compile_shader(SOURCES
            kompute-shaders/op_scale.comp
            kompute-shaders/op_scale_8.comp
            kompute-shaders/op_add.comp
            kompute-shaders/op_addrow.comp
            kompute-shaders/op_mul.comp
            kompute-shaders/op_silu.comp
            kompute-shaders/op_relu.comp
            kompute-shaders/op_gelu.comp
            kompute-shaders/op_softmax.comp
            kompute-shaders/op_norm.comp
            kompute-shaders/op_rmsnorm.comp
            kompute-shaders/op_diagmask.comp
            kompute-shaders/op_mul_mat_mat_f32.comp
            kompute-shaders/op_mul_mat_f16.comp
            kompute-shaders/op_mul_mat_q8_0.comp
            kompute-shaders/op_mul_mat_q4_0.comp
            kompute-shaders/op_mul_mat_q4_1.comp
            kompute-shaders/op_mul_mat_q4_k.comp
            kompute-shaders/op_mul_mat_q6_k.comp
            kompute-shaders/op_getrows_f32.comp
            kompute-shaders/op_getrows_f16.comp
            kompute-shaders/op_getrows_q4_0.comp
            kompute-shaders/op_getrows_q4_1.comp
            kompute-shaders/op_getrows_q6_k.comp
            kompute-shaders/op_rope_f16.comp
            kompute-shaders/op_rope_f32.comp
            kompute-shaders/op_cpy_f16_f16.comp
            kompute-shaders/op_cpy_f16_f32.comp
            kompute-shaders/op_cpy_f32_f16.comp
            kompute-shaders/op_cpy_f32_f32.comp
        )

        # Create a custom target for our generated shaders
        add_custom_target(generated_shaders DEPENDS
            shaderop_scale.h
            shaderop_scale_8.h
            shaderop_add.h
            shaderop_addrow.h
            shaderop_mul.h
            shaderop_silu.h
            shaderop_relu.h
            shaderop_gelu.h
            shaderop_softmax.h
            shaderop_norm.h
            shaderop_rmsnorm.h
            shaderop_diagmask.h
            shaderop_mul_mat_mat_f32.h
            shaderop_mul_mat_f16.h
            shaderop_mul_mat_q8_0.h
            shaderop_mul_mat_q4_0.h
            shaderop_mul_mat_q4_1.h
            shaderop_mul_mat_q4_k.h
            shaderop_mul_mat_q6_k.h
            shaderop_getrows_f32.h
            shaderop_getrows_f16.h
            shaderop_getrows_q4_0.h
            shaderop_getrows_q4_1.h
            shaderop_getrows_q6_k.h
            shaderop_rope_f16.h
            shaderop_rope_f32.h
            shaderop_cpy_f16_f16.h
            shaderop_cpy_f16_f32.h
            shaderop_cpy_f32_f16.h
            shaderop_cpy_f32_f32.h
        )

        # Create a custom command that depends on the generated_shaders
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ggml-kompute.stamp
            COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/ggml-kompute.stamp
            DEPENDS generated_shaders
            COMMENT "Ensuring shaders are generated before compiling ggml-kompute.cpp"
        )

        # Add the stamp to the main sources to ensure dependency tracking
        set(GGML_SOURCES_KOMPUTE ggml-kompute.cpp           ${CMAKE_CURRENT_BINARY_DIR}/ggml-kompute.stamp)
        set(GGML_HEADERS_KOMPUTE ../include/ggml-kompute.h  ${CMAKE_CURRENT_BINARY_DIR}/ggml-kompute.stamp)

        list(APPEND GGML_CDEF_PUBLIC GGML_USE_KOMPUTE)

        list(APPEND GGML_EXTRA_LIBS_PRIVATE kompute)
        list(APPEND GGML_EXTRA_INCLUDES     ${CMAKE_CURRENT_BINARY_DIR})
    else()
        message(WARNING "Kompute not found")
    endif()
endif()

if (GGML_CANN)
    if ("cann${CANN_INSTALL_DIR}" STREQUAL "cann" AND DEFINED ENV{ASCEND_TOOLKIT_HOME})
        set(CANN_INSTALL_DIR $ENV{ASCEND_TOOLKIT_HOME})
        message(STATUS "CANN: updated CANN_INSTALL_DIR from ASCEND_TOOLKIT_HOME=$ENV{ASCEND_TOOLKIT_HOME}")
    endif()

    if (CANN_INSTALL_DIR)
        # Only Support Linux.
        if (GGML_CANN)
            if (NOT UNIX)
                set(GGML_CANN OFF)
                message(WARNING "CANN: CANN toolkit supports unix but not ${CMAKE_SYSTEM_NAME}. Turning off GGML_CANN")
            endif()
        endif()

        # Supported platforms: x86-64, arm64
        if (GGML_CANN)
            if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
            elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
            else()
                set(GGML_CANN OFF)
                message(WARNING "CANN: CANN toolkit supports x86-64 and arm64 but not ${CMAKE_SYSTEM_PROCESSOR}. Turning off GGML_CANN")
            endif()
        endif()

        # Set header and libs
        if(GGML_CANN)
            set(CANN_INCLUDE_DIRS
                ${CANN_INSTALL_DIR}/include
                ${CANN_INSTALL_DIR}/include/aclnn
                ${CANN_INSTALL_DIR}/acllib/include
            )

            add_subdirectory(ggml-cann/kernels)
            list(APPEND CANN_LIBRARIES
                ascendcl
                nnopbase
                opapi
                acl_op_compiler
                ascendc_kernels
            )

            set(GGML_HEADERS_CANN "../include/ggml-cann.h")
            file(GLOB GGML_SOURCES_CANN "ggml-cann/*.cpp")
            list(APPEND GGML_SOURCES_CANN "ggml-cann.cpp")

            message(STATUS "CANN: CANN_INCLUDE_DIRS =  ${CANN_INCLUDE_DIRS}")
            message(STATUS "CANN: CANN_LIBRARIES =  ${CANN_LIBRARIES}")

            list(APPEND GGML_EXTRA_LIBS_PRIVATE ${CANN_LIBRARIES} )
            list(APPEND GGML_EXTRA_INCLUDES     ${CANN_INCLUDE_DIRS})
            list(APPEND GGML_EXTRA_LIBDIRS      ${CANN_INSTALL_DIR}/lib64)

            list(APPEND GGML_CDEF_PUBLIC GGML_USE_CANN)
        endif()
    else()
        set(GGML_CANN OFF)
        message(WARNING "CANN: Can't find CANN_INSTALL_DIR, do you forget to source set_var.sh. Turning off GGML_CANN")
    endif()

    if(NOT GGML_CANN)
        message(WARNING "CANN: GGML_CANN is turned OFF, see above for details.")
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
            list(APPEND C_FLAGS -Wdouble-promotion)
        endif()
    elseif (CCID STREQUAL "GNU")
        set(C_FLAGS   -Wdouble-promotion)
        set(CXX_FLAGS -Wno-array-bounds)

        if (NOT GGML_MUSA)
            if (CCVER VERSION_GREATER_EQUAL 7.1.0)
                list(APPEND CXX_FLAGS -Wno-format-truncation)
            endif()
        endif()
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

#
# libraries
#

# ggml

add_library(ggml-base STATIC
            ../include/ggml.h
            ../include/ggml-alloc
            ../include/ggml-alloc.h
            ../include/ggml-backend.h
            ../include/ggml-cpp.h
            ggml.c
            ggml-alloc.c
            ggml-backend.cpp
            ggml-threading.cpp
            ggml-threading.h
            ggml-quants.c # for quantize functions TODO: move dot fns to a separate file
            ggml-quants.h
            ggml-aarch64.c
            ggml-aarch64.h

)

add_subdirectory(ggml-cpu)

add_library(ggml
            ggml-backend-reg.cpp
)

target_link_libraries(ggml PUBLIC ggml-base ggml-cpu)

if (GGML_CUDA)
    add_subdirectory(ggml-cuda)
    target_link_libraries(ggml PUBLIC ggml-cuda)
endif()

if (GGML_METAL)
    add_subdirectory(ggml-metal)
    target_link_libraries(ggml PUBLIC ggml-metal)
endif()

if (GGML_BLAS)
    add_subdirectory(ggml-blas)
    target_link_libraries(ggml PUBLIC ggml-blas)
endif()

if (GGML_RPC)
    add_subdirectory(ggml-rpc)
    target_link_libraries(ggml PUBLIC ggml-rpc)
endif()

if (EMSCRIPTEN)
    set_target_properties(ggml PROPERTIES COMPILE_FLAGS "-msimd128")
endif()

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
    target_compile_definitions(ggml PRIVATE GGML_SHARED GGML_BUILD)
endif()
