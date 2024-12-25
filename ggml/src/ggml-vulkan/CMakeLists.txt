cmake_minimum_required(VERSION 3.19)
cmake_policy(SET CMP0114 NEW)

find_package(Vulkan COMPONENTS glslc REQUIRED)

function(detect_host_compiler)
    if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
        find_program(MSVC_COMPILER cl)
        find_program(GNU_C_COMPILER gcc)
        find_program(GNU_CXX_COMPILER g++)
        if (MSVC_COMPILER)
            set(HOST_C_COMPILER "${MSVC_COMPILER}" PARENT_SCOPE)
            set(HOST_CXX_COMPILER "${MSVC_COMPILER}" PARENT_SCOPE)
        elseif (CLANG_C_COMPILER AND CLANG_CXX_COMPILER)
            set(HOST_C_COMPILER "${CLANG_C_COMPILER}" PARENT_SCOPE)
            set(HOST_CXX_COMPILER "${CLANG_CXX_COMPILER}" PARENT_SCOPE)
        else()
            message(WARNING "Neither MSVC nor clang found")
        endif()
    else()
        find_program(CLANG_C_COMPILER clang)
        find_program(CLANG_CXX_COMPILER clang++)
        find_program(GNU_C_COMPILER gcc)
        find_program(GNU_CXX_COMPILER g++)
        if (GNU_C_COMPILER AND GNU_CXX_COMPILER)
            set(HOST_C_COMPILER "${GNU_C_COMPILER}" PARENT_SCOPE)
            set(HOST_CXX_COMPILER "${GNU_CXX_COMPILER}" PARENT_SCOPE)
        elseif (CLANG_C_COMPILER AND CLANG_CXX_COMPILER)
            set(HOST_C_COMPILER "${CLANG_C_COMPILER}" PARENT_SCOPE)
            set(HOST_CXX_COMPILER "${CLANG_CXX_COMPILER}" PARENT_SCOPE)
        else()
            message(WARNING "Neither clang nor gcc found")
        endif()
    endif()
endfunction()

if (Vulkan_FOUND)
    message(STATUS "Vulkan found")

    ggml_add_backend_library(ggml-vulkan
                             ggml-vulkan.cpp
                             ../../include/ggml-vulkan.h
                            )

    # Compile a test shader to determine whether GL_NV_cooperative_matrix2 is supported.
    # If it's not, there will be an error to stderr.
    # If it's supported, set a define to indicate that we should compile those shaders
    execute_process(COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o - -fshader-stage=compute --target-env=vulkan1.3 "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders/test_coopmat2_support.comp"
                    OUTPUT_VARIABLE glslc_output
                    ERROR_VARIABLE glslc_error)

    if (${glslc_error} MATCHES ".*extension not supported: GL_NV_cooperative_matrix2.*")
        message(STATUS "GL_NV_cooperative_matrix2 not supported by glslc")
    else()
        message(STATUS "GL_NV_cooperative_matrix2 supported by glslc")
        add_compile_definitions(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    endif()

    target_link_libraries(ggml-vulkan PRIVATE Vulkan::Vulkan)
    target_include_directories(ggml-vulkan PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

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

    if (NOT CMAKE_CROSSCOMPILING)
        add_subdirectory(vulkan-shaders)
    else()
        if (GGML_SHADERS_GEN_TOOLCHAIN)
            set(HOST_CMAKE_TOOLCHAIN_FILE ${GGML_SHADERS_GEN_TOOLCHAIN})
        else()
            detect_host_compiler()
            if (NOT HOST_C_COMPILER OR NOT HOST_CXX_COMPILER)
                message(FATAL_ERROR "Host compiler not found")
            else()
                message(STATUS "Host compiler: ${HOST_C_COMPILER} ${HOST_CXX_COMPILER}")
            endif()
            configure_file(${CMAKE_CURRENT_SOURCE_DIR}/host_toolchain.cmake.in ${CMAKE_BINARY_DIR}/host_toolchain.cmake @ONLY)
            set(HOST_CMAKE_TOOLCHAIN_FILE ${CMAKE_BINARY_DIR}/host_toolchain.cmake)
        endif()
        message(STATUS "vulkan-shaders-gen toolchain file: ${HOST_CMAKE_TOOLCHAIN_FILE}")

        include(ExternalProject)
        # Native build through ExternalProject_Add
        ExternalProject_Add(
            vulkan-shaders-gen
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders
            CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=${HOST_CMAKE_TOOLCHAIN_FILE}
                    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}
            BUILD_COMMAND ${CMAKE_COMMAND} --build .
            INSTALL_COMMAND ${CMAKE_COMMAND} --install .
            INSTALL_DIR ${CMAKE_BINARY_DIR}
        )
        ExternalProject_Add_StepTargets(vulkan-shaders-gen build install)
    endif()
    set (_ggml_vk_genshaders_cmd ${CMAKE_BINARY_DIR}/bin/vulkan-shaders-gen)
    set (_ggml_vk_header     ${CMAKE_CURRENT_BINARY_DIR}/ggml-vulkan-shaders.hpp)
    set (_ggml_vk_source     ${CMAKE_CURRENT_BINARY_DIR}/ggml-vulkan-shaders.cpp)
    set (_ggml_vk_input_dir  ${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders)
    set (_ggml_vk_output_dir ${CMAKE_CURRENT_BINARY_DIR}/vulkan-shaders.spv)

    file(GLOB _ggml_vk_shader_deps "${_ggml_vk_input_dir}/*.comp")

    if (CMAKE_CROSSCOMPILING)
        set(_ggml_vk_shader_deps ${_ggml_vk_shader_deps} vulkan-shaders-gen-build vulkan-shaders-gen-install)
    endif()

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

        DEPENDS ${_ggml_vk_shader_deps} ${_ggml_vk_genshaders_cmd}
        COMMENT "Generate vulkan shaders"
    )

    target_sources(ggml-vulkan PRIVATE ${_ggml_vk_source} ${_ggml_vk_header})

else()
    message(WARNING "Vulkan not found")
endif()
