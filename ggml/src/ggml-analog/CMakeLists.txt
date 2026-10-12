if (NOT GGML_CPU)
    message(FATAL_ERROR "GGML_ANALOG requires the CPU backend (GGML_CPU)")
endif()

if (GGML_BACKEND_DL)
    message(FATAL_ERROR "GGML_ANALOG requires a static build (GGML_BACKEND_DL is not supported)")
endif()

set(GGML_ANALOG_SOURCES
    ggml-analog.cpp
    analog-array.cpp
    analog-kernels.cpp
    analog-shader.cpp
    analog-vulkan.cpp
)

# the compute shaders of the array kernels are compiled to SPIR-V when a GLSL compiler is available,
# the SPIR-V modules are embedded in the library and dispatched at runtime
option(GGML_ANALOG_SHADERS "ggml: build the analog compute shaders (needs a GLSL compiler)" ON)

if (GGML_ANALOG_SHADERS)
    find_program(GGML_ANALOG_GLSLC NAMES glslangValidator glslc)
endif()

set(GGML_ANALOG_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vulkan")

if (GGML_ANALOG_GLSLC)
    message(STATUS "analog: compute shaders built with ${GGML_ANALOG_GLSLC}")

    set(GGML_ANALOG_SHADER_OUT "${CMAKE_CURRENT_BINARY_DIR}/analog-shaders.gen.h")
    set(GGML_ANALOG_SHADER_LIST
        analog_dac
        analog_program
        analog_xrms
        analog_readout
    )

    set(GGML_ANALOG_SHADER_SPIRV "")
    set(GGML_ANALOG_SHADER_CMDS "")
    set(GGML_ANALOG_SHADER_DEPS "${GGML_ANALOG_SHADER_DIR}/analog-common.glsl")

    foreach(SHADER ${GGML_ANALOG_SHADER_LIST})
        set(SHADER_SRC "${GGML_ANALOG_SHADER_DIR}/${SHADER}.comp")
        set(SHADER_SPV "${CMAKE_CURRENT_BINARY_DIR}/${SHADER}.spv")

        # Vulkan 1.1 (SPIR-V 1.3) is enough for the 64 bit arithmetic of the noise of the array
        list(APPEND GGML_ANALOG_SHADER_CMDS
             COMMAND ${GGML_ANALOG_GLSLC} --target-env vulkan1.1 -I${GGML_ANALOG_SHADER_DIR}
                     -o ${SHADER_SPV} ${SHADER_SRC})

        list(APPEND GGML_ANALOG_SHADER_DEPS ${SHADER_SRC})

        # the argument is a comma separated list, a list of CMake would be split into several
        # arguments by the generator
        if (GGML_ANALOG_SHADER_SPIRV)
            set(GGML_ANALOG_SHADER_SPIRV "${GGML_ANALOG_SHADER_SPIRV},${SHADER}=${SHADER_SPV}")
        else()
            set(GGML_ANALOG_SHADER_SPIRV "${SHADER}=${SHADER_SPV}")
        endif()
    endforeach()

    add_custom_command(OUTPUT ${GGML_ANALOG_SHADER_OUT}
                       ${GGML_ANALOG_SHADER_CMDS}
                       COMMAND ${CMAKE_COMMAND}
                               -DSPIRV=${GGML_ANALOG_SHADER_SPIRV}
                               -DOUT=${GGML_ANALOG_SHADER_OUT}
                               -P ${GGML_ANALOG_SHADER_DIR}/embed-shaders.cmake
                       DEPENDS ${GGML_ANALOG_SHADER_DEPS}
                       COMMENT "Building the analog compute shaders"
                       VERBATIM)

    add_custom_target(analog-shaders DEPENDS ${GGML_ANALOG_SHADER_OUT})

    set(GGML_ANALOG_SPIRV ON)
else()
    message(STATUS "analog: no GLSL compiler found (glslangValidator or glslc), the shaders are not built")
    set(GGML_ANALOG_SPIRV OFF)
endif()

# the dispatch of the shaders on a Vulkan device needs the headers, the loader is opened at runtime
find_path(GGML_ANALOG_VULKAN_INCLUDE_DIR vulkan/vulkan.h)
if (GGML_ANALOG_VULKAN_INCLUDE_DIR AND GGML_ANALOG_SPIRV)
    message(STATUS "analog: Vulkan dispatch enabled (${GGML_ANALOG_VULKAN_INCLUDE_DIR})")
    set(GGML_ANALOG_VULKAN ON)
else()
    set(GGML_ANALOG_VULKAN OFF)
endif()

ggml_add_backend_library(ggml-analog ${GGML_ANALOG_SOURCES})

if (TARGET analog-shaders)
    add_dependencies(ggml-analog analog-shaders)
endif()

# the emulated mixed-signal core reuses the CPU kernels
target_link_libraries(ggml-analog PRIVATE ggml-cpu)

if (GGML_ANALOG_SPIRV)
    target_include_directories(ggml-analog PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_compile_definitions(ggml-analog PRIVATE GGML_ANALOG_SPIRV)
endif()

if (GGML_ANALOG_VULKAN)
    target_include_directories(ggml-analog PRIVATE ${GGML_ANALOG_VULKAN_INCLUDE_DIR})
    target_compile_definitions(ggml-analog PRIVATE GGML_ANALOG_VULKAN)
    target_link_libraries(ggml-analog PRIVATE ${CMAKE_DL_LIBS})
endif()

# the array kernels parallelize their passes, use the OpenMP runtime when the build has one,
# otherwise they fall back to std::thread
if (GGML_OPENMP_ENABLED)
    target_link_libraries(ggml-analog PRIVATE ${GGML_OPENMP_TARGET_CXX})
endif()
