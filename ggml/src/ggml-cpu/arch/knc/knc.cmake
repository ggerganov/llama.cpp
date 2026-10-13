enable_language(ASM)
if (NOT GGML_OPENMP_ENABLED)
    message(FATAL_ERROR "The KNC CPU backend requires a working GNU OpenMP runtime")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(GGML_KNC_PROFILE "7120P" CACHE STRING "KNC SKU: 3120A, 5110P or 7120P")
set_property(CACHE GGML_KNC_PROFILE PROPERTY STRINGS 3120A 5110P 7120P)
if (NOT GGML_KNC_PROFILE MATCHES "^(3120A|5110P|7120P)$")
    message(FATAL_ERROR "Unsupported KNC SKU")
endif()
if (GGML_KNC_PROFILE STREQUAL "3120A")
    set(knc_l1 4)
    set(knc_l2 16)
else()
    set(knc_l1 8)
    set(knc_l2 32)
endif()
set(GGML_KNC_PREFETCH_L1 ${knc_l1} CACHE STRING "L1 lead in 16-value iterations; zero disables")
set(GGML_KNC_PREFETCH_L2 ${knc_l2} CACHE STRING "L2 lead in 16-value iterations; zero disables")
set(GGML_KNC_PREFETCH_STEP 64 CACHE STRING "Issue prefetch every 16 or 64 values")
option(GGML_KNC_PREFETCH_Q8 "Prefetch packed activations" ON)
set(knc ${CMAKE_CURRENT_SOURCE_DIR}/ggml-cpu/arch/knc)
set(knc_generated ${CMAKE_CURRENT_BINARY_DIR}/knc)
file(MAKE_DIRECTORY ${knc_generated})
execute_process(COMMAND ${Python3_EXECUTABLE} ${knc}/prefetch.py
    --source ${knc} --output ${knc_generated}
    --l1 ${GGML_KNC_PREFETCH_L1} --l2 ${GGML_KNC_PREFETCH_L2}
    --step ${GGML_KNC_PREFETCH_STEP} --q8 ${GGML_KNC_PREFETCH_Q8}
    RESULT_VARIABLE knc_result)
if (NOT knc_result EQUAL 0)
    message(FATAL_ERROR "Invalid KNC prefetch configuration")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${knc}/prefetch.py ${knc}/ggml-knc-native.S ${knc}/quant-native.S)
if (GGML_KNC_PREFETCH_Q8)
    set(knc_q8 1)
else()
    set(knc_q8 0)
endif()
configure_file(${knc}/candidate-config.h.in ${knc_generated}/candidate-config.h @ONLY)
list(APPEND ARCH_DEFINITIONS GGML_CPU_GENERIC GGML_USE_KNC)
list(APPEND GGML_CPU_SOURCES
    ${knc}/adapter.c ${knc}/knc-mq.c ${knc}/policy.c ${knc}/quant.c
    ${knc}/ggml-knc.c ${knc}/ggml-knc-imci.c
    ${knc_generated}/ggml-knc-native.S ${knc_generated}/quant-native.S)
target_include_directories(${GGML_CPU_NAME} PRIVATE ${knc} ${knc_generated})
set_source_files_properties(${knc}/ggml-knc-imci.c ${knc}/quant.c
    ${knc_generated}/ggml-knc-native.S ${knc_generated}/quant-native.S PROPERTIES
    COMPILE_DEFINITIONS TARGET_KNC_${GGML_KNC_PROFILE})
set_source_files_properties(${knc}/knc-mq.c ${knc}/policy.c ${knc}/quant.c PROPERTIES
    COMPILE_OPTIONS "-O2;-fno-fast-math;-fno-tree-vectorize;-fno-tree-slp-vectorize")
set_source_files_properties(${knc}/ggml-knc.c ${knc}/ggml-knc-imci.c PROPERTIES
    COMPILE_OPTIONS "-O3;-ffast-math;-funroll-loops;-ftree-vectorize;-ftree-slp-vectorize")
