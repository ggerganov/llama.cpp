# Function used to auto-detech soc type, if detect failed, will abort build
set(SOC_VERSION "")
function(detect_ascend_soc_type SOC_VERSION)
    execute_process(
        COMMAND bash -c "npu-smi info|awk -F' ' 'NF > 0 && NR==7 {print $3}'"
        OUTPUT_VARIABLE npu_info
        RESULT_VARIABLE npu_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if("${npu_info}" STREQUAL "" OR ${npu_result})
        message(FATAL_ERROR "Auto-detech ascend soc type failed, please specify manually or check ascend device working normally.")
    endif()
    set(${SOC_VERSION} "Ascend${npu_info}" PARENT_SCOPE)
endfunction()

if(NOT SOC_TYPE)
    detect_ascend_soc_type(SOC_VERSION)
    set(SOC_TYPE "${SOC_VERSION}")
endif()
message(STATUS "CANN: device SOC_TYPE:${SOC_TYPE}")

file(GLOB SRC_FILES
    get_row_f32.cpp
    get_row_f16.cpp
    get_row_q4_0.cpp
    get_row_q8_0.cpp
    quantize_f32_q8_0.cpp
    quantize_f16_q8_0.cpp
    quantize_float_to_q4_0.cpp
    dup.cpp
)

string(TOLOWER ${SOC_TYPE} SOC_VERSION)
set(ASCEND_CANN_PACKAGE_PATH ${CANN_INSTALL_DIR})
set(RUN_MODE "npu" CACHE STRING "run mode: npu/sim")

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the compiler package is installed.")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(ascendc_kernels STATIC
    ${SRC_FILES}
)

string(FIND "${SOC_VERSION}" "ascend310p" FIRST_310P_INDEX)
if(FIRST_310P_INDEX GREATER -1)
    ascendc_compile_definitions(ascendc_kernels PRIVATE -DASCEND_310P)
endif()
# ascendc_compile_definitions(ascendc_kernels PRIVATE -DASCENDC_DUMP)
