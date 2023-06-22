cmake_minimum_required(VERSION 3.20)

if(${INPUT_SHADER_FILE} STREQUAL "")
    message(FATAL_ERROR "No input file path provided via 'INPUT_SHADER_FILE'.")
endif()

if(${OUTPUT_HEADER_FILE} STREQUAL "")
    message(FATAL_ERROR "No output file path provided via 'OUTPUT_HEADER_FILE'.")
endif()

if(${HEADER_NAMESPACE} STREQUAL "")
    message(FATAL_ERROR "No header namespace provided via 'HEADER_NAMESPACE'.")
endif()

include(bin2h.cmake)

get_filename_component(BINARY_FILE_CONTENT ${INPUT_SHADER_FILE} NAME)
bin2h(SOURCE_FILE ${INPUT_SHADER_FILE} HEADER_FILE ${OUTPUT_HEADER_FILE} VARIABLE_NAME ${BINARY_FILE_CONTENT} HEADER_NAMESPACE ${HEADER_NAMESPACE})
file(APPEND ${OUTPUT_HEADER_FILE} "\n")