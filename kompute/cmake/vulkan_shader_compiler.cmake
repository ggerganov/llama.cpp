function(vulkan_compile_shader)
     find_program(GLS_LANG_VALIDATOR_PATH NAMES glslangValidator)
     if(GLS_LANG_VALIDATOR_PATH STREQUAL "GLS_LANG_VALIDATOR_PATH-NOTFOUND")
          message(FATAL_ERROR "glslangValidator not found.")
          return()
     endif()

     cmake_parse_arguments(SHADER_COMPILE "" "INFILE;OUTFILE;NAMESPACE;RELATIVE_PATH" "" ${ARGN})
     set(SHADER_COMPILE_INFILE_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_COMPILE_INFILE}")
     set(SHADER_COMPILE_SPV_FILE_FULL "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_COMPILE_INFILE}.spv")
     set(SHADER_COMPILE_HEADER_FILE_FULL "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_COMPILE_OUTFILE}")

     if(NOT SHADER_COMPILE_RELATIVE_PATH)
          set(SHADER_COMPILE_RELATIVE_PATH "${PROJECT_SOURCE_DIR}/cmake")
     endif()
    
     # .comp -> .spv
     add_custom_command(OUTPUT "${SHADER_COMPILE_SPV_FILE_FULL}"
                        COMMAND "${GLS_LANG_VALIDATOR_PATH}"
                        ARGS "-V"
                             "${SHADER_COMPILE_INFILE_FULL}"
                             "-o"
                             "${SHADER_COMPILE_SPV_FILE_FULL}"
                        COMMENT "Compile vulkan compute shader from file '${SHADER_COMPILE_INFILE_FULL}' to '${SHADER_COMPILE_SPV_FILE_FULL}'."
                        MAIN_DEPENDENCY "${SHADER_COMPILE_INFILE_FULL}")

     # Check if big or little endian
     include (TestBigEndian)
     TEST_BIG_ENDIAN(IS_BIG_ENDIAN)

     # .spv -> .hpp
     add_custom_command(OUTPUT "${SHADER_COMPILE_HEADER_FILE_FULL}"
                        COMMAND ${CMAKE_COMMAND}
                        ARGS "-DINPUT_SHADER_FILE=${SHADER_COMPILE_SPV_FILE_FULL}"
                             "-DOUTPUT_HEADER_FILE=${SHADER_COMPILE_HEADER_FILE_FULL}"
                             "-DHEADER_NAMESPACE=${SHADER_COMPILE_NAMESPACE}"
                             "-DIS_BIG_ENDIAN=${IS_BIG_ENDIAN}"
                             "-P"
                             "${SHADER_COMPILE_RELATIVE_PATH}/bin_file_to_header.cmake"
                        WORKING_DIRECTORY "${SHADER_COMPILE_RELATIVE_PATH}"
                        COMMENT "Converting compiled shader '${SHADER_COMPILE_SPV_FILE_FULL}' to header file '${SHADER_COMPILE_HEADER_FILE_FULL}'."
                        MAIN_DEPENDENCY "${SHADER_COMPILE_SPV_FILE_FULL}")
endfunction()
