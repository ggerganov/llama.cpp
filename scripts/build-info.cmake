set(TEMPLATE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build-info.h.in")
set(HEADER_FILE "${CMAKE_CURRENT_SOURCE_DIR}/build-info.h")
set(BUILD_NUMBER 0)
set(BUILD_COMMIT "unknown")

# Look for git
find_package(Git)
if(NOT Git_FOUND)
    execute_process(
        COMMAND which git
        OUTPUT_VARIABLE GIT_EXECUTABLE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT GIT_EXECUTABLE STREQUAL "")
        set(Git_FOUND TRUE)
        message(STATUS "Found Git using 'which': ${GIT_EXECUTABLE}")
    else()
        message(WARNING "Git not found using 'find_package' or 'which'. Build info will not be accurate. Consider installing Git or ensuring it is in the PATH.")
    endif()
endif()

# Get the commit count and hash
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE HEAD
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE GIT_HEAD_RESULT
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-list --count HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE COUNT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE GIT_COUNT_RESULT
    )
    if(GIT_HEAD_RESULT EQUAL 0 AND GIT_COUNT_RESULT EQUAL 0)
        set(BUILD_COMMIT ${HEAD})
        set(BUILD_NUMBER ${COUNT})
    endif()
endif()

# Only write the header if it's changed to prevent unnecessary recompilation
if(EXISTS ${HEADER_FILE})
    file(STRINGS ${HEADER_FILE} CONTENTS REGEX "BUILD_COMMIT \"([^\"]*)\"")
    list(GET CONTENTS 0 EXISTING)
    if(NOT EXISTING STREQUAL "#define BUILD_COMMIT \"${BUILD_COMMIT}\"")
        configure_file(${TEMPLATE_FILE} ${HEADER_FILE})
    endif()
else()
    configure_file(${TEMPLATE_FILE} ${HEADER_FILE})
endif()
