set(TEMPLATE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build-info.h.in")
set(HEADER_FILE "${CMAKE_CURRENT_SOURCE_DIR}/build-info.h")
set(BUILD_NUMBER 0)
set(BUILD_COMMIT "unknown")
set(BUILD_COMPILER "unknown")
set(BUILD_TARGET "unknown")

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

if(GIT_HEAD_RESULT EQUAL 0 AND GIT_COUNT_RESULT EQUAL 0)
    set(BUILD_COMMIT ${HEAD})
    set(BUILD_NUMBER ${COUNT})
endif()

execute_process(
    COMMAND sh -c "$@ --version | head -1" _ ${CMAKE_C_COMPILER}
    OUTPUT_VARIABLE OUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE RES
)
if (RES EQUAL 0)
    set(BUILD_COMPILER ${OUT})
endif()

execute_process(
    COMMAND ${CMAKE_C_COMPILER} -dumpmachine
    OUTPUT_VARIABLE OUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE RES
)
if (RES EQUAL 0)
    set(BUILD_TARGET ${OUT})
endif()

# Only write the header if it's changed to prevent unnecessary recompilation
if(EXISTS ${HEADER_FILE})
    file(READ ${HEADER_FILE} CONTENTS)
    string(REGEX MATCH "BUILD_COMMIT \"([^\"]*)\"" _ ${CONTENTS})
    set(OLD_COMMIT ${CMAKE_MATCH_1})
    string(REGEX MATCH "BUILD_COMPILER \"([^\"]*)\"" _ ${CONTENTS})
    set(OLD_COMPILER ${CMAKE_MATCH_1})
    string(REGEX MATCH "BUILD_TARGET \"([^\"]*)\"" _ ${CONTENTS})
    set(OLD_TARGET ${CMAKE_MATCH_1})
    if (
        NOT OLD_COMMIT   STREQUAL BUILD_COMMIT   OR
        NOT OLD_COMPILER STREQUAL BUILD_COMPILER OR
        NOT OLD_TARGET   STREQUAL BUILD_TARGET
    )
        configure_file(${TEMPLATE_FILE} ${HEADER_FILE})
    endif()
else()
    configure_file(${TEMPLATE_FILE} ${HEADER_FILE})
endif()
