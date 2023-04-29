set(HEAD "unknown")
set(COUNT 0)

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

if(Git_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE TEMP_HEAD
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GIT_HEAD_RESULT
  )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-list --count HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE TEMP_COUNT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GIT_COUNT_RESULT
  )
  if(GIT_HEAD_RESULT EQUAL 0 AND GIT_COUNT_RESULT EQUAL 0)
    set(HEAD ${TEMP_HEAD})
    set(COUNT ${TEMP_COUNT})
  endif()
endif()

file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/build-info.h" "\
#ifndef BUILD_INFO_H\n\
#define BUILD_INFO_H\n\
\n\
#define BUILD_NUMBER ${COUNT}\n\
#define BUILD_COMMIT \"${HEAD}\"\n\
\n\
#endif // BUILD_INFO_H\n\
")
