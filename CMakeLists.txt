cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("jarvis.cpp" C CXX)
include(CheckIncludeFileCXX)

#set(CMAKE_WARN_DEPRECATED YES)
set(CMAKE_WARN_UNUSED_CLI YES)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(JARVIS_STANDALONE ON)

    include(git-vars)

    # configure project version
    # TODO
else()
    set(JARVIS_STANDALONE OFF)
endif()

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    option(JARVIS_WASM_SINGLE_FILE "jarvis: embed WASM inside the generated jarvis.js" ON)
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

#
# option list
#

# debug
option(JARVIS_ALL_WARNINGS           "jarvis: enable all compiler warnings"                   ON)
option(JARVIS_ALL_WARNINGS_3RD_PARTY "jarvis: enable all compiler warnings in 3rd party libs" OFF)

# build
option(JARVIS_FATAL_WARNINGS "jarvis: enable -Werror flag" OFF)

# sanitizers
option(JARVIS_SANITIZE_THREAD    "jarvis: enable thread sanitizer"    OFF)
option(JARVIS_SANITIZE_ADDRESS   "jarvis: enable address sanitizer"   OFF)
option(JARVIS_SANITIZE_UNDEFINED "jarvis: enable undefined sanitizer" OFF)

# utils
option(JARVIS_BUILD_COMMON "jarvis: build common utils library" ${JARVIS_STANDALONE})

# extra artifacts
option(JARVIS_BUILD_TESTS    "jarvis: build tests"          ${JARVIS_STANDALONE})
option(JARVIS_BUILD_EXAMPLES "jarvis: build examples"       ${JARVIS_STANDALONE})
option(JARVIS_BUILD_SERVER   "jarvis: build server example" ${JARVIS_STANDALONE})

# 3rd party libs
option(JARVIS_CURL "jarvis: use libcurl to download model from an URL" OFF)

# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)

# override ggml options
set(GGML_SANITIZE_THREAD    ${JARVIS_SANITIZE_THREAD})
set(GGML_SANITIZE_ADDRESS   ${JARVIS_SANITIZE_ADDRESS})
set(GGML_SANITIZE_UNDEFINED ${JARVIS_SANITIZE_UNDEFINED})
set(GGML_ALL_WARNINGS       ${JARVIS_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS     ${JARVIS_FATAL_WARNINGS})

# change the default for these ggml options
if (NOT DEFINED GGML_JARVISFILE)
    set(GGML_JARVISFILE_DEFAULT ON)
endif()

if (NOT DEFINED GGML_AMX)
    set(GGML_AMX ON)
endif()

if (NOT DEFINED GGML_CUDA_GRAPHS)
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif()

# transition helpers
function (jarvis_option_depr TYPE OLD NEW)
    if (${OLD})
        message(${TYPE} "${OLD} is deprecated and will be removed in the future.\nUse ${NEW} instead\n")
        set(${NEW} ON PARENT_SCOPE)
    endif()
endfunction()

jarvis_option_depr(FATAL_ERROR JARVIS_CUBLAS              GGML_CUDA)
jarvis_option_depr(WARNING     JARVIS_CUDA                GGML_CUDA)
jarvis_option_depr(WARNING     JARVIS_KOMPUTE             GGML_KOMPUTE)
jarvis_option_depr(WARNING     JARVIS_METAL               GGML_METAL)
jarvis_option_depr(WARNING     JARVIS_METAL_EMBED_LIBRARY GGML_METAL_EMBED_LIBRARY)
jarvis_option_depr(WARNING     JARVIS_NATIVE              GGML_NATIVE)
jarvis_option_depr(WARNING     JARVIS_RPC                 GGML_RPC)
jarvis_option_depr(WARNING     JARVIS_SYCL                GGML_SYCL)
jarvis_option_depr(WARNING     JARVIS_SYCL_F16            GGML_SYCL_F16)
jarvis_option_depr(WARNING     JARVIS_CANN                GGML_CANN)

#
# build the library
#

if (NOT TARGET ggml)
    add_subdirectory(ggml)
    # ... otherwise assume ggml is added by a parent CMakeLists.txt
endif()
add_subdirectory(src)

#
# install
#

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(JARVIS_BUILD_NUMBER        ${BUILD_NUMBER})
set(JARVIS_BUILD_COMMIT        ${BUILD_COMMIT})
set(JARVIS_INSTALL_VERSION 0.0.${BUILD_NUMBER})

set(JARVIS_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(JARVIS_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(JARVIS_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")


# At the moment some compile definitions are placed within the ggml/src
# directory but not exported on the `ggml` target. This could be improved by
# determining _precisely_ which defines are necessary for the jarvis-config
# package.
#
set(GGML_TRANSIENT_DEFINES)
get_target_property(GGML_DIRECTORY ggml SOURCE_DIR)
get_directory_property(GGML_DIR_DEFINES DIRECTORY ${GGML_DIRECTORY} COMPILE_DEFINITIONS)
if (GGML_DIR_DEFINES)
    list(APPEND GGML_TRANSIENT_DEFINES ${GGML_DIR_DEFINES})
endif()
get_target_property(GGML_TARGET_DEFINES ggml COMPILE_DEFINITIONS)
if (GGML_TARGET_DEFINES)
    list(APPEND GGML_TRANSIENT_DEFINES ${GGML_TARGET_DEFINES})
endif()
get_target_property(GGML_LINK_LIBRARIES ggml LINK_LIBRARIES)

set_target_properties(jarvis PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/jarvis.h)
install(TARGETS jarvis LIBRARY PUBLIC_HEADER)

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/jarvis-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/jarvis-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jarvis
    PATH_VARS JARVIS_INCLUDE_INSTALL_DIR
              JARVIS_LIB_INSTALL_DIR
              JARVIS_BIN_INSTALL_DIR )

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/jarvis-version.cmake
    VERSION ${JARVIS_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/jarvis-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/jarvis-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jarvis)

install(
    FILES convert_hf_to_gguf.py
    PERMISSIONS
        OWNER_READ
        OWNER_WRITE
        OWNER_EXECUTE
        GROUP_READ
        GROUP_EXECUTE
        WORLD_READ
        WORLD_EXECUTE
    DESTINATION ${CMAKE_INSTALL_BINDIR})

configure_file(cmake/jarvis.pc.in
        "${CMAKE_CURRENT_BINARY_DIR}/jarvis.pc"
        @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/jarvis.pc"
        DESTINATION lib/pkgconfig)

#
# utils, programs, examples and tests
#

if (JARVIS_BUILD_COMMON)
    add_subdirectory(common)
endif()

if (JARVIS_BUILD_COMMON AND JARVIS_BUILD_TESTS AND NOT CMAKE_JS_VERSION)
    include(CTest)
    add_subdirectory(tests)
endif()

if (JARVIS_BUILD_COMMON AND JARVIS_BUILD_EXAMPLES)
    add_subdirectory(examples)
    add_subdirectory(pocs)
endif()
