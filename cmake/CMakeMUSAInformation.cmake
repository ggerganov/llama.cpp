
# reuse cxx things

include(CMakeLanguageInformation)
include(CMakeCommonLanguageInclude)

include(Compiler/Clang)

__compiler_clang(MUSA)
__compiler_clang_cxx_standards(MUSA)

set(CMAKE_INCLUDE_FLAG_MUSA "-I")

set(CMAKE_MUSA_RUNTIME_LIBRARY_DEFAULT "SHARED")
set(CMAKE_MUSA_RUNTIME_LIBRARY_LINK_OPTIONS_STATIC  "")
set(CMAKE_MUSA_RUNTIME_LIBRARY_LINK_OPTIONS_SHARED  "")

# Populated by CMakeHIPInformation.cmake
set(CMAKE_MUSA_RUNTIME_LIBRARIES_STATIC "")
set(CMAKE_MUSA_RUNTIME_LIBRARIES_SHARED "")

# compile a C++ file into an object file
if(NOT CMAKE_MUSA_COMPILE_OBJECT)
  set(CMAKE_MUSA_COMPILE_OBJECT
    "<CMAKE_MUSA_COMPILER> -x musa --cuda-gpu-arch=${CMAKE_MUSA_ARCHITECTURES} -fPIC <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")
endif()
