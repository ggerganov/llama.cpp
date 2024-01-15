# Install script for directory: /home/jianyuzh/ws/llama.cpp/develop/examples

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/baby-llama/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/batched/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/batched-bench/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/beam-search/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/benchmark/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/convert-llama2c-to-ggml/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/embedding/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/finetune/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/infill/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/llama-bench/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/llava/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/sycl/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/main/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/tokenize/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/parallel/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/perplexity/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/quantize/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/quantize-stats/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/save-load-state/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/simple/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/speculative/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/lookahead/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/lookup/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/train-text-from-scratch/cmake_install.cmake")
  include("/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/export-lora/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
