# Copyright (c)      2023 Christopher Taylor
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
# look for cray pmi...
pkg_check_modules(PC_PMI_CRAY QUIET cray-pmi)
# look for the rest if we couldn't find the cray package
if(NOT PC_PMI_CRAY_FOUND)
  pkg_check_modules(PC_PMI QUIET pmi)
endif()

find_path(
  PMI_INCLUDE_DIR pmi2.h
  HINTS ${PMI_ROOT}
        ENV
        PMI_ROOT
        ${PMI_DIR}
        ENV
        PMI_DIR
        ${PC_PMI_CRAY_INCLUDEDIR}
        ${PC_PMI_CRAY_INCLUDE_DIRS}
        ${PC_PMI_INCLUDEDIR}
        ${PC_PMI_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

find_library(
  PMI_LIBRARY
  NAMES pmi
  HINTS ${PMI_ROOT}
        ENV
        PMI_ROOT
        ${PC_PMI_CRAY_LIBDIR}
        ${PC_PMI_CRAY_LIBRARY_DIRS}
        ${PC_PMI_LIBDIR}
        ${PC_PMI_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

# Set PMI_ROOT in case the other hints are used
if(PMI_ROOT)
  # The call to file is for compatibility with windows paths
  file(TO_CMAKE_PATH ${PMI_ROOT} PMI_ROOT)
elseif("$ENV{PMI_ROOT}")
  file(TO_CMAKE_PATH $ENV{PMI_ROOT} PMI_ROOT)
else()
  file(TO_CMAKE_PATH "${PMI_INCLUDE_DIR}" PMI_INCLUDE_DIR)
  string(REPLACE "/include" "" PMI_ROOT "${PMI_INCLUDE_DIR}")
endif()

if(NOT PMI_LIBRARY OR NOT PMI_INCLUDE_DIR)
  set(PMI_FOUND=OFF)
  return()
endif()

# hpx_error( "PMI_LIBRARY OR PMI_INCLUDE_DIR not found, please install PMI or
# set \ the right PMI_ROOT path" )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PMI DEFAULT_MSG PMI_LIBRARY PMI_INCLUDE_DIR)

mark_as_advanced(PMI_ROOT PMI_LIBRARY PMI_INCLUDE_DIR)
