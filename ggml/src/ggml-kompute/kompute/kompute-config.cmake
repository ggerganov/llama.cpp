# General purpose GPU compute framework built on Vulkan to
# support 1000s of cross vendor graphics cards
# (AMD, Qualcomm, NVIDIA & friends). Blazing fast, mobile-enabled,
# asynchronous and optimized for advanced GPU data processing use cases.
# Backed by the Linux Foundation. 
#
# Finding this module will define the following variables:
#  KOMPUTE_FOUND - True if the core library has been found
#  KOMPUTE_LIBRARIES - Path to the core library archive
#  KOMPUTE_INCLUDE_DIRS - Path to the include directories. Gives access
#                     to kompute.h, as a single include which must be included in every
#                     file that uses this interface. Else it also points to the
#                     directory for individual includes.

find_path(KOMPUTE_INCLUDE_DIR
          NAMES kompute.h)

find_library(KOMPUTE_LIBRARY
             NAMES kompute
             HINTS ${KOMPUTE_LIBRARY_ROOT})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(KOMPUTE REQUIRED_VARS KOMPUTE_LIBRARY KOMPUTE_INCLUDE_DIR)

if(KOMPUTE_FOUND)
    set(KOMPUTE_LIBRARIES ${KOMPUTE_LIBRARY})
    set(KOMPUTE_INCLUDE_DIRS ${KOMPUTE_INCLUDE_DIR})
endif()
