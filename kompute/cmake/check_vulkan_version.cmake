# Current issue: Only checks the result of GPU0
function(check_vulkan_version)
    cmake_parse_arguments(VULKAN_CHECK_VERSION "" "INCLUDE_DIR" "" ${ARGN})
    message(STATUS "Ensuring the currently installed driver supports the Vulkan version requested by the Vulkan Header.")

    # Get the current Vulkan Header version (e.g. 1.2.189).
    # This snippet is based on: https://gitlab.kitware.com/cmake/cmake/-/blob/v3.23.1/Modules/FindVulkan.cmake#L140-156
    if(VULKAN_CHECK_VERSION_INCLUDE_DIR)
        set(VULKAN_CORE_H ${VULKAN_CHECK_VERSION_INCLUDE_DIR}/vulkan/vulkan_core.h)
        if(EXISTS ${VULKAN_CORE_H})
            file(STRINGS ${VULKAN_CORE_H} VULKAN_HEADER_VERSION_LINE REGEX "^#define VK_HEADER_VERSION ")
            string(REGEX MATCHALL "[0-9]+" VULKAN_HEADER_VERSION "${VULKAN_HEADER_VERSION_LINE}")
            file(STRINGS ${VULKAN_CORE_H} VULKAN_HEADER_VERSION_LINE2 REGEX "^#define VK_HEADER_VERSION_COMPLETE ")
            if(NOT ${VULKAN_HEADER_VERSION_LINE2} STREQUAL "")
                string(REGEX MATCHALL "[0-9]+" VULKAN_HEADER_VERSION2 "${VULKAN_HEADER_VERSION_LINE2}")
                list(LENGTH VULKAN_HEADER_VERSION2 _len)
                # Versions >= 1.2.175 have an additional numbers in front of e.g. '0, 1, 2' instead of '1, 2'
                if(_len EQUAL 3)
                    list(REMOVE_AT VULKAN_HEADER_VERSION2 0)
                endif()
                list(APPEND VULKAN_HEADER_VERSION2 ${VULKAN_HEADER_VERSION})
                list(JOIN VULKAN_HEADER_VERSION2 "." VULKAN_HEADER_VERSION)
            else()
                file(STRINGS ${VULKAN_CORE_H} VULKAN_HEADER_API_VERSION_1_2 REGEX "^#define VK_API_VERSION_1_2.*")
                if(NOT ${VULKAN_HEADER_API_VERSION_1_2} STREQUAL "")
                    set(VULKAN_HEADER_VERSION "1.2.${VULKAN_HEADER_VERSION}")
                else()
                    file(STRINGS ${VULKAN_CORE_H} VULKAN_HEADER_API_VERSION_1_1 REGEX "^#define VK_API_VERSION_1_1.*")
                    if(NOT ${VULKAN_HEADER_API_VERSION_1_1} STREQUAL "")
                        set(VULKAN_HEADER_VERSION "1.1.${VULKAN_HEADER_VERSION}")
                    else()
                        message(FATAL_ERROR "'${VULKAN_CORE_H}' does not contain a supported Vulkan version. Probably because its < 1.2.0.")
                    endif()
                endif()
            endif()
        else()
            message(FATAL_ERROR "'${VULKAN_CORE_H}' does not exist. Try calling 'find_package(Vulkan REQUIRED)' before you call this function or set 'Vulkan_INCLUDE_DIR' manually!")
            return()
        endif()
    else()
        message(FATAL_ERROR "Invalid Vulkan include directory given. Try calling 'find_package(Vulkan REQUIRED)' before you call this function or set 'Vulkan_INCLUDE_DIR' manually!")
        return()
    endif()
    message(STATUS "Found Vulkan Header version: ${VULKAN_HEADER_VERSION}")

    # Get Vulkan version supported by driver
    find_program(VULKAN_INFO_PATH NAMES vulkaninfo)
    if(VULKAN_INFO_PATH STREQUAL "VULKAN_INFO_PATH-NOTFOUND")
        message(FATAL_ERROR "vulkaninfo not found. The Vulkan SDK might not be installed properly. If you know what you are doing, you can disable the Vulkan version check by setting 'KOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK' to 'ON' (-DKOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK=ON).")
        return()
    endif()

    execute_process(COMMAND "vulkaninfo"
                    OUTPUT_VARIABLE VULKAN_INFO_OUTPUT
                    RESULT_VARIABLE VULKAN_INFO_RETURN)
    if(NOT ${VULKAN_INFO_RETURN} EQUAL 0)
        message(FATAL_ERROR "Running vulkaninfo failed with return code ${VULKAN_INFO_RETURN}. Make sure you have 'vulkan-tools' installed. Result:\n${VULKAN_INFO_OUTPUT}?")
        return()
    else()
        message(STATUS "Running vulkaninfo was successful. Parsing the output...")
    endif()

    # Check if running vulkaninfo was successfully
    string(FIND "${VULKAN_INFO_OUTPUT}" "Vulkan Instance Version" VULKAN_INFO_SUCCESSFUL)
    if(VULKAN_INFO_SUCCESSFUL LESS 0)
        message(FATAL_ERROR "Running vulkaninfo failed. Make sure you have 'vulkan-tools' installed and DISPLAY is configured. If you know what you are doing, you can disable the Vulkan version check by setting 'KOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK' to 'ON' (-DKOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK=ON). Result:\n${VULKAN_INFO_OUTPUT}?")
    endif()

    string(REGEX MATCHALL "(GPU[0-9]+)" GPU_IDS "${VULKAN_INFO_OUTPUT}")
    if(NOT GPU_IDS)
        message(FATAL_ERROR "No GPU supporting Vulkan found in vulkaninfo. Does your GPU (driver) support Vulkan?")
    endif()

    string(REGEX MATCHALL "apiVersion[ ]*=[ ]*[0-9a-fA-F]*[ ]*[(]*([0-9]+[.][0-9]+[.][0-9]+)[)]*" GPU_API_VERSIONS ${VULKAN_INFO_OUTPUT})
    if(NOT GPU_API_VERSIONS)
        message(FATAL_ERROR "No valid Vulkan API version found in vulkaninfo. Does your GPU (driver) support Vulkan?")
    endif()

    # Check length
    # message(FATAL_ERROR "GPUS: ${GPU_IDS}")
    list(LENGTH GPU_IDS GPU_IDS_LENGTH)
    list(LENGTH GPU_API_VERSIONS GPU_API_VERSIONS_LENGTH)
    if(NOT ${GPU_IDS_LENGTH} EQUAL ${GPU_API_VERSIONS_LENGTH})
        message(FATAL_ERROR "Found ${GPU_IDS_LENGTH} GPUs, but ${GPU_API_VERSIONS_LENGTH} API versions in vulkaninfo. We expected to find an equal amount of them.")
    endif()

    # Compare versions
    set(VALID_GPU "")
    set(VALID_VULKAN_VERSION "")
    math(EXPR ITER_LEN "${GPU_IDS_LENGTH} - 1")
    foreach(INDEX RANGE ${ITER_LEN})
        list(GET GPU_IDS ${INDEX} GPU)
        list(GET GPU_API_VERSIONS ${INDEX} API_VERSION)

        # Extract API version
        if(${API_VERSION} MATCHES "apiVersion[ ]*=[ ]*[0-9a-fA-F]*[ ]*[(]*([0-9]+[.][0-9]+[.][0-9]+)[)]*")
            set(VULKAN_DRIVER_VERSION ${CMAKE_MATCH_1})
        else()
            message(FATAL_ERROR "API version match failed. This should not have happened...")
        endif()

        message(STATUS "${GPU} supports Vulkan API version '${VULKAN_DRIVER_VERSION}'.")

        # Compare driver and header version
        if(${VULKAN_DRIVER_VERSION} VERSION_LESS ${VULKAN_HEADER_VERSION})
        # Version missmatch. Let us check if the minor version is the same.
            if(${VULKAN_DRIVER_VERSION} MATCHES "[0-9]+[.]([0-9]+)[.][0-9]+")
                set(VULKAN_DRIVER_MINOR_VERSION ${CMAKE_MATCH_1})
            else()
                message(FATAL_ERROR "Invalid Vulkan driver version '${VULKAN_DRIVER_VERSION}' found. Expected version in the following format: '[0-9]+.[0-9]+.[0-9]+'")
            endif()
            if(${VULKAN_HEADER_VERSION} MATCHES "[0-9]+[.]([0-9]+)[.][0-9]+")
                set(VULKAN_HEADER_MINOR_VERSION ${CMAKE_MATCH_1})
            else()
                message(FATAL_ERROR "Invalid Vulkan Header version '${VULKAN_HEADER_VERSION}' found. Expected version in the following format: '[0-9]+.[0-9]+.[0-9]+'")
            endif()

            if(${VULKAN_DRIVER_MINOR_VERSION} EQUAL ${VULKAN_HEADER_MINOR_VERSION})
                message(WARNING "Your GPU driver does not support Vulkan > ${VULKAN_DRIVER_VERSION}, but you try to use Vulkan Header ${VULKAN_HEADER_VERSION}. At least your driver supports the same minor version (${VULKAN_DRIVER_MINOR_VERSION}), so this should be fine but keep it in mind in case you encounter any strange behavior.")
                set(VALID_GPU ${GPU})
                set(VALID_VULKAN_VERSION ${VULKAN_DRIVER_VERSION})
                break()
            else()
                message(STATUS "${GPU} does not support Vulkan > ${VULKAN_DRIVER_VERSION}.")
            endif()
        else()
            set(VALID_GPU ${GPU})
            set(VALID_VULKAN_VERSION ${VULKAN_DRIVER_VERSION})
            break()
        endif()
    endforeach()

    if("${VALID_GPU}" STREQUAL "")
        message(FATAL_ERROR "None of your GPUs supports Vulkan Header ${VULKAN_HEADER_VERSION}. Please try updating your driver, or downgrade your Vulkan headers. If you know what you are doing, you can disable the Vulkan version check by setting 'KOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK' to 'ON' (-DKOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK=ON).")
    else()
        message("Valid GPU (${VALID_GPU}) for Vulkan header version ${VULKAN_HEADER_VERSION} found. ${VALID_GPU} supports up to Vulkan ${VALID_VULKAN_VERSION}.")
    endif()

endfunction()
