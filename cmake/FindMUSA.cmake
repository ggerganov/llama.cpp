# find MUSA things

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)
include(${CMAKE_ROOT}/Modules/CMakeFindDependencyMacro.cmake)

if(DEFINED ENV{MUSA_HOME})
    set(MUSA_HOME $ENV{MUSA_HOME})
else()
    set(MUSA_HOME /usr/local/musa)
endif()

set(MUSA_MCC ${MUSA_HOME}/bin/mcc)

if (DEFINED ENV{MUSA_ARCH})
    set(MUSA_ARCH $ENV{MUSA_ARCH})
elseif(NOT MUSA_ARCH)
    set(MUSA_ARCH "21")
endif()

if(NOT MUSA_INCLUDE_DIR)
    set(MUSA_INCLUDE_DIR ${MUSA_HOME}/include)
endif()

if(NOT MUSA_LIBRARY_DIR)
    set(MUSA_LIBRARY_DIR ${MUSA_HOME}/lib)
endif()

if(NOT MUSA_LIBRARIES)
    find_library(
        MUSA_MUSA_LIBRARY
        NAMES libmusa.so
        PATHS ${MUSA_LIBRARY_DIR}
    )

    find_library(
        MUSA_MUBLAS_LIBRARY
        NAMES libmublas.so
        PATHS ${MUSA_LIBRARY_DIR}
    )

    find_library(
        MUSA_MUSART_LIBRARY
        NAMES libmusart.so
        PATHS ${MUSA_LIBRARY_DIR}
    )

    if(MUSA_MUSA_LIBRARY AND MUSA_MUBLAS_LIBRARY AND MUSA_MUSART_LIBRARY)
        set(MUSA_LIBRARIES "${MUSA_MUSA_LIBRARY};${MUSA_MUBLAS_LIBRARY};${MUSA_MUSART_LIBRARY}")
        set(MUSA_MUSA_LIBRARY CACHE STRING "${MUSA_MUSA_LIBRARY}")
        set(MUSA_MUBLAS_LIBRARY CACHE STRING "${MUSA_MUBLAS_LIBRARY}")
        set(MUSA_MUSART_LIBRARY CACHE STRING "${MUSA_MUSART_LIBRARY}")
    endif()
endif()

if(MUSA_LIBRARIES)
    if(NOT TARGET MUSA::musa)
        add_library(MUSA::musa SHARED IMPORTED)
        set_target_properties(MUSA::musa PROPERTIES
            IMPORTED_LOCATION ${MUSA_MUSA_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${MUSA_INCLUDE_DIR}
        )
    endif()

    if(NOT TARGET MUSA::mublas)
        add_library(MUSA::mublas SHARED IMPORTED)
        set_target_properties(MUSA::mublas PROPERTIES
            IMPORTED_LOCATION ${MUSA_MUBLAS_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${MUSA_INCLUDE_DIR}
        )
    endif()

    if(NOT TARGET MUSA::musart)
        add_library(MUSA::musart SHARED IMPORTED)
        set_target_properties(MUSA::musart PROPERTIES
            IMPORTED_LOCATION ${MUSA_MUSART_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${MUSA_INCLUDE_DIR}
        )
    endif()

    set(MUSA_INCLUDE_DIR ${MUSA_INCLUDE_DIR} CACHE STRING "")
    set(MUSA_LIBRARY_DIR ${MUSA_LIBRARY_DIR} CACHE STRING "")
    set(MUSA_LIBRARIES ${MUSA_LIBRARIES} CACHE STRING "")
endif()

find_package_handle_standard_args(
    MUSA
    REQUIRED_VARS
    MUSA_ARCH
    MUSA_MCC
    MUSA_INCLUDE_DIR
    MUSA_LIBRARIES
    MUSA_LIBRARY_DIR
)
mark_as_advanced(
    MUSA_INCLUDE_DIR
    MUSA_LIBRARIES
    MUSA_LIBRARY_DIR
    CMAKE_MUSA_ARCHITECTURES
    CMAKE_MUSA_COMPILER
)
