# Copyright (c) 2019-2023 Ste||ar Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#
macro(setup_openshmem)
 
  if(NOT TARGET PkgConfig::OPENSHMEM)

    set(OPENSHMEM_PC "")

    find_package(MPI)
    if (LLAMA_MPI AND MPI_C_FOUND)
      set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${MPI_LIBDIR}/pkgconfig")

      set(OPENSHMEM_PC "oshmem")
      pkg_search_module(OPENSHMEM IMPORTED_TARGET GLOBAL ${OPENSHMEM_PC})

      if(NOT OPENSHMEM_FOUND)
        find_program(OSHMEM_INFO NAMES oshmem_info ompi_info REQUIRED)

        if(NOT OSHMEM_INFO)
          message(
            FATAL_ERROR
              "oshmem_info and/or ompi_info not found! pkg-config cannot find OpenMPI's `${OPENSHMEM_PC}.pc`"
          )
        endif()

        set(OSHMEM_INFO_OUTPUT
            "${CMAKE_CURRENT_SOURCE_DIR}/oshmem_info_stdout.log"
        )
        set(OSHMEM_INFO_ERROR
            "${CMAKE_CURRENT_SOURCE_DIR}/oshmem_info_error.log"
        )

        execute_process(
          COMMAND bash -c "${OSHMEM_INFO} --path libdir"
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          RESULT_VARIABLE OSHMEM_INFO_STATUS
          OUTPUT_FILE ${OSHMEM_INFO_OUTPUT}
          ERROR_FILE ${OSHMEM_INFO_ERROR}
        )

        if(OSHMEM_INFO_STATUS)
          message(
            FATAL_ERROR
              "${OSHMEM_INFO} Failed! Program status code: ${OSHMEM_INFO_STATUS}"
          )
        endif()

        file(READ ${OSHMEM_INFO_OUTPUT} OSHMEM_INFO_OUTPUT_CONTENT)

        if(NOT DEFINED OSHMEM_INFO_OUTPUT_CONTENT)
          message(
            FATAL_ERROR
              "${OSHMEM_INFO} Failed! Check: ${OSHMEM_INFO_ERROR}\n${OSHMEM_INFO_OUTPUT_CONTENT}"
          )
        endif()

        if("${OSHMEM_INFO_OUTPUT_CONTENT}" STREQUAL "")
          message(
            FATAL_ERROR
              "${OSHMEM_INFO} Failed! Check: ${OSHMEM_INFO_ERROR}\n${OSHMEM_INFO_OUTPUT_CONTENT}"
          )
        endif()

        string(REGEX MATCH "(\/.*)" OSHMEM_LIBDIR_PATH
                     ${OSHMEM_INFO_OUTPUT_CONTENT}
        )

        string(STRIP ${OSHMEM_LIBDIR_PATH} OSHMEM_LIBDIR_PATH)

        set(ENV{PKG_CONFIG_PATH}
            "$ENV{PKG_CONFIG_PATH}:${OSHMEM_LIBDIR_PATH}/pkgconfig"
        )

        pkg_search_module(OPENSHMEM IMPORTED_TARGET GLOBAL ${OPENSHMEM_PC})

        if(NOT OPENSHMEM_FOUND)

          set(OSHMEM_INFO_INCOUTPUT
              "${CMAKE_CURRENT_SOURCE_DIR}/oshmem_info_stdout_inc.log"
          )
          set(OSHMEM_INFO_INCERROR
              "${CMAKE_CURRENT_SOURCE_DIR}/oshmem_info_error_inc.log"
          )

          execute_process(
            COMMAND bash -c "${OSHMEM_INFO} --path incdir"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE OSHMEM_INFO_INCSTATUS
            OUTPUT_FILE ${OSHMEM_INFO_INCOUTPUT}
            ERROR_FILE ${OSHMEM_INFO_INCERROR}
          )

          if(OSHMEM_INFO_INCSTATUS)
            message(
              FATAL_ERROR
                "${OSHMEM_INFO} Failed! Program status code: ${OSHMEM_INFO_INCSTATUS}"
            )
          endif()
          file(READ ${OSHMEM_INFO_INCOUTPUT} OSHMEM_INFO_OUTPUT_INCCONTENT)

          if(NOT DEFINED OSHMEM_INFO_OUTPUT_INCCONTENT)
            message(
              FATAL_ERROR
                "${OSHMEM_INFO} Failed! Check: ${OSHMEM_INFO_INCERROR}"
            )
          endif()

          if("${OSHMEM_INFO_OUTPUT_INCCONTENT}" STREQUAL "")
            message(
              FATAL_ERROR
                "${OSHMEM_INFO} Failed! Check: ${OSHMEM_INFO_INCERROR}\n${OSHMEM_INFO_OUTPUT_INCCONTENT}"
            )
          endif()

          string(REGEX MATCH "(\/.*)" OSHMEM_INCDIR_PATH
                       ${OSHMEM_INFO_OUTPUT_INCCONTENT}
          )

          string(STRIP ${OSHMEM_INCDIR_PATH} OSHMEM_INCDIR_PATH)

          set(OPENSHMEM_CFLAGS
              "-I${OSHMEM_INCDIR_PATH} -pthread -I${OSHMEM_LIBDIR_PATH}"
          )
          set(OPENSHMEM_LDFLAGS "-loshmem")
          set(OPENSHMEM_LIBRARY_DIRS "${OSHMEM_LIBDIR_PATH}")

          add_library(PkgConfig::OPENSHMEM INTERFACE IMPORTED GLOBAL)

          set(OPENSHMEM_FOUND ON)
        endif()
      endif()
    else()

      include(cmake/FindOpenShmemPmi.cmake)

      set(PMI_AUTOCONF_OPTS "")
      if(NOT PMI_LIBRARY OR NOT PMI_FOUND)
        set(PMI_AUTOCONF_OPTS "--enable-pmi-simple")
      else()
        string(REGEX MATCH "(^\/[^\/]+)" PMI_INCLUDE_DIR_ROOT_PATH
                     ${PMI_INCLUDE_DIR}
        )
        string(REGEX MATCH "(^\/[^\/]+)" PMI_LIBRARY_ROOT_PATH ${PMI_LIBRARY})
        set(PMI_AUTOCONF_OPTS
            "--with-pmi=${PMI_INCLUDE_DIR_ROOT_PATH} --with-pmi-libdir=${PMI_LIBRARY_ROOT_PATH}"
        )
      endif()

      set(OPENSHMEM_PC "osss-ucx")

      pkg_search_module(OPENSHMEM IMPORTED_TARGET GLOBAL ${OPENSHMEM_PC})
      if(NOT OPENSHMEM_FOUND)
        set(OPENSHMEM_PC "sandia-openshmem")
        pkg_search_module(OPENSHMEM IMPORTED_TARGET GLOBAL ${OPENSHMEM_PC})
      endif()
    endif()
  endif()

  if(OPENSHMEM_CFLAGS)
    set(IS_PARAM "0")
    set(PARAM_FOUND "0")
    set(NEWPARAM "")
    set(IDX 0)
    set(FLAG_LIST "")

    foreach(X IN ITEMS ${OPENSHMEM_CFLAGS})
      string(FIND "${X}" "--param" PARAM_FOUND)
      if(NOT "${PARAM_FOUND}" EQUAL "-1")
        set(IS_PARAM "1")
        set(NEWPARAM "SHELL:${X}")
      endif()
      if("${PARAM_FOUND}" EQUAL "-1"
         AND "${IS_PARAM}" EQUAL "0"
         OR "${IS_PARAM}" EQUAL "-1"
      )
        list(APPEND FLAG_LIST "${X}")
        set(IS_PARAM "0")
      elseif("${PARAM_FOUND}" EQUAL "-1" AND "${IS_PARAM}" EQUAL "1")
        list(APPEND FLAG_LIST "${NEWPARAM}
          ${X}"
        )
        set(NEWPARAM "")
        set(IS_PARAM "0")
      endif()
    endforeach()

    list(LENGTH OPENSHMEM_CFLAGS IDX)
    foreach(X RANGE ${IDX})
      list(POP_FRONT OPENSHMEM_CFLAGS NEWPARAM)
    endforeach()

    foreach(X IN ITEMS ${FLAG_LIST})
      list(APPEND OPENSHMEM_CFLAGS "${X}")
    endforeach()
  endif()

  if(OPENSHMEM_CFLAGS_OTHER)
    set(IS_PARAM "0")
    set(PARAM_FOUND "0")
    set(NEWPARAM "")
    set(IDX 0)
    set(FLAG_LIST "")

    foreach(X IN ITEMS ${OPENSHMEM_CFLAGS_OTHER})
      string(FIND "${X}" "--param" PARAM_FOUND)
      if(NOT "${PARAM_FOUND}" EQUAL "-1")
        set(IS_PARAM "1")
        set(NEWPARAM "SHELL:${X}")
      endif()
      if("${PARAM_FOUND}" EQUAL "-1"
         AND "${IS_PARAM}" EQUAL "0"
         OR "${IS_PARAM}" EQUAL "-1"
      )
        list(APPEND FLAG_LIST "${X}")
        set(IS_PARAM "0")
      elseif("${PARAM_FOUND}" EQUAL "-1" AND "${IS_PARAM}" EQUAL "1")
        list(APPEND FLAG_LIST "${NEWPARAM}
          ${X}"
        )
        set(NEWPARAM "")
        set(IS_PARAM "0")
      endif()
    endforeach()

    list(LENGTH OPENSHMEM_CFLAGS_OTHER IDX)
    foreach(X RANGE ${IDX})
      list(POP_FRONT OPENSHMEM_CFLAGS_OTHER NEWPARAM)
    endforeach()

    foreach(X IN ITEMS ${FLAG_LIST})
      list(APPEND OPENSHMEM_CFLAGS_OTHER "${X}")
    endforeach()
  endif()

  if(OPENSHMEM_LDFLAGS)
    set(IS_PARAM "0")
    set(PARAM_FOUND "0")
    set(NEWPARAM "")
    set(IDX 0)
    set(DIRIDX 0)
    set(SKIP 0)
    set(FLAG_LIST "")
    set(DIR_LIST "")
    set(LIB_LIST "")

    foreach(X IN ITEMS ${OPENSHMEM_LDFLAGS})
      string(FIND "${X}" "--param" PARAM_FOUND)
      string(FIND "${X}" "-lsma" IDX)
      string(FIND "${X}" "-l" LIDX)
      string(FIND "${X}" "-L" DIRIDX)
      string(FIND "${X}" "-Wl" SKIP)

      if("${SKIP}" EQUAL "-1")
        if(NOT "${PARAM_FOUND}" EQUAL "-1")
          set(IS_PARAM "1")
          set(NEWPARAM "SHELL:${X}")
        endif()
        if("${PARAM_FOUND}" EQUAL "-1"
           AND "${IDX}" EQUAL "-1"
           AND "${IS_PARAM}" EQUAL "0"
           OR "${IS_PARAM}" EQUAL "-1"
        )
          list(APPEND FLAG_LIST "${X}")
          set(IS_PARAM "0")
        elseif("${PARAM_FOUND}" EQUAL "-1" AND "${IS_PARAM}" EQUAL "1")
          list(APPEND FLAG_LIST "${NEWPARAM}
          ${X}"
          )
          set(NEWPARAM "")
          set(IS_PARAM "0")
        elseif(NOT "${IDX}" EQUAL "-1" AND NOT "${LIDX}" EQUAL "-1")
          set(TMPSTR "")
          string(REPLACE "-l" "" TMPSTR "${X}")
          list(APPEND LIB_LIST "${TMPSTR}")
          set(IDX 0)
        elseif("${IDX}" EQUAL "-1" AND NOT "${LIDX}" EQUAL "-1")
          list(APPEND FLAG_LIST "${X}")
        endif()
        if(NOT "${DIRIDX}" EQUAL "-1")
          set(TMPSTR "")
          string(REPLACE "-L" "" TMPSTR "${X}")
          list(APPEND DIR_LIST "${TMPSTR}")
        endif()
      endif()
    endforeach()

    set(IDX 0)
    list(LENGTH LIB_LIST IDX)

    if(NOT "${IDX}" EQUAL "0")
      set(IDX 0)

      if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(NEWLINK "SHELL:-Wl,--whole-archive
          "
        )
        foreach(X IN ITEMS ${LIB_LIST})
          set(DIRSTR "")
          string(REPLACE ";" "
          " DIRSTR "${DIR_LIST}"
          )
          foreach(Y IN ITEMS ${DIR_LIST})
            find_library(
              FOUND_LIB
              NAMES ${X} "lib${X}" "lib${X}.a"
              PATHS ${Y}
              HINTS ${Y} NO_CACHE
              NO_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH
            )

            list(LENGTH FOUND_LIB IDX)
            if(NOT "${IDX}" EQUAL "0")
              string(APPEND NEWLINK "${FOUND_LIB}")
              set(FOUND_LIB "")
            endif()
          endforeach()
        endforeach()
        string(APPEND NEWLINK "
          -Wl,--no-whole-archive"
        )
        string(FIND "SHELL:-Wl,--whole-archive
          -Wl,--no-whole-archive" "${NEWLINK}" IDX
        )
        if("${IDX}" EQUAL "-1")
          list(APPEND OPENSHMEM_LDFLAGS "${NEWLINK}")
       endif()
      elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(APPLE)
          set(NEWLINK "SHELL:-Wl,-force_load,")
        else()
          set(NEWLINK "SHELL:
          "
          )
        endif()
        foreach(X IN ITEMS ${LIB_LIST})
          set(DIRSTR "")
          string(REPLACE ";" "
          " DIRSTR "${DIR_LIST}"
          )
          foreach(Y IN ITEMS ${DIR_LIST})
            find_library(
              FOUND_LIB
              NAMES ${X} "lib${X}" "lib${X}.a"
              PATHS ${Y}
              HINTS ${Y} NO_CACHE
              NO_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH
            )

            list(LENGTH FOUND_LIB IDX)
            if(NOT "${IDX}" EQUAL "0")
              string(APPEND NEWLINK "${FOUND_LIB}")
              set(FOUND_LIB "")
            endif()
          endforeach()
        endforeach()
        string(FIND "SHELL:" "${NEWLINK}" IDX)
        if("${IDX}" EQUAL "-1")
          list(APPEND OPENSHMEM_LDFLAGS "${NEWLINK}")
        endif()
      endif()
    endif()
  endif()

  if(OPENSHMEM_LDFLAGS_OTHER)
    unset(FOUND_LIB)
    set(IS_PARAM "0")
    set(PARAM_FOUND "0")
    set(NEWPARAM "")
    set(SKIP 0)
    set(IDX 0)
    set(DIRIDX 0)
    set(FLAG_LIST "")
    set(DIR_LIST "")
    set(LIB_LIST "")

    foreach(X IN ITEMS ${OPENSHMEM_LDFLAGS_OTHER})
      string(FIND "${X}" "--param" PARAM_FOUND)
      string(FIND "${X}" "-lsma" IDX)
      string(FIND "${X}" "-L" DIRIDX)
      string(FIND "${X}" "-Wl" SKIP)

      if("${SKIP}" EQUAL "-1")
        if(NOT "${PARAM_FOUND}" EQUAL "-1")
          set(IS_PARAM "1")
          set(NEWPARAM "SHELL:${X}")
        endif()
        if("${PARAM_FOUND}" EQUAL "-1"
           AND "${IDX}" EQUAL "-1"
           AND "${IS_PARAM}" EQUAL "0"
           OR "${IS_PARAM}" EQUAL "-1"
        )
          list(APPEND FLAG_LIST "${X}")
          set(IS_PARAM "0")
        elseif("${PARAM_FOUND}" EQUAL "-1" AND "${IS_PARAM}" EQUAL "1")
          list(APPEND FLAG_LIST "${NEWPARAM}
          ${X}"
          )
          set(NEWPARAM "")
          set(IS_PARAM "0")
        elseif(NOT "${IDX}" EQUAL "-1" AND NOT "${LIDX}" EQUAL "-1")
          set(TMPSTR "")
          string(REPLACE "-l" "" TMPSTR "${X}")
          list(APPEND LIB_LIST "${TMPSTR}")
          set(IDX 0)
        elseif("${IDX}" EQUAL "-1" AND NOT "${LIDX}" EQUAL "-1")
          list(APPEND FLAG_LIST "${X}")
        endif()
        if(NOT "${DIRIDX}" EQUAL "-1")
          set(TMPSTR "")
          string(REPLACE "-L" "" TMPSTR "${X}")
          list(APPEND DIR_LIST "${TMPSTR}")
        endif()
      endif()
    endforeach()

    set(IDX 0)
    list(LENGTH OPENSHMEM_LDFLAGS_OTHER IDX)
    foreach(X RANGE ${IDX})
      list(POP_FRONT OPENSHMEM_LDFLAGS_OTHER NEWPARAM)
    endforeach()

    foreach(X IN ITEMS ${FLAG_LIST})
      list(APPEND OPENSHMEM_LDFLAGS_OTHER "${X}")
    endforeach()

    set(IDX 0)
    list(LENGTH LIB_LIST IDX)
    if(NOT "${IDX}" EQUAL "0")
      set(IDX 0)
      if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(NEWLINK "SHELL:-Wl,--whole-archive
          "
        )
        foreach(X IN ITEMS ${LIB_LIST})
          set(DIRSTR "")
          string(REPLACE ";" "
          " DIRSTR "${DIR_LIST}"
          )
          foreach(Y IN ITEMS ${DIR_LIST})
            find_library(
              FOUND_LIB
              NAMES ${X} "lib${X}" "lib${X}.a"
              PATHS ${Y}
              HINTS ${Y} NO_CACHE
              NO_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH
            )

            list(LENGTH FOUND_LIB IDX)
            if(NOT "${IDX}" EQUAL "0")
              string(APPEND NEWLINK "${FOUND_LIB}")
              set(FOUND_LIB "")
            endif()
          endforeach()
        endforeach()
        string(APPEND NEWLINK "
          -Wl,--no-whole-archive"
        )

        string(FIND "SHELL:-Wl,--whole-archive
          -Wl,--no-whole-archive" "${NEWLINK}" IDX
        )
        if("${IDX}" EQUAL "-1")
          list(APPEND OPENSHMEM_LDFLAGS_OTHER "${NEWLINK}")
        endif()
      elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(APPLE)
          set(NEWLINK "SHELL:-Wl,-force_load,")
        else()
          set(NEWLINK "SHELL:
          "
          )
        endif()
        foreach(X IN ITEMS ${LIB_LIST})
          set(DIRSTR "")
          string(REPLACE ";" "
          " DIRSTR "${DIR_LIST}"
          )
          foreach(Y IN ITEMS ${DIR_LIST})
            find_library(
              FOUND_LIB
              NAMES ${X} "lib${X}" "lib${X}.a"
              PATHS ${Y}
              HINTS ${Y} NO_CACHE
              NO_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH
            )

            list(LENGTH FOUND_LIB IDX)
            if(NOT "${IDX}" EQUAL "0")
              string(APPEND NEWLINK "${FOUND_LIB}")
              set(FOUND_LIB "")
            endif()
          endforeach()
        endforeach()
        string(FIND "SHELL:" "${NEWLINK}" IDX)
        if("${IDX}" EQUAL "-1")
          list(APPEND OPENSHMEM_LDFLAGS "${NEWLINK}")
        endif()
      endif()
    endif()
  endif()

  if(OPENSHMEM_STATIC_CFLAGS)
    set(IS_PARAM "0")
    set(PARAM_FOUND "0")
    set(NEWPARAM "")
    set(IDX 0)
    set(FLAG_LIST "")

    foreach(X IN ITEMS ${OPENSHMEM_STATIC_CFLAGS})
      string(FIND "${X}" "--param" PARAM_FOUND)
      if(NOT "${PARAM_FOUND}" EQUAL "-1")
        set(IS_PARAM "1")
        set(NEWPARAM "SHELL:${X}")
      endif()
      if("${PARAM_FOUND}" EQUAL "-1"
         AND "${IS_PARAM}" EQUAL "0"
         OR "${IS_PARAM}" EQUAL "-1"
      )
        list(APPEND FLAG_LIST "${X}")
        set(IS_PARAM "0")
      elseif("${PARAM_FOUND}" EQUAL "-1" AND "${IS_PARAM}" EQUAL "1")
        list(APPEND FLAG_LIST "${NEWPARAM}
          ${X}"
        )
        set(NEWPARAM "")
        set(IS_PARAM "0")
      endif()
    endforeach()

    list(LENGTH OPENSHMEM_STATIC_CFLAGS IDX)
    foreach(X RANGE ${IDX})
      list(POP_FRONT OPENSHMEM_STATIC_CFLAGS NEWPARAM)
    endforeach()

    foreach(X IN ITEMS ${FLAG_LIST})
      list(APPEND OPENSHMEM_STATIC_CFLAGS "${X}")
    endforeach()
  endif()

  if(OPENSHMEM_STATIC_CFLAGS_OTHER)
    set(IS_PARAM "0")
    set(PARAM_FOUND "0")
    set(NEWPARAM "")
    set(IDX 0)
    set(FLAG_LIST "")
   foreach(X IN ITEMS ${OPENSHMEM_STATIC_CFLAGS_OTHER})
      string(FIND "${X}" "--param" PARAM_FOUND)
      if(NOT "${PARAM_FOUND}" EQUAL "-1")
        set(IS_PARAM "1")
        set(NEWPARAM "SHELL:${X}")
      endif()
      if("${PARAM_FOUND}" EQUAL "-1"
         AND "${IS_PARAM}" EQUAL "0"
         OR "${IS_PARAM}" EQUAL "-1"
      )
        list(APPEND FLAG_LIST "${X}")
        set(IS_PARAM "0")
      elseif("${PARAM_FOUND}" EQUAL "-1" AND "${IS_PARAM}" EQUAL "1")
        list(APPEND FLAG_LIST "${NEWPARAM}
          ${X}"
        )
        set(NEWPARAM "")
        set(IS_PARAM "0")
      endif()
    endforeach()

    list(LENGTH OPENSHMEM_STATIC_CFLAGS_OTHER IDX)
    foreach(X RANGE ${IDX})
      list(POP_FRONT OPENSHMEM_STATIC_CFLAGS_OTHER NEWPARAM)
    endforeach()

    foreach(X IN ITEMS ${FLAG_LIST})
      list(APPEND OPENSHMEM_STATIC_CFLAGS_OTHER "${X}")
    endforeach()
  endif()

  if(OPENSHMEM_STATIC_LDFLAGS)
    unset(FOUND_LIB)
    set(IS_PARAM "0")
    set(PARAM_FOUND "0")
    set(NEWPARAM "")
    set(SKIP 0)
    set(IDX 0)
    set(DIRIDX 0)
    set(FLAG_LIST "")
    set(DIR_LIST "")
    set(LIB_LIST "")
    foreach(X IN ITEMS ${OPENSHMEM_STATIC_LDFLAGS})
      string(FIND "${X}" "--param" PARAM_FOUND)
      if("${HPX_WITH_PARCELPORT_OPENSHMEM_CONDUIT}" STREQUAL "mpi")
        string(FIND "${X}" "-loshmem" IDX)
      else()
        string(FIND "${X}" "-lsma" IDX)
      endif()
      string(FIND "${X}" "-L" DIRIDX)
      string(FIND "${X}" "-Wl" SKIP)

      if("${SKIP}" EQUAL "-1")
        if(NOT "${PARAM_FOUND}" EQUAL "-1")
          set(IS_PARAM "1")
          set(NEWPARAM "SHELL:${X}")
        endif()
        if("${PARAM_FOUND}" EQUAL "-1"
           AND "${IDX}" EQUAL "-1"
           AND "${IS_PARAM}" EQUAL "0"
           OR "${IS_PARAM}" EQUAL "-1"
        )
          list(APPEND FLAG_LIST "${X}")
          set(IS_PARAM "0")
        elseif("${PARAM_FOUND}" EQUAL "-1" AND "${IS_PARAM}" EQUAL "1")
          list(APPEND FLAG_LIST "${NEWPARAM}
          ${X}"
          )
          set(NEWPARAM "")
          set(IS_PARAM "0")
        elseif(NOT "${IDX}" EQUAL "-1" AND NOT "${LIDX}" EQUAL "-1")
          set(TMPSTR "")
          string(REPLACE "-l" "" TMPSTR "${X}")
          list(APPEND LIB_LIST "${TMPSTR}")
          set(IDX 0)
        elseif("${IDX}" EQUAL "-1" AND NOT "${LIDX}" EQUAL "-1")
          list(APPEND FLAG_LIST "${X}")
        endif()
        if(NOT "${DIRIDX}" EQUAL "-1")
          set(TMPSTR "")
          string(REPLACE "-L" "" TMPSTR "${X}")
          list(APPEND DIR_LIST "${TMPSTR}")
        endif()
      endif()
    endforeach()
    set(IDX 0)
    list(LENGTH OPENSHMEM_STATIC_LDFLAGS IDX)
    foreach(X RANGE ${IDX})
      list(POP_FRONT OPENSHMEM_STATIC_LDFLAGS NEWPARAM)
    endforeach()

    foreach(X IN ITEMS ${FLAG_LIST})
      list(APPEND OPENSHMEM_STATIC_LDFLAGS "${X}")
    endforeach()

    set(IDX 0)
    list(LENGTH LIB_LIST IDX)
    if(NOT "${IDX}" EQUAL "0")
      set(IDX 0)
      if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(NEWLINK "SHELL:-Wl,--whole-archive
          "
        )
        foreach(X IN ITEMS ${LIB_LIST})
          set(DIRSTR "")
          string(REPLACE ";" "
          " DIRSTR "${DIR_LIST}"
          )
          foreach(Y IN ITEMS ${DIR_LIST})
            find_library(
              FOUND_LIB
              NAMES ${X} "lib${X}" "lib${X}.a"
              PATHS ${Y}
              HINTS ${Y} NO_CACHE
              NO_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH
            )

            list(LENGTH FOUND_LIB IDX)

            if(NOT "${IDX}" EQUAL "0")
              string(APPEND NEWLINK "${FOUND_LIB}")
              set(FOUND_LIB "")
            endif()
          endforeach()
        endforeach()
        string(APPEND NEWLINK "
          -Wl,--no-whole-archive"
        )

        string(FIND "SHELL:-Wl,--whole-archive
          -Wl,--no-whole-archive" "${NEWLINK}" IDX
        )
        if("${IDX}" EQUAL "-1")
          list(APPEND OPENSHMEM_STATIC_LDFLAGS "${NEWLINK}")
        endif()
     elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(APPLE)
          set(NEWLINK "SHELL:-Wl,-force_load,")
        else()
          set(NEWLINK "SHELL:
          "
          )
        endif()
        foreach(X IN ITEMS ${LIB_LIST})
          set(DIRSTR "")
          string(REPLACE ";" "
          " DIRSTR "${DIR_LIST}"
          )
          foreach(Y IN ITEMS ${DIR_LIST})
            find_library(
              FOUND_LIB
              NAMES ${X} "lib${X}" "lib${X}.a"
              PATHS ${Y}
              HINTS ${Y} NO_CACHE
              NO_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH
            )

            list(LENGTH FOUND_LIB IDX)
            if(NOT "${IDX}" EQUAL "0")
              string(APPEND NEWLINK "${FOUND_LIB}")
              set(FOUND_LIB "")
            endif()
          endforeach()
        endforeach()
        string(FIND "SHELL:" "${NEWLINK}" IDX)
        if("${IDX}" EQUAL "-1")
          list(APPEND OPENSHMEM_LDFLAGS "${NEWLINK}")
        endif()
      endif()
    endif()
  endif()

  if(OPENSHMEM_STATIC_LDFLAGS_OTHER)
    unset(FOUND_LIB)
    set(IS_PARAM "0")
    set(PARAM_FOUND "0")
    set(NEWPARAM "")
    set(SKIP 0)
    set(IDX 0)
    set(DIRIDX 0)
    set(FLAG_LIST "")
    set(DIR_LIST "")
    set(LIB_LIST "")

    foreach(X IN ITEMS ${OPENSHMEM_STATIC_LDFLAGS_OTHER})
      string(FIND "${X}" "--param" PARAM_FOUND)
      if("${HPX_WITH_PARCELPORT_OPENSHMEM_CONDUIT}" STREQUAL "mpi")
        string(FIND "${X}" "-loshmem" IDX)
      else()
        string(FIND "${X}" "-lsma" IDX)
      endif()
      string(FIND "${X}" "-L" DIRIDX)
      string(FIND "${X}" "-Wl" SKIP)

      if("${SKIP}" EQUAL "-1")
        if(NOT "${PARAM_FOUND}" EQUAL "-1")
          set(IS_PARAM "1")
          set(NEWPARAM "SHELL:${X}")
        endif()
        if("${PARAM_FOUND}" EQUAL "-1"
           AND "${IDX}" EQUAL "-1"
           AND "${IS_PARAM}" EQUAL "0"
           OR "${IS_PARAM}" EQUAL "-1"
        )
          list(APPEND FLAG_LIST "${X}")
          set(IS_PARAM "0")
        elseif("${PARAM_FOUND}" EQUAL "-1" AND "${IS_PARAM}" EQUAL "1")
          list(APPEND FLAG_LIST "${NEWPARAM}
          ${X}"
          )
          set(NEWPARAM "")
          set(IS_PARAM "0")
        elseif(NOT "${IDX}" EQUAL "-1" AND NOT "${LIDX}" EQUAL "-1")
          set(TMPSTR "")
          string(REPLACE "-l" "" TMPSTR "${X}")
          list(APPEND LIB_LIST "${TMPSTR}")
          set(IDX 0)
        elseif("${IDX}" EQUAL "-1" AND NOT "${LIDX}" EQUAL "-1")
          list(APPEND FLAG_LIST "${X}")
        endif()
        if(NOT "${DIRIDX}" EQUAL "-1")
          set(TMPSTR "")
          string(REPLACE "-L" "" TMPSTR "${X}")
          list(APPEND DIR_LIST "${TMPSTR}")
        endif()
      endif()
    endforeach()

    set(IDX 0)
    list(LENGTH OPENSHMEM_STATIC_LDFLAGS_OTHER IDX)
    foreach(X RANGE ${IDX})
      list(POP_FRONT OPENSHMEM_STATIC_LDFLAGS_OTHER NEWPARAM)
    endforeach()

    foreach(X IN ITEMS ${FLAG_LIST})
      list(APPEND OPENSHMEM_STATIC_LDFLAGS_OTHER "${X}")
    endforeach()

    set(IDX 0)
    list(LENGTH LIB_LIST IDX)
    if(NOT "${IDX}" EQUAL "0")
      set(IDX 0)
      if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(NEWLINK "SHELL:-Wl,--whole-archive
          "
        )
        foreach(X IN ITEMS ${LIB_LIST})
          set(DIRSTR "")
          string(REPLACE ";" "
          " DIRSTR "${DIR_LIST}"
          )
          foreach(Y IN ITEMS ${DIR_LIST})
            find_library(
              FOUND_LIB
              NAMES ${X} "lib${X}" "lib${X}.a"
              PATHS ${Y}
              HINTS ${Y} NO_CACHE
              NO_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH
            )

            list(LENGTH FOUND_LIB IDX)

            message(STATUS "${FOUND_LIB}
          ${X}"
            )
            if(NOT "${IDX}" EQUAL "0")
              string(APPEND NEWLINK "${FOUND_LIB}")
              set(FOUND_LIB "")
            endif()
          endforeach()
        endforeach()
        string(APPEND NEWLINK "
          -Wl,--no-whole-archive"
        )
        string(FIND "SHELL:-Wl,--whole-archive
          -Wl,--no-whole-archive" "${NEWLINK}" IDX
        )
        if("${IDX}" EQUAL "-1")
          list(APPEND OPENSHMEM_STATIC_LDFLAGS_OTHER "${NEWLINK}")
        endif()
      elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(APPLE)
          set(NEWLINK "SHELL:-Wl,-force_load,")
        else()
          set(NEWLINK "SHELL:
          "
          )
        endif()
        foreach(X IN ITEMS ${LIB_LIST})
          set(DIRSTR "")
          string(REPLACE ";" "
          " DIRSTR "${DIR_LIST}"
          )
          foreach(Y IN ITEMS ${DIR_LIST})
            find_library(
              FOUND_LIB
              NAMES ${X} "lib${X}" "lib${X}.a"
              PATHS ${Y}
              HINTS ${Y} NO_CACHE
              NO_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH
            )

            list(LENGTH FOUND_LIB IDX)
            if(NOT "${IDX}" EQUAL "0")
              string(APPEND NEWLINK "${FOUND_LIB}")
              set(FOUND_LIB "")
            endif()
          endforeach()
        endforeach()
        string(FIND "SHELL:" "${NEWLINK}" IDX)
        if("${IDX}" EQUAL "-1")
          list(APPEND OPENSHMEM_LDFLAGS "${NEWLINK}")
        endif()
      endif()
    endif()
  endif()

  if(OPENSHMEM_DIR)
    list(TRANSFORM OPENSHMEM_CFLAGS
         REPLACE "${OPENSHMEM_DIR}/install"
                 "$<BUILD_INTERFACE:${OPENSHMEM_DIR}/install>"
    )
    list(TRANSFORM OPENSHMEM_LDFLAGS
         REPLACE "${OPENSHMEM_DIR}/install"
                 "$<BUILD_INTERFACE:${OPENSHMEM_DIR}/install>"
    )
    list(TRANSFORM OPENSHMEM_LIBRARY_DIRS
         REPLACE "${OPENSHMEM_DIR}/install"
                 "$<BUILD_INTERFACE:${OPENSHMEM_DIR}/install>"
    )

    message(STATUS "OPENSHMEM_CFLAGS:\t${OPENSHMEM_CFLAGS}")
    message(STATUS "OPENSHMEM_LDFLAGS:\t${OPENSHMEM_LDFLAGS}")
    message(STATUS "OPENSHMEM_LIBRARY_DIRS:\t${OPENSHMEM_LIBRARY_DIRS}")

    set_target_properties(
      PkgConfig::OPENSHMEM PROPERTIES INTERFACE_COMPILE_OPTIONS
                                      "${OPENSHMEM_CFLAGS}"
    )
    set_target_properties(
      PkgConfig::OPENSHMEM PROPERTIES INTERFACE_LINK_OPTIONS
                                      "${OPENSHMEM_LDFLAGS}"
    )
    set_target_properties(
      PkgConfig::OPENSHMEM PROPERTIES INTERFACE_LINK_DIRECTORIES
                                      "${OPENSHMEM_LIBRARY_DIRS}"
    )
    set(OPENSHMEM_FOUND ON)
  else()
    message(STATUS "OPENSHMEM_CFLAGS:\t${OPENSHMEM_CFLAGS}")
    message(STATUS "OPENSHMEM_LDFLAGS:\t${OPENSHMEM_LDFLAGS}")
    message(STATUS "OPENSHMEM_LIBRARY_DIRS:\t${OPENSHMEM_LIBRARY_DIRS}")

    set_target_properties(
      PkgConfig::OPENSHMEM PROPERTIES INTERFACE_COMPILE_OPTIONS
                                      "${OPENSHMEM_CFLAGS}"
    )
    set_target_properties(
      PkgConfig::OPENSHMEM PROPERTIES INTERFACE_LINK_OPTIONS
                                      "${OPENSHMEM_LDFLAGS}"
    )
    set_target_properties(
      PkgConfig::OPENSHMEM PROPERTIES INTERFACE_LINK_DIRECTORIES
                                      "${OPENSHMEM_LIBRARY_DIRS}"
    )
    set(OPENSHMEM_FOUND ON)
  endif()
endmacro()
