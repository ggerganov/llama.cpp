# - Find OCaml
# Try to find OCaml.
# 
# The following variables are defined:
#  CMAKE_OCaml_EXECUTABLE - The Objective Caml top level
#  CMAKE_OCaml_LEX - The Objective Caml lexer generator
#  CMAKE_OCaml_YACC - The Objective Caml parser generator
#  CMAKE_OCaml_FIND - The Objective Caml package manager
#  CMAKE_OCaml_VERSION - The Objective Caml version
# 
# If CMAKE_OCaml_FIND is not defined, the following variables are also defined
#  CMAKE_OCaml_COMPILER - The Objective Caml bytecode compiler
#  CMAKE_OCaml_OPT_COMPILER - The Objective Caml native-code compiler
#  CMAKE_OCaml_DEP - Dependency generator for Objective Caml
# 
# Copyright (c) 2010-2014, Judicaël Bedouet, j dot bedouet at infonie dot fr.
# 
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

find_program(CMAKE_OCaml_EXECUTABLE ocaml)

if(CMAKE_OCaml_EXECUTABLE)
  get_filename_component(OCaml_ROOT_DIR ${CMAKE_OCaml_EXECUTABLE} PATH)
  get_filename_component(OCaml_ROOT_DIR ${OCaml_ROOT_DIR}         PATH)
endif()

find_program(CMAKE_OCaml_FIND ocamlfind
  HINTS         ${OCaml_ROOT_DIR}
  PATH_SUFFIXES bin
  )
  
if (WIN32)
	find_program (CMAKE_Flexlink_EXECUTABLE flexlink)
endif(WIN32)

if(CMAKE_OCaml_EXECUTABLE AND NOT CMAKE_OCaml_FIND)
  
  find_program(CMAKE_OCaml_COMPILER ocamlc.opt ocamlc
    HINTS         ${OCaml_ROOT_DIR}
    PATH_SUFFIXES bin
    )
  
  find_program(CMAKE_OCaml_OPT_COMPILER ocamlopt.opt ocamlopt
    HINTS         ${OCaml_ROOT_DIR}
    PATH_SUFFIXES bin
    )
  
  find_program(CMAKE_OCaml_DEP ocamldep.opt ocamldep
    HINTS         ${OCaml_ROOT_DIR}
    PATH_SUFFIXES bin
    )
  
  if(CMAKE_OCaml_COMPILER)
    set(TMP_VERSION_CMD ${CMAKE_OCaml_COMPILER})
  endif()
  
else()
  set(TMP_VERSION_CMD ${CMAKE_OCaml_FIND} ocamlc)
endif()

find_program(CMAKE_OCaml_LEX ocamllex.opt ocamllex
  HINTS         ${OCaml_ROOT_DIR}
  PATH_SUFFIXES bin
  )

find_program(CMAKE_OCaml_YACC ocamlyacc
  HINTS         ${OCaml_ROOT_DIR}
  PATH_SUFFIXES bin
  )

if(TMP_VERSION_CMD)
  
  execute_process(
    COMMAND         ${TMP_VERSION_CMD} -version
    OUTPUT_VARIABLE CMAKE_OCaml_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  
 execute_process(
   COMMAND         ${TMP_VERSION_CMD} -where
   OUTPUT_VARIABLE CMAKE_OCaml_STD_LIBRARY_PATH
   OUTPUT_STRIP_TRAILING_WHITESPACE
   )
  
endif()

include (FindPackageHandleStandardArgs)

if(CMAKE_OCaml_EXECUTABLE AND NOT CMAKE_OCaml_FIND)
  
  find_package_handle_standard_args(OCaml "Could NOT find OCaml. Please specify CMAKE_OCaml_EXECUTABLE."
    CMAKE_OCaml_VERSION
    CMAKE_OCaml_EXECUTABLE
    CMAKE_OCaml_COMPILER
    CMAKE_OCaml_OPT_COMPILER
    CMAKE_OCaml_LEX
    CMAKE_OCaml_YACC
    CMAKE_OCaml_DEP
    )
  
  mark_as_advanced(
    CMAKE_OCaml_COMPILER
    CMAKE_OCaml_OPT_COMPILER
    CMAKE_OCaml_DEP
    )
  
else()
  
  find_package_handle_standard_args(OCaml "Could NOT find OCaml. Please specify CMAKE_OCaml_EXECUTABLE."
    CMAKE_OCaml_VERSION
    CMAKE_OCaml_EXECUTABLE
    CMAKE_OCaml_FIND
    CMAKE_OCaml_LEX
    CMAKE_OCaml_YACC
    )
  
  mark_as_advanced(CMAKE_OCaml_FIND)
  
endif()

mark_as_advanced(
  CMAKE_OCaml_EXECUTABLE
  CMAKE_OCaml_LEX
  CMAKE_OCaml_YACC
  )
