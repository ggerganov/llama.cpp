# - Find CamlIDL
# Try to find camlidl.
#
# The following variables are defined:
#  CAMLIDL_EXECUTABLE - The camlidl executable
#
# Copyright (c) 2010, Judicaël Bedouet, j dot bedouet at infonie dot fr.
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

find_program (CAMLIDL_EXECUTABLE camlidl)

if (CAMLIDL_EXECUTABLE)
  get_filename_component (CamlIDL_ROOT_DIR ${CAMLIDL_EXECUTABLE} PATH)
  get_filename_component (CamlIDL_ROOT_DIR ${CamlIDL_ROOT_DIR}   PATH)
endif (CAMLIDL_EXECUTABLE)

find_library (CAMLIDL_LIBRARY camlidl
  HINTS         ${CamlIDL_ROOT_DIR}
  PATH_SUFFIXES lib/ocaml
  )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args (CamlIDL DEFAULT_MSG
  CAMLIDL_EXECUTABLE
  CAMLIDL_LIBRARY
)

mark_as_advanced (
  CAMLIDL_EXECUTABLE
  CAMLIDL_LIBRARY
)

macro (gen_caml_idl gen_c_files gen_ocaml_files)
  foreach (_idl_file ${ARGN})
    
    get_filename_component (_idl_file_name   "${_idl_file}" NAME)
    get_filename_component (_idl_file_namewe "${_idl_file}" NAME_WE)
    
    set (_idl_file_copy "${CMAKE_CURRENT_BINARY_DIR}/${_idl_file_name}")
    
    add_custom_command (OUTPUT ${_idl_file_copy}
      COMMAND           ${CMAKE_COMMAND} -E copy_if_different ${_idl_file} ${_idl_file_copy}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT           "Copying ${_idl_file} to ${_idl_file_copy}"
      )
    
    get_source_file_property (_compile_flags "${_idl_file}" COMPILE_FLAGS)
    
    if (NOT _compile_flags)
      set (_compile_flags)
    endif (NOT _compile_flags)
    
    separate_arguments (_compile_flags)
    
    set (${gen_c_files}
      ${CMAKE_CURRENT_BINARY_DIR}/${_idl_file_namewe}_stubs.c
      )
    
    if (_compile_flags MATCHES "-header")
      list (APPEND ${gen_c_files} ${CMAKE_CURRENT_BINARY_DIR}/${_idl_file_namewe}.h)
    endif (_compile_flags MATCHES "-header")
    
    set (${gen_ocaml_files}
      ${CMAKE_CURRENT_BINARY_DIR}/${_idl_file_namewe}.mli
      ${CMAKE_CURRENT_BINARY_DIR}/${_idl_file_namewe}.ml
      )
    
    add_custom_command (OUTPUT ${${gen_c_files}} ${${gen_ocaml_files}}
      COMMAND           ${CAMLIDL_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR} ${_compile_flags} ${_idl_file}
      DEPENDS           ${_idl_file_copy}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      )
    
    add_custom_target (ocaml.${_idl_file_namewe}.ml DEPENDS ${${gen_c_files}} ${${gen_ocaml_files}})

    if (NOT EXISTS ${_idl_file_copy})
      execute_process (
	COMMAND           ${CMAKE_COMMAND} -E copy_if_different ${_idl_file} ${_idl_file_copy}
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	)
      execute_process (
	COMMAND           ${CAMLIDL_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR} ${_compile_flags} ${_idl_file}
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	)
    endif (NOT EXISTS ${_idl_file_copy})
    
  endforeach (_idl_file)
endmacro (gen_caml_idl)

#macro (add_ocaml_c_library name)
    
#  add_library (${name}.so SHARED ${ARGN})

#  set_target_properties (${name}.so PROPERTIES
#    PREFIX      "dll"
#    SUFFIX      ".so"
#    LINK_FLAGS  "-flat_namespace -undefined suppress -read_only_relocs suppress"
#    )

#endmacro (add_ocaml_c_library)
