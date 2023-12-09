# - OCamlDep script
# Compute OCaml dependencies.
#
# Call this script with cmake -D ocamldep=<ocamldep>
#                             -D ocamlfind=<ocamlfind>
#                             -D filename=<filename>
#                             -D output=<output>
#                             -P OcamlDep.cmake
#
# Copyright (c) 2010, Judicaël Bedouet, j dot bedouet at infonie dot fr.
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

get_filename_component (name "${filename}" NAME)

set (dep_file      "${output}/Dependencies/${name}.dep.cmake")
set (temp_dep_file "${dep_file}.tmp")

file (MAKE_DIRECTORY "${output}/Dependencies")

if(ocamlfind)
  set(ocamldep ${ocamlfind} dep)
endif()

execute_process (
  COMMAND         ${ocamldep} -modules ${filename}
  OUTPUT_VARIABLE line
  RESULT_VARIABLE result
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

if (NOT result EQUAL 0)
  message (SEND_ERROR "Can't run ${ocamldep} on ${filename}")
endif (NOT result EQUAL 0)

set (regex "^.+:(.+)$")
if (line MATCHES ${regex})
  string (REGEX REPLACE ${regex} "\\1" deps ${line})
  file (WRITE ${temp_dep_file} "SET (${name}_DEPENDS ${deps})")
else (line MATCHES ${regex})
  file (WRITE ${temp_dep_file} "SET (${name}_DEPENDS)")
endif (line MATCHES ${regex})

execute_process (COMMAND ${CMAKE_COMMAND} -E copy_if_different ${temp_dep_file} ${dep_file})
