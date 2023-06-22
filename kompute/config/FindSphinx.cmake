# Look for an executable called sphinx-build
find_program(SPHINX_EXECUTABLE
    NAMES sphinx-build
    DOC "Path to sphinx-build executable")

if(SPHINX_EXECUTABLE STREQUAL "SPHINX_EXECUTABLE-NOTFOUND")
    message(FATAL_ERROR "sphinx-build not found.")
endif()

include(FindPackageHandleStandardArgs)

# Handle standard arguments to find_package like REQUIRED and QUIET
find_package_handle_standard_args(
    Sphinx
    "Failed to find sphinx-build executable"
    SPHINX_EXECUTABLE)
