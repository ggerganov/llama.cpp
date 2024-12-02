
find_package(Doxygen REQUIRED)

# Parameters to be replaced in Doxifile.in as "@VAR@"
file(GLOB_RECURSE DOXYGEN_INPUT_FILES_RAW
    ${PROJECT_SOURCE_DIR}/src/include/kompute/*.hpp)
# Need to do a string replace as files have to be 
# space separated and with double quotes
string(REPLACE ";" "\" \"" 
    DOXYGEN_INPUT_FILES "${DOXYGEN_INPUT_FILES_RAW}")
set(DOXYGEN_OUTPUT_DIR 
    ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
# DOXIGEN_DOT_PATH is also set automatically

# Parameters only used inside cmake script
set(DOXYGEN_INDEX_FILE
    ${DOXYGEN_OUTPUT_DIR}/html/index.html)
set(DOXYGEN_XML_PATH
    ${DOXYGEN_OUTPUT_DIR}/xml/)
set(DOXYFILE_IN
    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
set(SPHINX_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD
    ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
set(CODECOV_DOCS_DIR
    ${SPHINX_BUILD}/codecov/)
set(CODECOV_DOCS_INDEX_FILE
    ${CODECOV_DOCS_DIR}/index.html)

# Perform replacement with cmake vars inside Doxifine.in
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
add_custom_command(
    OUTPUT ${DOXYGEN_INDEX_FILE}
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
    MAIN_DEPENDENCY ${DOXYFILE_IN} ${DOXYFILE_OUT}
    COMMENT "Generating docs"
)

if(KOMPUTE_OPT_CODE_COVERAGE)
    add_custom_target(gendoxygen ALL 
        DEPENDS ${DOXYGEN_INDEX_FILE} ${CODECOV_DOCS_INDEX_FILE})
else()
    add_custom_target(gendoxygen ALL 
        DEPENDS ${DOXYGEN_INDEX_FILE})
endif()

#####################################################
########### CODECOV DOCS ###############
#####################################################

add_custom_command(
    OUTPUT ${CODECOV_DOCS_INDEX_FILE}
    COMMAND ${CMAKE_COMMAND}
        -E copy_directory
        ${CODECOV_DIR_HTML}
        ${CODECOV_DOCS_DIR}
    COMMAND ${CMAKE_COMMAND}
        -E copy
        ${PROJECT_SOURCE_DIR}/docs/assets/gcov.css
        ${CODECOV_DOCS_DIR}/gcov.css
    DEPENDS codecov_genhtml)

#####################################################
########### Sphinx ###############
#####################################################

find_package(Sphinx REQUIRED)

add_custom_target(gensphinx ALL
    COMMAND
    ${SPHINX_EXECUTABLE} -b html 
        ${SPHINX_SOURCE} ${SPHINX_BUILD}
        -Dbreathe_projects.Kompute=${DOXYGEN_XML_PATH}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS gendoxygen
    COMMENT "Generating documentation with Sphinx")

# Copy the output doxygen html files
add_custom_target(gendocsall ALL
    COMMAND ${CMAKE_COMMAND}
        -E copy_directory
        ${DOXYGEN_OUTPUT_DIR}/html/
        ${SPHINX_BUILD}/doxygen/
# Copy the CNAME file from the repo
    COMMAND ${CMAKE_COMMAND}
        -E copy
        ${PROJECT_SOURCE_DIR}/CNAME
        ${SPHINX_BUILD}/CNAME
# Create assets directory
    COMMAND ${CMAKE_COMMAND}
        -E make_directory
        ${SPHINX_BUILD}/_static/assets/
# Copy the custom asset folder
    COMMAND ${CMAKE_COMMAND}
        -E copy_directory
        ${PROJECT_SOURCE_DIR}/docs/assets/
        ${SPHINX_BUILD}/_static/assets/
    DEPENDS gensphinx)

