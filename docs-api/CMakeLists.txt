find_package(Doxygen)

if (DOXYGEN_FOUND)
    message("Doxygen found. Generating documentation...")
    add_custom_target(doc ALL
        COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
else ()
    message(FATAL_ERROR "Doxygen is needed to build the documentation. Please install it.")
endif ()
