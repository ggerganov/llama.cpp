# TODO: should not use this
if (WIN32)
    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

#
# libraries
#

# jarvis

add_library(jarvis
            ../include/jarvis.h
            jarvis.cpp
            jarvis-vocab.cpp
            jarvis-grammar.cpp
            jarvis-sampling.cpp
            unicode.h
            unicode.cpp
            unicode-data.cpp
            )

target_include_directories(jarvis PUBLIC . ../include)
target_compile_features   (jarvis PUBLIC cxx_std_11) # don't bump

target_link_libraries(jarvis PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(jarvis PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(jarvis PRIVATE JARVIS_SHARED JARVIS_BUILD)
endif()
