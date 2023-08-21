# common

set(TARGET common)

add_library(${TARGET} OBJECT
    common.h
    common.cpp
    console.h
    console.cpp
    grammar-parser.h
    grammar-parser.cpp
    )

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PUBLIC .)
target_compile_features(${TARGET} PUBLIC cxx_std_11)
target_link_libraries(${TARGET} PRIVATE llama)
