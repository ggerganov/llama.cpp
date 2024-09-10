add_library(internvl OBJECT
            internvl.cpp
            internvl.h
            clip.cpp
            clip.h
            )

target_link_libraries(internvl PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(internvl PUBLIC .)
target_include_directories(internvl PUBLIC ../..)
target_include_directories(internvl PUBLIC ../../common)

target_compile_features(internvl PRIVATE cxx_std_11)

add_library(internvl_static STATIC $<TARGET_OBJECTS:internvl>)
if (BUILD_SHARED_LIBS)
    set_target_properties(internvl PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(internvl PRIVATE INTERNVL_SHARED INTERNVL_BUILD)
    add_library(internvl_shared SHARED $<TARGET_OBJECTS:internvl>)
    target_link_libraries(internvl_shared PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS internvl_shared LIBRARY)
endif()

if (NOT MSVC)
    target_compile_options(internvl PRIVATE -Wno-cast-qual) # stb_image.h
endif()

if(TARGET BUILD_INFO)
    add_dependencies(internvl BUILD_INFO)
endif()

set(TARGET llama-internvl-cli)
add_executable(${TARGET} internvl-cli.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-internvl-cli)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common internvl ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
