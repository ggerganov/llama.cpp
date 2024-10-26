add_library(llava OBJECT
            llava.cpp
            llava.h
            clip.cpp
            clip.h
            )

target_link_libraries(llava PRIVATE ggml jarvis ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(llava PUBLIC .)
target_include_directories(llava PUBLIC ../..)
target_include_directories(llava PUBLIC ../../common)

target_compile_features(llava PRIVATE cxx_std_11)

add_library(llava_static STATIC $<TARGET_OBJECTS:llava>)
if (BUILD_SHARED_LIBS)
    set_target_properties(llava PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llava PRIVATE JARVIS_SHARED JARVIS_BUILD)
    add_library(llava_shared SHARED $<TARGET_OBJECTS:llava>)
    target_link_libraries(llava_shared PRIVATE ggml jarvis ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS llava_shared LIBRARY)
endif()

if (NOT MSVC)
    target_compile_options(llava PRIVATE -Wno-cast-qual) # stb_image.h
endif()

if(TARGET BUILD_INFO)
    add_dependencies(llava BUILD_INFO)
endif()

set(TARGET jarvis-llava-cli)
add_executable(${TARGET} llava-cli.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME jarvis-llava-cli)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set(TARGET jarvis-minicpmv-cli)
add_executable(${TARGET} minicpmv-cli.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME jarvis-minicpmv-cli)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
