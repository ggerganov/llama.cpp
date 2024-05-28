add_library(minicpmv OBJECT
            minicpmv.cpp
            minicpmv.h
            clip.cpp
            clip.h
            )

target_link_libraries(minicpmv PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(minicpmv PUBLIC .)
target_include_directories(minicpmv PUBLIC ../..)
target_include_directories(minicpmv PUBLIC ../../common)

target_compile_features(minicpmv PRIVATE cxx_std_11)

add_library(minicpmv_static STATIC $<TARGET_OBJECTS:minicpmv>)
if (BUILD_SHARED_LIBS)
    set_target_properties(minicpmv PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(minicpmv PRIVATE LLAMA_SHARED LLAMA_BUILD)
    add_library(minicpmv_shared SHARED $<TARGET_OBJECTS:minicpmv>)
    target_link_libraries(minicpmv_shared PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS minicpmv_shared LIBRARY)
endif()

if (NOT MSVC)
    target_compile_options(minicpmv PRIVATE -Wno-cast-qual) # stb_image.h
endif()

if(TARGET BUILD_INFO)
    add_dependencies(minicpmv BUILD_INFO)
endif()

set(TARGET minicpmv-cli)
add_executable(minicpmv-cli minicpmv-cli.cpp)
install(TARGETS minicpmv-cli RUNTIME)
target_link_libraries(minicpmv-cli PRIVATE common minicpmv_wrapper minicpmv ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(minicpmv PRIVATE cxx_std_11)

add_library(minicpmv_wrapper OBJECT
        minicpmv_wrapper.cpp
)
target_link_libraries(minicpmv_wrapper PRIVATE minicpmv ${CMAKE_THREAD_LIBS_INIT})