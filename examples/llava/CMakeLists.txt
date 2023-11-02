set(TARGET llava)

add_library(${TARGET} STATIC llava.cpp llava.h clip.cpp clip.h)
target_link_libraries(${TARGET} PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(${TARGET} PUBLIC .)
target_include_directories(${TARGET} PUBLIC ../..)
target_include_directories(${TARGET} PUBLIC ../../common)

target_compile_features(${TARGET} PRIVATE cxx_std_11)
if (NOT MSVC)
    target_compile_options(${TARGET} PRIVATE -Wno-cast-qual) # stb_image.h
    endif()
if(TARGET BUILD_INFO)
    add_dependencies(${TARGET} BUILD_INFO)
endif()

set(TARGET llava-cli)
add_executable(${TARGET} llava-cli.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama llava ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
