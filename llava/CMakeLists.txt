set(TARGET llava)

if (BUILD_SHARED_LIBS)
    add_library(${TARGET} SHARED llava.cpp llava.h clip.cpp clip.h)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    target_link_libraries(${TARGET} PUBLIC Threads::Threads common llama ggml ${LLAMA_EXTRA_LIBS})
    install(TARGETS ${TARGET} LIBRARY)
else()
    add_library(${TARGET} llava.cpp llava.h clip.cpp clip.h)
    target_link_libraries(${TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
endif()

target_include_directories(${TARGET} PUBLIC .)
target_include_directories(${TARGET} PUBLIC ..)
target_include_directories(${TARGET} PUBLIC ../common)

target_compile_features(${TARGET} PRIVATE cxx_std_11)
if (NOT MSVC)
    target_compile_options(${TARGET} PRIVATE -Wno-cast-qual) # stb_image.h
    endif()
if(TARGET BUILD_INFO)
    add_dependencies(${TARGET} BUILD_INFO)
endif()

