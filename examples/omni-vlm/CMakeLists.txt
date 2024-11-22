add_library(omni_vlm OBJECT
            omni-vlm.cpp
            omni-vlm.h
            clip.cpp
            clip.h
            )

target_link_libraries(omni_vlm PRIVATE ggml_llama llama ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(omni_vlm PUBLIC .)
target_include_directories(omni_vlm PUBLIC ../..)
target_include_directories(omni_vlm PUBLIC ../../common)

target_compile_features(omni_vlm PRIVATE cxx_std_11)

add_library(omni_vlm_static STATIC $<TARGET_OBJECTS:omni_vlm>)
if (BUILD_SHARED_LIBS)
    set_target_properties(omni_vlm PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(omni_vlm PRIVATE LLAMA_SHARED LLAMA_BUILD)
    add_library(omni_vlm_shared SHARED $<TARGET_OBJECTS:omni_vlm>)
    target_link_libraries(omni_vlm_shared PRIVATE ggml_llama llama ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS omni_vlm_shared LIBRARY)
endif()

if (NOT MSVC)
    target_compile_options(omni_vlm PRIVATE -Wno-cast-qual) # stb_image.h
endif()

if(TARGET BUILD_INFO)
    add_dependencies(omni_vlm BUILD_INFO)
endif()

set(TARGET omni-vlm-cli)
add_executable(${TARGET} omni-vlm-cli.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME omni-vlm-cli)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common omni_vlm ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

#=== for omni-vlm-wrapper
add_library(omni_vlm_wrapper_shared SHARED omni-vlm-wrapper.cpp $<TARGET_OBJECTS:omni_vlm>)
target_link_libraries(omni_vlm_wrapper_shared PRIVATE common ggml_llama llama ${CMAKE_THREAD_LIBS_INIT})
# For Nexa SDK library installation
set_target_properties(omni_vlm_wrapper_shared PROPERTIES
                     PUBLIC_HEADER "omni-vlm-wrapper.h"
                     POSITION_INDEPENDENT_CODE ON
                     OUTPUT_NAME "omni_vlm_wrapper_shared")
install(TARGETS omni_vlm_wrapper_shared 
        LIBRARY
        PUBLIC_HEADER DESTINATION include)

# set(TARGET omni-vlm-wrapper-cli)
# add_executable(${TARGET} omni-vlm-wrapper-cli.cpp)
# set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME omni-vlm-wrapper-cli)
# install(TARGETS ${TARGET} RUNTIME)
# target_link_libraries(${TARGET} PRIVATE omni_vlm_wrapper_shared ${CMAKE_THREAD_LIBS_INIT})
# target_compile_features(${TARGET} PRIVATE cxx_std_11)
