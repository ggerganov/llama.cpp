add_library(omni_vlm_v2 OBJECT
            omni-vlm-v2.cpp
            omni-vlm-v2.h
            clip-v2.cpp
            clip-v2.h
            )

target_link_libraries(omni_vlm_v2 PRIVATE ggml_llama llama ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(omni_vlm_v2 PUBLIC .)
target_include_directories(omni_vlm_v2 PUBLIC ../..)
target_include_directories(omni_vlm_v2 PUBLIC ../../common)

target_compile_features(omni_vlm_v2 PRIVATE cxx_std_11)

add_library(omni_vlm_v2_static STATIC $<TARGET_OBJECTS:omni_vlm_v2>)
if (BUILD_SHARED_LIBS)
    set_target_properties(omni_vlm_v2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(omni_vlm_v2 PRIVATE LLAMA_SHARED LLAMA_BUILD)
    add_library(omni_vlm_v2_shared SHARED $<TARGET_OBJECTS:omni_vlm_v2>)
    target_link_libraries(omni_vlm_v2_shared PRIVATE ggml_llama llama ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS omni_vlm_v2_shared LIBRARY)
endif()

if (NOT MSVC)
    target_compile_options(omni_vlm_v2 PRIVATE -Wno-cast-qual) # stb_image.h
endif()

if(TARGET BUILD_INFO)
    add_dependencies(omni_vlm_v2 BUILD_INFO)
endif()

set(TARGET omni-vlm-v2-cli)
add_executable(${TARGET} omni-vlm-v2-cli.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME omni-vlm-v2-cli)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common omni_vlm_v2 ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

#=== for omni-vlm-wrapper
add_library(omni_vlm_v2_wrapper_shared SHARED omni-vlm-v2-wrapper.cpp $<TARGET_OBJECTS:omni_vlm_v2>)
target_link_libraries(omni_vlm_v2_wrapper_shared PRIVATE common ggml_llama llama ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS omni_vlm_v2_wrapper_shared LIBRARY)

# set(TARGET omni-vlm-wrapper-cli)
# add_executable(${TARGET} omni-vlm-wrapper-cli.cpp)
# set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME omni-vlm-wrapper-cli)
# install(TARGETS ${TARGET} RUNTIME)
# target_link_libraries(${TARGET} PRIVATE omni_vlm_v2_wrapper_shared ${CMAKE_THREAD_LIBS_INIT})
# target_compile_features(${TARGET} PRIVATE cxx_std_11)
