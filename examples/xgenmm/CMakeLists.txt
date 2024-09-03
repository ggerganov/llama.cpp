add_library(xgenmm OBJECT
            xgenmm.cpp
            xgenmm.h
            clip.cpp
            clip.h
            )

target_link_libraries(xgenmm PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(xgenmm PUBLIC .)
target_include_directories(xgenmm PUBLIC ../..)
target_include_directories(xgenmm PUBLIC ../../common)

target_compile_features(xgenmm PRIVATE cxx_std_11)

add_library(xgenmm_static STATIC $<TARGET_OBJECTS:xgenmm>)
if (BUILD_SHARED_LIBS)
    set_target_properties(xgenmm PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(xgenmm PRIVATE LLAMA_SHARED LLAMA_BUILD)
    add_library(xgenmm_shared SHARED $<TARGET_OBJECTS:xgenmm>)
    target_link_libraries(xgenmm_shared PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS xgenmm_shared LIBRARY)
endif()

if (NOT MSVC)
    target_compile_options(xgenmm PRIVATE -Wno-cast-qual) # stb_image.h
endif()

if(TARGET BUILD_INFO)
    add_dependencies(xgenmm BUILD_INFO)
endif()


set(TARGET test_anyres_img)
add_executable(test_anyres_img test_anyres_img.cpp)
install(TARGETS test_anyres_img RUNTIME)
target_link_libraries(test_anyres_img PRIVATE common xgenmm ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(xgenmm PRIVATE cxx_std_11)


set(TARGET test_anyres_handle_patches)
add_executable(test_anyres_handle_patches test_anyres_handle_patches.cpp)
install(TARGETS test_anyres_handle_patches RUNTIME)
target_link_libraries(test_anyres_handle_patches PRIVATE common xgenmm ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(xgenmm PRIVATE cxx_std_11)

set(TARGET test_patch_ops)
add_executable(test_patch_ops test_patch_ops.cpp)
install(TARGETS test_patch_ops RUNTIME)
target_link_libraries(test_patch_ops PRIVATE common xgenmm ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(xgenmm PRIVATE cxx_std_11)


# not implemented yet
# set(TARGET xgenmm-cli)
# add_executable(xgenmm-cli xgenmm-cli.cpp)
# install(TARGETS xgenmm-cli RUNTIME)
# target_link_libraries(xgenmm-cli PRIVATE common xgenmm_io xgenmm ${CMAKE_THREAD_LIBS_INIT})
# target_compile_features(xgenmm PRIVATE cxx_std_11)

# add_library(xgenmm_io OBJECT
#         xgenmm_io.cpp
# )
# target_link_libraries(xgenmm_io PRIVATE xgenmm ${CMAKE_THREAD_LIBS_INIT})            