set(TARGET mtl-export)
add_executable(${TARGET} mtl-export.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()

if (APPLE)
    #
    # mtl

    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(METAL_FRAMEWORK    Metal      REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit   REQUIRED)
    find_library(METALPERFORMANCE_FRAMEWORK MetalPerformanceShaders REQUIRED)

    set(TEST_TARGET mtl)
    add_executable(${TEST_TARGET} mtl.cpp mtl.h mtl.m)
    target_link_libraries(${TEST_TARGET} PRIVATE
        ggml
        ${FOUNDATION_LIBRARY}
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
        ${METALPERFORMANCE_FRAMEWORK}
    )
endif()

