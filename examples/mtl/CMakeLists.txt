if (APPLE)
    #
    # mtl

    find_library(FOUNDATION_LIBRARY         Foundation              REQUIRED)
    find_library(METAL_FRAMEWORK            Metal                   REQUIRED)
    find_library(METALKIT_FRAMEWORK         MetalKit                REQUIRED)
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

    # TODO: temporary until the kernels are ready
    # custom command to build mtl.metal into a library
    # depends on the mtl.metal file
    add_custom_target(mtl.metallib-tmp ALL DEPENDS ${CMAKE_BINARY_DIR}/mtl.metallib)

    add_custom_command(
        OUTPUT  ${CMAKE_BINARY_DIR}/mtl.metallib
        COMMAND xcrun -sdk macosx metal    -c ${CMAKE_CURRENT_SOURCE_DIR}/mtl.metal -o ${CMAKE_BINARY_DIR}/mtl.air
        COMMAND xcrun -sdk macosx metallib            ${CMAKE_BINARY_DIR}/mtl.air   -o ${CMAKE_BINARY_DIR}/mtl.metallib
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/mtl.metal
        COMMENT "Building mtl.metallib"
    )
endif()

