set(TARGET TCPportquiz)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(${TARGET} TCPportquiz.cpp)

set(TARGET Carmichael)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(${TARGET} Carmichael.cpp)

set(TARGET Carmichael2)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(${TARGET} Carmichael2.cpp)

install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET} PRIVATE ws2_32)
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_11)
