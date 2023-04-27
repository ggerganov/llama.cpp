set(TARGET server)
add_executable(${TARGET} server.cpp)

# Boost is needed for Crow
find_package(Boost 1.81.0)
include_directories(${Boost_INCLUDE_DIRS})

target_link_libraries(${TARGET} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

