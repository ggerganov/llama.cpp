
pybind11_add_module(kp src/main.cpp)

include_directories(
        ${PROJECT_SOURCE_DIR}/include/)

target_link_libraries(
        kp PRIVATE
        kompute::kompute)
