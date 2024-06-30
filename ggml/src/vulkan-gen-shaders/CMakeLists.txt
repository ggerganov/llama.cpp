
set(TARGET vulkan-gen-shaders)
add_executable(${TARGET} vulkan-gen-shaders.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_11)
