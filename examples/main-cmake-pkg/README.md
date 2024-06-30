# llama.cpp/example/main-cmake-pkg

This program builds [llama-cli](../main) using a relocatable CMake package. It serves as an example of using the `find_package()` CMake command to conveniently include [llama.cpp](https://github.com/ggerganov/llama.cpp) in projects which live outside of the source tree.

## Building

Because this example is "outside of the source tree", it is important to first build/install llama.cpp using CMake. An example is provided here, but please see the [llama.cpp build instructions](../..) for more detailed build instructions.

### Considerations

When hardware acceleration libraries are used (e.g. CUDA, Metal, etc.), CMake must be able to locate the associated CMake package.

### Build llama.cpp and install to C:\LlamaCPP directory

```cmd
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=OFF -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
cmake --install build --prefix C:/LlamaCPP
```

### Build llama-cli-cmake-pkg


```cmd
cd ..\examples\main-cmake-pkg
cmake -B build -DBUILD_SHARED_LIBS=OFF -DCMAKE_PREFIX_PATH="C:/LlamaCPP/lib/cmake/Llama" -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
cmake --install build --prefix C:/MyLlamaApp
```
