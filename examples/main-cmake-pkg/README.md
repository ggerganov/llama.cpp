# llama.cpp/example/main-cmake-pkg

This program builds the [main](../main) application using a relocatable CMake package. It serves as an example of using the `find_package()` CMake command to conveniently include [llama.cpp](https://github.com/ggerganov/llama.cpp) in projects which live outside of the source tree.

## Building

Because this example is "outside of the source tree", it is important to first build/install llama.cpp using CMake. An example is provided here, but please see the [llama.cpp build instructions](../..) for more detailed build instructions.

### Considerations

When hardware acceleration libraries are used (e.g. CUDA, Metal, CLBlast, etc.), CMake must be able to locate the associated CMake package. In the example below, when building _main-cmake-pkg_ notice the `CMAKE_PREFIX_PATH` includes the Llama CMake package location _in addition to_ the CLBlast package—which was used when compiling _llama.cpp_.

### Build llama.cpp and install to C:\LlamaCPP directory

In this case, CLBlast was already installed so the CMake package is referenced in `CMAKE_PREFIX_PATH`.

```cmd
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=OFF -DLLAMA_CLBLAST=ON -DCMAKE_PREFIX_PATH=C:/CLBlast/lib/cmake/CLBlast -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
cmake --install build --prefix C:/LlamaCPP
```

### Build main-cmake-pkg


```cmd
cd ..\examples\main-cmake-pkg
cmake -B build -DBUILD_SHARED_LIBS=OFF -DCMAKE_PREFIX_PATH="C:/CLBlast/lib/cmake/CLBlast;C:/LlamaCPP/lib/cmake/Llama" -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
cmake --install build --prefix C:/MyLlamaApp
```
