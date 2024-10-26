# jarvis.cpp/example/main-cmake-pkg

This program builds [jarvis-cli](../main) using a relocatable CMake package. It serves as an example of using the `find_package()` CMake command to conveniently include [jarvis.cpp](https://github.com/ggerganov/jarvis.cpp) in projects which live outside of the source tree.

## Building

Because this example is "outside of the source tree", it is important to first build/install jarvis.cpp using CMake. An example is provided here, but please see the [jarvis.cpp build instructions](../..) for more detailed build instructions.

### Considerations

When hardware acceleration libraries are used (e.g. CUDA, Metal, etc.), CMake must be able to locate the associated CMake package.

### Build jarvis.cpp and install to C:\JarvisCPP directory

```cmd
git clone https://github.com/ggerganov/jarvis.cpp
cd jarvis.cpp
cmake -B build -DBUILD_SHARED_LIBS=OFF -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
cmake --install build --prefix C:/JarvisCPP
```

### Build jarvis-cli-cmake-pkg


```cmd
cd ..\examples\main-cmake-pkg
cmake -B build -DBUILD_SHARED_LIBS=OFF -DCMAKE_PREFIX_PATH="C:/JarvisCPP/lib/cmake/Jarvis" -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
cmake --install build --prefix C:/MyJarvisApp
```
