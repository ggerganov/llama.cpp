rm -rf build
mkdir -p build
cmake -S . -B build -GXcode -DCMAKE_TOOLCHAIN_FILE=cmake/ios.toolchain.cmake \
    -DPLATFORM=MAC_UNIVERSAL \
    -DBUILD_SHARED_LIBS=YES
cmake --build build --config Release
