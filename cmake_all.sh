cd llama.cpp
rm -r build
cmake -B build
cd build
cmake --build . --config Release
cd ..