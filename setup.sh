mkdir -p build
cd build
source /opt/intel/oneapi/setvars.sh

#cmake .. -DLLAMA_CLBLAST=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx

#cmake .. -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON
cmake .. -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build . --config Release --target main
