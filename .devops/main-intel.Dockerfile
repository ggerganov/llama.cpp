ARG ONEAPI_VERSION=2024.0.1-devel-ubuntu22.04

FROM intel/oneapi-basekit:$ONEAPI_VERSION as build

ARG LLAMA_SYCL_F16=OFF
RUN apt-get update && \
    apt-get install -y git

WORKDIR /app

COPY . .

RUN mkdir build && \
    cd build && \
    if [ "${LLAMA_SYCL_F16}" = "ON" ]; then \
        echo "LLAMA_SYCL_F16 is set" && \
        export OPT_SYCL_F16="-DLLAMA_SYCL_F16=ON"; \
    fi && \
    cmake .. -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx ${OPT_SYCL_F16} && \
    cmake --build . --config Release --target main

FROM intel/oneapi-basekit:$ONEAPI_VERSION as runtime

COPY --from=build /app/build/bin/main /main

ENV LC_ALL=C.utf8

ENTRYPOINT [ "/main" ]
