ARG ONEAPI_VERSION=2024.0.1-devel-ubuntu22.04
ARG UBUNTU_VERSION=22.04

FROM intel/hpckit:$ONEAPI_VERSION as build

RUN apt-get update && \
    apt-get install -y git

WORKDIR /app

COPY . .

# for some reasons, "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=Intel10_64lp -DLLAMA_NATIVE=ON" give worse performance
RUN mkdir build && \
    cd build && \
    cmake .. -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx && \
    cmake --build . --config Release --target main server

FROM ubuntu:$UBUNTU_VERSION as runtime

COPY --from=build /app/build/bin/server /server

ENV LC_ALL=C.utf8

ENTRYPOINT [ "/server" ]
