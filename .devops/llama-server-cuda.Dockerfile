ARG UBUNTU_VERSION=20.04

# This needs to generally match the container host's environment.
ARG CUDA_VERSION=11.4.0
# Target the CUDA build image
ARG BASE_CUDA_DEV_CONTAINER=registry.cn-hangzhou.aliyuncs.com/reg_pub/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
# Target the CUDA runtime image
ARG BASE_CUDA_RUN_CONTAINER=registry.cn-hangzhou.aliyuncs.com/reg_pub/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER} AS build
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Africa/Johannesburg 
# CUDA architecture to build for (defaults to all supported archs)
ARG CUDA_DOCKER_ARCH=default
ADD https://cmake.org/files/v3.25/cmake-3.25.2-linux-x86_64.tar.gz /tmp/
WORKDIR /tmp
RUN tar -xf cmake-3.25.2-linux-x86_64.tar.gz  && \
     cp -rd cmake-3.25.2-linux-x86_64/bin /usr/local/ && \
     cp -rd cmake-3.25.2-linux-x86_64/doc /usr/local/  && \
     cp -rd cmake-3.25.2-linux-x86_64/share /usr/local/ && \
     rm -r /tmp/*
RUN  apt-get update && \
    apt-get install -y build-essential git  libcurl4-openssl-dev && \


COPY . .

# Use the default CUDA archs if not specified
RUN if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
        export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON ${CMAKE_ARGS} -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . && \
    cmake --build build --config Release --target llama-server -j$(nproc) && \
    mkdir -p /app/lib && \
    find build -name "*.so" -exec cp {} /app/lib \;

FROM ${BASE_CUDA_RUN_CONTAINER} AS runtime

RUN apt-get update && \
    apt-get install -y libcurl4-openssl-dev libgomp1 curl

COPY --from=build /app/lib/ /
COPY --from=build /app/build/bin/llama-server /llama-server

# Must be set to 0.0.0.0 so it can listen to requests from host machine
ENV LLAMA_ARG_HOST=0.0.0.0

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/llama-server" ]
