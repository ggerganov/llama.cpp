ARG UBUNTU_VERSION=22.04
# This needs to generally match the container host's environment.
ARG CUDA_VERSION=11.7.1
# Target the CUDA build image
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
# Target the CUDA runtime image
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER} as build

# Unless otherwise specified, we make a fat build.
ARG CUDA_DOCKER_ARCH=all

# Install gcc-10 and g++-10 to work around and issue CUDA 11.7.1 and gcc-11
RUN apt-get update && \
    apt-get install -y build-essential git
    apt-get install -y build-essential git g++-10  gcc-10

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100

WORKDIR /app

COPY . .

# Set nvcc architecture
ENV CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH}
# Enable cuBLAS
ENV LLAMA_CUBLAS=1

RUN make

FROM ${BASE_CUDA_RUN_CONTAINER} as runtime

COPY --from=build /app/server /server

ENTRYPOINT [ "/server" ]
