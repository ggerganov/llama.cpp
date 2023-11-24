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
# ARG CUDA_DOCKER_ARCH=sm_86

RUN apt-get update && \
    apt-get install -y build-essential git wget python3 python3-pip

WORKDIR /app

COPY . .

# Set nvcc architecture
ENV CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH}
# Enable cuBLAS
ENV LLAMA_CUBLAS=1
ENV LLAMA_CUDA_MMV_Y=2
ENV LLAMA_CUDA_DMMV_X=64
ENV LLAMA_CUDA_F16=true

RUN make -j

# Accept the build argument into an environment variable
ARG MODEL_URL
ENV MODEL_URL=${MODEL_URL}

# Use the environment variable to download the model
RUN wget $MODEL_URL -O /model.gguf

WORKDIR /install
RUN pip install --install-option="--prefix=/install" runpod

FROM ${BASE_CUDA_RUN_CONTAINER} as runtime
COPY --from=build /install /usr/local
COPY --from=build /app/server /server
COPY --from=build /model.gguf model.gguf
COPY --from=build /app/models models

# CMD ["/bin/sh", "-c", "/server --model model.gguf --threads $(nproc) -ngl 99 -np $(nproc) -cb"]
# CMD ["/server --host 0.0.0.0 --threads 8 -ngl 999 -np 8 -cb -m model.gguf -c 16384"]
CMD [ "python", "-u", "/handler.py" ]
