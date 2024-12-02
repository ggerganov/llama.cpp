FROM ubuntu:20.04 as sdk-builder

ARG VULKAN_SDK_VERSION

# First install vulkan 
RUN apt-get update
RUN apt-get install -y curl unzip tar wget
RUN wget -O VulkanSDK.tar.gz https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz?u=true && \
    mkdir VulkanSDK && \
    cd VulkanSDK && \
    tar xvf /VulkanSDK.tar.gz

RUN cd VulkanSDK/${VULKAN_SDK_VERSION}
ENV VULKAN_SDK="/VulkanSDK/${VULKAN_SDK_VERSION}/x86_64"
ENV PATH="${VULKAN_SDK}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${VULKAN_SDK}/lib"
ENV VK_LAYER_PATH="${VULKAN_SDK}/etc/explicit_layer.d"

# Configure dependencies required to setup external apt-repos
RUN apt-get install \
        apt-transport-https \
        ca-certificates \
        gnupg \
        software-properties-common \
        wget -y

# Adding kitware repo to upgrade to latest cmake (17+)
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
# Adding security repo to add cmake dependency on libssl 1.0
RUN echo "deb http://security.ubuntu.com/ubuntu bionic-security main" | tee -a /etc/apt/sources.list.d/bionic.list
RUN apt-get update --fix-missing -y
RUN apt-get upgrade -y

# Install build dependencies
RUN apt-get install -y libssl1.0-dev 
RUN apt-get install -y cmake g++
RUN apt-get install -y libglm-dev libxcb-dri3-0 libxcb-present0
RUN apt-get install -y libpciaccess0 libpng-dev libxcb-keysyms1-dev
RUN apt-get install -y libxcb-dri3-dev
RUN apt-get install -y libx11-dev
RUN apt-get install -y libwayland-dev libxrandr-dev
RUN apt-get install -y wget
RUN apt-get install -y libglfw3-dev
RUN apt-get install -y git
RUN apt-get install -y python python3-pip
RUN apt-get install -y wayland-protocols

# Rerun update & upgrade to download libmirclient
RUN apt-get upgrade -y
RUN apt-get update --fix-missing -y
RUN apt-get install -y libmirclient-dev

RUN apt-get install -y zip pkg-config
RUN apt-get install -y ninja-build
RUN apt-get install -y qt5-default
RUN apt-get install -y qt5-qmake

# Python deps
RUN pip install jsonschema

RUN /VulkanSDK/${VULKAN_SDK_VERSION}/vulkansdk -j $(nproc)

# Cleanup to reduce image size
RUN rm -rf /VulkanSDK/${VULKAN_SDK_VERSION}/source

RUN mkdir /workspace
WORKDIR /workspace

# Store build in slim down image (reduce from 16GB to 1GB)
FROM ubuntu:20.04

ARG VULKAN_SDK_VERSION

ENV VULKAN_SDK="/VulkanSDK/${VULKAN_SDK_VERSION}/x86_64"
ENV PATH="${VULKAN_SDK}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${VULKAN_SDK}/lib"
ENV VK_LAYER_PATH="${VULKAN_SDK}/etc/explicit_layer.d"

COPY --from=sdk-builder ${VULKAN_SDK} ${VULKAN_SDK}

