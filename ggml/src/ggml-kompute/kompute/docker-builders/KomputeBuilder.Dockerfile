
ARG VULKAN_SDK_VERSION
ARG SWIFTSHADER_VERSION

# Using FROM reference as COPY command doesn't support args
FROM axsauze/vulkan-sdk:${VULKAN_SDK_VERSION} as vulkansdk-image
FROM axsauze/swiftshader:${SWIFTSHADER_VERSION} as swiftshader-image

# Ubuntu as actual image base
FROM ubuntu:22.04

# Repeating args for context in this image
ARG VULKAN_SDK_VERSION
ARG SWIFTSHADER_VERSION

ENV VULKAN_SDK="/VulkanSDK/${VULKAN_SDK_VERSION}/x86_64"
ENV PATH="${VULKAN_SDK}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${VULKAN_SDK}/lib"
ENV VK_LAYER_PATH="${VULKAN_SDK}/etc/explicit_layer.d"

# Base packages from default ppa
RUN apt-get update -y
RUN apt-get install -y wget
RUN apt-get install -y gnupg
RUN apt-get install -y ca-certificates
RUN apt-get install -y software-properties-common

# Repository for latest git (needed for gh actions)
RUN add-apt-repository -y ppa:git-core/ppa

# Refresh repositories
RUN apt-get update -y --fix-missing

RUN apt install -y git
RUN apt-get install -y gcc
RUN apt-get install -y cmake
RUN apt-get install -y g++

# Swiftshader dependencies
RUN apt-get install -y libx11-dev zlib1g-dev
RUN apt-get install -y libxext-dev

# Vulkan wayland client dependency
RUN apt-get install -y libwayland-client0

# GLSLANG tools for tests
RUN apt-get install -y glslang-tools

# Setup Python
RUN apt-get install -y python3-pip

# Setup Node for nektos/act (local Github Actions) tests
RUN apt-get install -y nodejs

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

COPY --from=vulkansdk-image ${VULKAN_SDK} ${VULKAN_SDK}
COPY --from=swiftshader-image /swiftshader/ /swiftshader/

RUN mkdir builder
WORKDIR /builder


