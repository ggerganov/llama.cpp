
ARG VULKAN_SDK_VERSION

# Using FROM reference as COPY command doesn't support args
FROM axsauze/vulkan-sdk:$VULKAN_SDK_VERSION as vulkansdk-image

# Ubuntu as actual image base
FROM ubuntu:22.04 as swiftshader-builder

# Repeating ARG for context in this image
ARG VULKAN_SDK_VERSION

# Base packages from default ppa
RUN apt-get update -y
RUN apt-get install -y wget
RUN apt-get install -y gnupg
RUN apt-get install -y ca-certificates
RUN apt-get install -y software-properties-common

# Build dependencies
RUN apt install -y git
RUN apt-get install -y cmake g++

# Setup Vulkan
ENV VULKAN_SDK="/VulkanSDK/${VULKAN_SDK_VERSION}/x86_64"
ENV PATH="${VULKAN_SDK}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${VULKAN_SDK}/lib"
ENV VK_LAYER_PATH="${VULKAN_SDK}/etc/explicit_layer.d"

COPY --from=vulkansdk-image ${VULKAN_SDK} ${VULKAN_SDK}

# Dependencies for swiftshader
# RUN apt-get install -y g++-8 gcc-8
RUN apt-get install -y gcc
RUN apt-get install -y libx11-dev zlib1g-dev
RUN apt-get install -y libxext-dev

# Run swiftshader via env VK_ICD_FILENAMES=/swiftshader/vk_swiftshader_icd.json
RUN git clone https://github.com/google/swiftshader swiftshader-build
RUN cmake swiftshader-build/. -Bswiftshader-build/build/
RUN cmake --build swiftshader-build/build/. --parallel 8
RUN cp -r swiftshader-build/build/Linux/ swiftshader/


# Store build in slim down image
FROM ubuntu:22.04

COPY --from=swiftshader-builder /swiftshader/ /swiftshader/

