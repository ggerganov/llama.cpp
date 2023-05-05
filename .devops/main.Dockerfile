ARG UBUNTU_VERSION=22.04

FROM ubuntu:$UBUNTU_VERSION as build

RUN apt-get update && \
    apt-get install -y build-essential

WORKDIR /app

COPY . .

RUN LLAMA_AVX2_ONLY=1 make

FROM ubuntu:$UBUNTU_VERSION as runtime

COPY --from=build /app/main /main

ENTRYPOINT [ "/main" ]
