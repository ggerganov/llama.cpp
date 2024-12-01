ARG UBUNTU_VERSION=22.04

FROM ubuntu:$UBUNTU_VERSION AS build

RUN apt-get update && \
    apt-get install -y build-essential git cmake libcurl4-openssl-dev

WORKDIR /app

COPY . .


RUN \
    # Build multiple versions of the CPU backend
    scripts/build-cpu.sh avx         -DGGML_AVX=ON -DGGML_AVX2=OFF && \
    scripts/build-cpu.sh avx2        -DGGML_AVX=ON -DGGML_AVX2=ON && \
    scripts/build-cpu.sh avx512      -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX512=ON && \
    scripts/build-cpu.sh amx         -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX512=ON -DGGML_AVX_VNNI=ON -DGGML_AVX512_VNNI=ON -DGGML_AMX_TILE=ON -DGGML_AMX_INT8=ON && \
    # Build llama-server
    cmake -S . -B build -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --target llama-server -j $(nproc) && \
    # Copy the built libraries to /app/lib
    mkdir -p /app/lib && \
    mv libggml-cpu* /app/lib/ && \
    find build -name "*.so" -exec cp {} /app/lib/ \;

FROM ubuntu:$UBUNTU_VERSION AS runtime

RUN apt-get update && \
    apt-get install -y libcurl4-openssl-dev libgomp1 curl

COPY --from=build /app/build/bin/llama-server /llama-server
COPY --from=build /app/lib/ /

ENV LC_ALL=C.utf8
# Must be set to 0.0.0.0 so it can listen to requests from host machine
ENV LLAMA_ARG_HOST=0.0.0.0

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/llama-server" ]
