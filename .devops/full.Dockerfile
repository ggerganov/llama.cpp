ARG UBUNTU_VERSION=22.04

FROM ubuntu:$UBUNTU_VERSION AS build

RUN apt-get update && \
    apt-get install -y build-essential git cmake libcurl4-openssl-dev

WORKDIR /app

COPY . .

RUN cmake -S . -B build -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j $(nproc) && \
    mkdir -p /app/lib && \
    find build -name "*.so" -exec cp {} /app/lib/ \;

FROM ubuntu:$UBUNTU_VERSION as runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential python3 python3-pip git libcurl4-openssl-dev libgomp1

COPY requirements.txt   /app/requirements.txt
COPY requirements       /app/requirements
COPY .devops/tools.sh   /app/tools.sh

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt

COPY --from=build /app/build/bin/ /app/
COPY --from=build /app/lib/ /app/
COPY --from=build /app/convert_hf_to_gguf.py /app/
COPY --from=build /app/gguf-py /app/gguf-py

ENV LC_ALL=C.utf8

ENTRYPOINT ["/app/tools.sh"]
