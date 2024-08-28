ARG UBUNTU_VERSION=22.04

FROM ubuntu:$UBUNTU_VERSION AS build

RUN apt-get update && \
    apt-get install -y build-essential python3 python3-pip git libcurl4-openssl-dev libgomp1

COPY requirements.txt   requirements.txt
COPY requirements       requirements

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

WORKDIR /app

COPY . .

ENV LLAMA_CURL=1

# Only build targets used by tools.sh
RUN make -j$(nproc) llama-quantize llama-cli llama-server

ENV LC_ALL=C.utf8

ENTRYPOINT ["/app/.devops/tools.sh"]
