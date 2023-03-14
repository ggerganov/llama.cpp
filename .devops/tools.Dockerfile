ARG UBUNTU_VERSION=22.04

FROM ubuntu:$UBUNTU_VERSION as build

RUN apt-get update && \
    apt-get install -y build-essential python3 python3-pip

RUN pip install --upgrade pip setuptools wheel \
    && pip install torch torchvision torchaudio sentencepiece numpy

WORKDIR /app

COPY . .

RUN make

ENTRYPOINT ["/app/.devops/tools.sh"]