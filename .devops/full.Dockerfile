ARG UBUNTU_VERSION=22.04

FROM ubuntu:$UBUNTU_VERSION as build

RUN apt-get update && \
    apt-get install -y build-essential python3 python3-pip

RUN pip install --upgrade pip setuptools wheel \
    && pip install numpy requests sentencepiece tqdm \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

COPY . .

RUN make

ENTRYPOINT ["/app/.devops/tools.sh"]
