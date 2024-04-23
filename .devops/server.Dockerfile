ARG UBUNTU_VERSION=22.04

FROM ubuntu:$UBUNTU_VERSION as build

RUN apt-get update && \
    apt-get install -y build-essential git libcurl4-openssl-dev

WORKDIR /app

COPY . .

ENV LLAMA_CURL=1

RUN make

FROM ubuntu:$UBUNTU_VERSION as runtime

RUN apt-get update && \
    apt-get install -y libcurl4-openssl-dev

COPY --from=build /app/server /server

ENV LC_ALL=C.utf8

ENTRYPOINT [ "/server" ]
