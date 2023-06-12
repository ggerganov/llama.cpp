ARG UBUNTU_VERSION=22.04

FROM ubuntu:$UBUNTU_VERSION as build

RUN apt-get update && \
    apt-get install -y build-essential git

WORKDIR /app

COPY . .

RUN make

FROM ubuntu:$UBUNTU_VERSION as runtime

COPY --from=build /app/main /main

ENV LC_ALL=C.utf8

ENTRYPOINT [ "/main" ]
