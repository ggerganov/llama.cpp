FROM debian:12

RUN apt-get update -y && apt-get install -y build-essential g++ git
WORKDIR /app/src
COPY . .
RUN make -j
VOLUME [ "/models" ]
ENTRYPOINT [ "/app/src/start.sh" ]
