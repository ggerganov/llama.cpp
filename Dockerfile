FROM debian:12

RUN apt-get update -y && apt-get install -y build-essential g++ git wget
WORKDIR /app/src
COPY . .
RUN make -j
WORKDIR /models
RUN wget https://huggingface.co/TheBloke/LLaMa-7B-GGML/resolve/main/llama-7b.ggmlv3.q4_0.bin
ENTRYPOINT [ "/app/src/start.sh" ]
