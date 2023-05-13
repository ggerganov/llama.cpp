FROM ubuntu

ENV NODE_VERSION=18.16.0
ENV NODE_DIR /node
ENV NODE_PATH ${NODE_DIR}/lib/node_modules

ENV LLAMA_CPP_BRANCH=master
ENV GPT_LLAMA_CPP_BRANCH=master

ENV PATH ${NODE_DIR}/bin:${PATH}

# Make sure apt is non-interactive
RUN set -x \
  && echo 'debconf debconf/frontend select Noninteractive' \
   | debconf-set-selections

# Install deps
RUN set -x \
  && apt update --yes \
  && apt install --yes --no-install-recommends \
    ca-certificates \
    curl \
    g++ \
    gcc \
    git \
    make \
    python-is-python3 \
    python3-pip \
    xz-utils


# Install node
RUN set -x \
  && mkdir --parents "${NODE_DIR}" \
  && curl \
    --location \
    --output /tmp/node.tar.gz \
    "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.gz" \
  && tar \
    --strip-components=1 \
    --ungzip \
    --extract \
    --file="/tmp/node.tar.gz" \
    --directory="${NODE_DIR}" \
  && rm -f /tmp/node.tar.gz

# Install LLaMA CPP
RUN set -x \
    && git clone \
      --branch "${LLAMA_CPP_BRANCH}" \
      --depth 1 \
      https://github.com/ggerganov/llama.cpp \
    && cd /llama.cpp \
    && make -j \
    && python -m pip install -r requirements.txt \
    && mkdir -p models

# Install GPT LLaMA CPP
RUN set -x \
    && git clone \
      --branch "${GPT_LLAMA_CPP_BRANCH}" \
      --depth 1 \
      https://github.com/keldenl/gpt-llama.cpp \
    && cd /gpt-llama.cpp \
    && npm install

EXPOSE 443
WORKDIR /gpt-llama.cpp
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["npm start"]
