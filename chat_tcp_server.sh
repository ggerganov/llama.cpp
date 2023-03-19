#!/usr/bin/env bash

PORT=${PORT:-8080}
MODEL=${MODEL:-models/7B/ggml-model-q4_0.bin}

./main -l ${PORT} -m $MODEL
