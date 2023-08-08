#!/usr/bin/env sh

for ngl in {0..35}
do
    ./main --model models/nvme/llama-7b-ggml-q4_0.bin --seed 1337 --ignore-eos --n-predict 128 --ctx-size 2048 --threads 8 -ngl $ngl -mmq
done
