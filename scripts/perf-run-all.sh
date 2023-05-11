#!/bin/bash
#
# Measure the performance (time per token) of the various quantization techniques
#

QUANTIZE=0
if [ "$1" != "" ]; then
    echo "Quantizing"
    QUANTIZE=1
fi

if [ "$QUANTIZE" != "0" ]; then
    #
    # quantize
    #

    # 7B
    time ./bin/quantize ../models/7B/ggml-model-f16.bin ../models/7B/ggml-model-q4_0.bin q4_0 2>&1 | tee ../qnt-7b-q4_0.txt
    time ./bin/quantize ../models/7B/ggml-model-f16.bin ../models/7B/ggml-model-q4_1.bin q4_1 2>&1 | tee ../qnt-7b-q4_1.txt
    time ./bin/quantize ../models/7B/ggml-model-f16.bin ../models/7B/ggml-model-q5_0.bin q5_0 2>&1 | tee ../qnt-7b-q5_0.txt
    time ./bin/quantize ../models/7B/ggml-model-f16.bin ../models/7B/ggml-model-q5_1.bin q5_1 2>&1 | tee ../qnt-7b-q5_1.txt
    time ./bin/quantize ../models/7B/ggml-model-f16.bin ../models/7B/ggml-model-q8_0.bin q8_0 2>&1 | tee ../qnt-7b-q8_0.txt

    # 13B
    time ./bin/quantize ../models/13B/ggml-model-f16.bin ../models/13B/ggml-model-q4_0.bin q4_0 2>&1 | tee ../qnt-13b-q4_0.txt
    time ./bin/quantize ../models/13B/ggml-model-f16.bin ../models/13B/ggml-model-q4_1.bin q4_1 2>&1 | tee ../qnt-13b-q4_1.txt
    time ./bin/quantize ../models/13B/ggml-model-f16.bin ../models/13B/ggml-model-q5_0.bin q5_0 2>&1 | tee ../qnt-13b-q5_0.txt
    time ./bin/quantize ../models/13B/ggml-model-f16.bin ../models/13B/ggml-model-q5_1.bin q5_1 2>&1 | tee ../qnt-13b-q5_1.txt
    time ./bin/quantize ../models/13B/ggml-model-f16.bin ../models/13B/ggml-model-q8_0.bin q8_0 2>&1 | tee ../qnt-13b-q8_0.txt
fi

#
# perf
# run each command twice
#

set -x

# 7B - 4 threads
     ./bin/main -m ../models/7B/ggml-model-f16.bin  -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | grep "I believe"
time ./bin/main -m ../models/7B/ggml-model-f16.bin  -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | tee ../perf-7b-f16.txt  | grep llama_print_timings
     ./bin/main -m ../models/7B/ggml-model-q4_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | grep "I believe"
time ./bin/main -m ../models/7B/ggml-model-q4_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | tee ../perf-7b-q4_0.txt | grep llama_print_timings
     ./bin/main -m ../models/7B/ggml-model-q4_1.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | grep "I believe"
time ./bin/main -m ../models/7B/ggml-model-q4_1.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | tee ../perf-7b-q4_1.txt | grep llama_print_timings
     ./bin/main -m ../models/7B/ggml-model-q5_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | grep "I believe"
time ./bin/main -m ../models/7B/ggml-model-q5_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | tee ../perf-7b-q5_0.txt | grep llama_print_timings
     ./bin/main -m ../models/7B/ggml-model-q5_1.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | grep "I believe"
time ./bin/main -m ../models/7B/ggml-model-q5_1.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | tee ../perf-7b-q5_1.txt | grep llama_print_timings
     ./bin/main -m ../models/7B/ggml-model-q8_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | grep "I believe"
time ./bin/main -m ../models/7B/ggml-model-q8_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | tee ../perf-7b-q8_0.txt | grep llama_print_timings

# 7B - 8 threads
     ./bin/main -m ../models/7B/ggml-model-f16.bin  -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | grep "I believe"
time ./bin/main -m ../models/7B/ggml-model-f16.bin  -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | tee ../perf-7b-f16.txt  | grep llama_print_timings
     ./bin/main -m ../models/7B/ggml-model-q4_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | grep "I believe"
time ./bin/main -m ../models/7B/ggml-model-q4_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | tee ../perf-7b-q4_0.txt | grep llama_print_timings
     ./bin/main -m ../models/7B/ggml-model-q4_1.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | grep "I believe"
time ./bin/main -m ../models/7B/ggml-model-q4_1.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | tee ../perf-7b-q4_1.txt | grep llama_print_timings
     ./bin/main -m ../models/7B/ggml-model-q5_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | grep "I believe"
time ./bin/main -m ../models/7B/ggml-model-q5_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | tee ../perf-7b-q5_0.txt | grep llama_print_timings
     ./bin/main -m ../models/7B/ggml-model-q5_1.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | grep "I believe"
time ./bin/main -m ../models/7B/ggml-model-q5_1.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | tee ../perf-7b-q5_1.txt | grep llama_print_timings
     ./bin/main -m ../models/7B/ggml-model-q8_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | grep "I believe"
time ./bin/main -m ../models/7B/ggml-model-q8_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | tee ../perf-7b-q8_0.txt | grep llama_print_timings

# 13B - 4 threads
     ./bin/main -m ../models/13B/ggml-model-f16.bin  -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | grep "I believe"
time ./bin/main -m ../models/13B/ggml-model-f16.bin  -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | tee ../perf-13b-f16.txt  | grep llama_print_timings
     ./bin/main -m ../models/13B/ggml-model-q4_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | grep "I believe"
time ./bin/main -m ../models/13B/ggml-model-q4_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | tee ../perf-13b-q4_0.txt | grep llama_print_timings
     ./bin/main -m ../models/13B/ggml-model-q4_1.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | grep "I believe"
time ./bin/main -m ../models/13B/ggml-model-q4_1.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | tee ../perf-13b-q4_1.txt | grep llama_print_timings
     ./bin/main -m ../models/13B/ggml-model-q5_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | grep "I believe"
time ./bin/main -m ../models/13B/ggml-model-q5_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | tee ../perf-13b-q5_0.txt | grep llama_print_timings
     ./bin/main -m ../models/13B/ggml-model-q5_1.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | grep "I believe"
time ./bin/main -m ../models/13B/ggml-model-q5_1.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | tee ../perf-13b-q5_1.txt | grep llama_print_timings
     ./bin/main -m ../models/13B/ggml-model-q8_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | grep "I believe"
time ./bin/main -m ../models/13B/ggml-model-q8_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 4 2>&1 | tee ../perf-13b-q8_0.txt | grep llama_print_timings

# 13B - 8 threads
     ./bin/main -m ../models/13B/ggml-model-f16.bin  -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | grep "I believe"
time ./bin/main -m ../models/13B/ggml-model-f16.bin  -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | tee ../perf-13b-f16.txt  | grep llama_print_timings
     ./bin/main -m ../models/13B/ggml-model-q4_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | grep "I believe"
time ./bin/main -m ../models/13B/ggml-model-q4_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | tee ../perf-13b-q4_0.txt | grep llama_print_timings
     ./bin/main -m ../models/13B/ggml-model-q4_1.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | grep "I believe"
time ./bin/main -m ../models/13B/ggml-model-q4_1.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | tee ../perf-13b-q4_1.txt | grep llama_print_timings
     ./bin/main -m ../models/13B/ggml-model-q5_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | grep "I believe"
time ./bin/main -m ../models/13B/ggml-model-q5_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | tee ../perf-13b-q5_0.txt | grep llama_print_timings
     ./bin/main -m ../models/13B/ggml-model-q5_1.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | grep "I believe"
time ./bin/main -m ../models/13B/ggml-model-q5_1.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | tee ../perf-13b-q5_1.txt | grep llama_print_timings
     ./bin/main -m ../models/13B/ggml-model-q8_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | grep "I believe"
time ./bin/main -m ../models/13B/ggml-model-q8_0.bin -p "I believe the meaning of life is" --no-mmap -c 2048 --ignore-eos -s 1 -n 64 -t 8 2>&1 | tee ../perf-13b-q8_0.txt | grep llama_print_timings
