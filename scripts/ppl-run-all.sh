#!/bin/bash

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

#
# perplexity
#

# 7B
time ./bin/perplexity -m ../models/7B/ggml-model-f16.bin  -f ./wiki.test.raw --no-mmap -t 12 2>&1 | tee ../ppl-7b-f16.txt
time ./bin/perplexity -m ../models/7B/ggml-model-q4_0.bin -f ./wiki.test.raw --no-mmap -t 12 2>&1 | tee ../ppl-7b-q4_0.txt
time ./bin/perplexity -m ../models/7B/ggml-model-q4_1.bin -f ./wiki.test.raw --no-mmap -t 12 2>&1 | tee ../ppl-7b-q4_1.txt
time ./bin/perplexity -m ../models/7B/ggml-model-q5_0.bin -f ./wiki.test.raw --no-mmap -t 12 2>&1 | tee ../ppl-7b-q5_0.txt
time ./bin/perplexity -m ../models/7B/ggml-model-q5_1.bin -f ./wiki.test.raw --no-mmap -t 12 2>&1 | tee ../ppl-7b-q5_1.txt
time ./bin/perplexity -m ../models/7B/ggml-model-q8_0.bin -f ./wiki.test.raw --no-mmap -t 12 2>&1 | tee ../ppl-7b-q8_0.txt

# 13B
time ./bin/perplexity -m ../models/13B/ggml-model-f16.bin  -f ./wiki.test.raw --no-mmap -t 12 2>&1 | tee ../ppl-13b-f16.txt
time ./bin/perplexity -m ../models/13B/ggml-model-q4_0.bin -f ./wiki.test.raw --no-mmap -t 12 2>&1 | tee ../ppl-13b-q4_0.txt
time ./bin/perplexity -m ../models/13B/ggml-model-q4_1.bin -f ./wiki.test.raw --no-mmap -t 12 2>&1 | tee ../ppl-13b-q4_1.txt
time ./bin/perplexity -m ../models/13B/ggml-model-q5_0.bin -f ./wiki.test.raw --no-mmap -t 12 2>&1 | tee ../ppl-13b-q5_0.txt
time ./bin/perplexity -m ../models/13B/ggml-model-q5_1.bin -f ./wiki.test.raw --no-mmap -t 12 2>&1 | tee ../ppl-13b-q5_1.txt
time ./bin/perplexity -m ../models/13B/ggml-model-q8_0.bin -f ./wiki.test.raw --no-mmap -t 12 2>&1 | tee ../ppl-13b-q8_0.txt
