#!/bin/bash

#
# Temporary script - will be removed in the future
#

cd `dirname $0`
cd ..

# ./main -m models/available/Llama2/7B/llama-2-7b.ggmlv3.q4_0.bin \
#        --color \
#        --ctx_size 2048 \
#        -n -1 \
#        -ins -b 256 \
#        --top_k 10000 \
#        --temp 0.2 \
#        --repeat_penalty 1.1 \
#        -t 8

# epetition_penalty –（可选）float 重复惩罚的参数。在 1.0 和无穷大之间。1.0 意味着没有惩罚。默认为 1.0。
# temperature –（可选）float 用于对下一个标记概率进行建模的值。必须是严格正的。默认为 1.0。
# top_k –（可选）int 为 top-k 过滤保留的最高概率词汇表标记的数量。在 1 和无穷大之间。默认为 50。
./main -m ../models/ggmls/slqm-bc2-13b-chat-q2_k.gguf \
       --color \
       --ctx_size 2048 \
       -n -1 \
       -ins -b 256 \
       --top_k 10000 \
       --temp 0.2 \
       --repeat_penalty 1.1 \
       -t 2
