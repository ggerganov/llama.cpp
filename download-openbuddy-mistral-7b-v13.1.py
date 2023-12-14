# https://modelscope.cn/models/OpenBuddy/openbuddy-mistral-7b-v13.1/summary

# 在涵盖数学、历史、法律和其他科目的大规模多任务语言理解测试中，Mistral 的模型准确率达到 60.1%，而 Llama 2 模型 70 亿参数和 130 亿参数两个版本的准确率分别为 44% 和 55%。

# 在常识推理和阅读理解基准测试中，Mistral 的表现也优于 Llama 2 的模型。

# 只有在编码方面 Mistral 落后于 Meta 。Mistral 7B 在 "Humaneval " 和 "MBPP " 两项基准测试中的准确率分别为 30.5% 和 47.5%，而 Llama 2 的 70 亿模式的准确率分别为 31.1% 和 52.5%。

# 下载
import torch
from modelscope import snapshot_download, Model
model_dir = snapshot_download("OpenBuddy/openbuddy-mistral-7b-v13.1", revision = 'v1.0.0',cache_dir="../models")

# 转换
# python convert.py  ../models/OpenBuddy/openbuddy-mistral-7b-v13.1/

# 量化
# ./quantize ../models/OpenBuddy/openbuddy-mistral-7b-v13.1/ggml-model-f16.gguf ../models/ggmls/openbuddy-mistral-7b-v13.1-q4_0.gguf q4_0

#推理
# ./main -m ../models/ggmls/openbuddy-mistral-7b-v13.1-q4_0.gguf -n 128 -p "展示上个季度所有销售额超过 10000 美元的订单,写出SQL" -t 2 -ngl 10
# ./main -m ../models/ggmls/openbuddy-mistral-7b-v13.1-q4_0.gguf -n 256 -p "小丽有3个兄弟, 他们各有2个姐妹, 问小丽有几个姐妹" -t 2 -ngl 10
# ./main -m ../models/ggmls/openbuddy-mistral-7b-v13.1-q4_0.gguf  -ngl 10  -n 256 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-baichuan.txt

# ./main -ngl 32 -m mistral-7b-instruct-v0.1.Q4_K_M.gguf --color -c 4096 --temp 0.7 --repeat_penalty 1.1 -n -1 -p "<s>[INST]{prompt} [/INST]"
