# https://modelscope.cn/models/TabbyML/Mistral-7B/summary

# 下载
import torch
from modelscope import snapshot_download, Model
model_dir = snapshot_download("TabbyML/Mistral-7B",cache_dir="../models")

# 转换
# python convert.py  ../models/TabbyML/Mistral-7B/

# 量化
# ./quantize ../models/TabbyML/Mistral-7B/ggml-model-f16.gguf ../models/ggml-model-f16-zephyr-7b-beta-q8_0.gguf q8_0
# ./quantize ../models/TabbyML/Mistral-7B/ggml-model-f16.gguf ../models/ggml-model-f16-zephyr-7b-beta-q5_0.gguf q5_0

#推理
# ./main -m ../models/ggml-model-f16-zephyr-7b-beta-q5_0.gguf -n 128 -p "How many helicopters can a human eat in one sitting?" -t 2 -ngl 4

