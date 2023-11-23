# https://modelscope.cn/models/modelscope/zephyr-7b-beta/summary
# 7b达到gpt3.5  超越 llama2-70b
# Zephyr-7B-α是一系列 Zephyr 经过训练的语言模型中的第一个模型，是 Mistral-7B-v0.1 的微调版本，在使用直接偏好优化的混合公开合成数据集上进行训练。

# 数据也显示，Zephyr高级RAG任务效果可以和GPT-3.5、Claude 2相抗衡。
# 他们还继续补充道，Zephyr不仅在RAG上效果突出，而且在路由、查询规划、检索复杂SQL语句、结构化数据提取方面也表现良好。
# 但在编码和数学等更复杂的任务上，Zephyr-7B-beta落后于专有模型，需要更多的研究来缩小差距。

# 开发人员却表示，最有趣的不是各项指标，而是模型的训练方式。
# 亮点总结如下：
#     微调最好的小型开源预训练模型：Mistral 7B
#     大规模偏好数据集的用法：UltraFeedback
#     不用强化学习，使用直接偏好优化（DPO）
#     意料之外的是，偏好数据集的过拟合会产生更好的效果

# 用这种方法微调模型，成本只需500美元，也就是在16个A100上跑8小时。


# 下载
import torch
from modelscope import snapshot_download, Model
model_dir = snapshot_download("modelscope/zephyr-7b-beta",cache_dir="../models")

# 转换
python convert.py  ../models/modelscope/zephyr-7b-beta/

# 量化
# ./quantize ../models/modelscope/zephyr-7b-beta/ggml-model-f16.gguf ../models/ggml-model-f16-zephyr-7b-beta-q8_0.gguf q8_0
./quantize ../models/modelscope/zephyr-7b-beta/ggml-model-f16.gguf ../models/ggml-model-f16-zephyr-7b-beta-q5_0.gguf q5_0

#推理
./main -m ../models/ggml-model-f16-zephyr-7b-beta-q5_0.gguf -n  256 -p "为什么我爸妈结婚的时候没邀请我参加婚礼" -t 2 -ngl 4
./main -m ../models/ggml-model-f16-zephyr-7b-beta-q5_0.gguf -n  256 -p "能给我讲讲GPT-4吗?" -t 2 -ngl 4
./main -m ../models/ggml-model-f16-zephyr-7b-beta-q5_0.gguf   -ngl 10  -n 256 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-baichuan.txt

