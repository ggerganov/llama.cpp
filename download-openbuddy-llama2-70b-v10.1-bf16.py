# https://modelscope.cn/models/CarbonAgent/llama-2-13b-chat.Q4/summary
# 将自然语言转换为 SQL 查询语言

# 这个模型的主要目标是在垂直行业中进行专业数据代理。通过使用 llama-2-13b-chat，业务人员可以直接使用自然语言来查询数据库，而无需掌握复杂的 SQL 查询语法。这不仅可以提升业务人员的工作效率，也可以减少对 IT 人员的依赖。
# 例如，销售人员可以通过输入 "展示上个季度所有销售额超过 10000 美元的订单"，llama-2-13b-chat 会将这个查询转换为对应的 SQL 查询，如 "SELECT * FROM orders WHERE sales > 10000 AND quarter = 'Q2';"。



# 9月4日，OpenBuddy发布700亿参数跨语言大模型 OpenBuddy-LLaMA2-70B，并以可商用的形态全面开源！现在已经全面上架魔搭ModelScope社区。
# 70B模型在能力表现上，相较于早前发布的较小规模模型，在文本生成、复杂逻辑推理以及自然语言处理等任务有了非常显著的提升。据其内测用户及多项能力测试指标反馈，目前70B模型在语言能力和逻辑推理能力可对标为GPT3.5的开源平替！OpenBuddy社区希望用开源激发中国大模型行业的潜能。
# GitHub链接：https://github.com/OpenBuddy/OpenBuddy

# from modelscope.hub.snapshot_download import snapshot_download
# model_dir = snapshot_download('OpenBuddy/openbuddy-llama2-70b-v10.1-bf16', 'v1.0.0',cache_dir="../models")
# model_dir = snapshot_download('Xorbits/Llama-2-70B-Chat-GGML', 'v1.0.0',cache_dir="../models")
from modelscope.hub.file_download import model_file_download
model_dir = model_file_download(model_id='Xorbits/Llama-2-70B-Chat-GGML',file_path='llama-2-70b-chat.ggmlv3.q3_K_S.bin',cache_dir="../models")


# python convert.py  ../models/OpenBuddy/openbuddy-llama2-70b-v10.1-bf16/

# ./quantize ../models/OpenBuddy/openbuddy-llama2-70b-v10.1-bf16/ggml-model-f16.gguf ../models/ggmls/openbuddy-llama2-70b-v10.1-bf16-q3_k_s.gguf q3_k_s

# ./main -m ../models/ggmls/openbuddy-llama2-70b-v10.1-bf16-q3_k_s.gguf -n 128 -p "展示上个季度所有销售额超过 10000 美元的订单,写出对应的SQL语句" -t 2 -ngl 4
# ./main -t 10 -ngl 40 -gqa 8 -m llama-2-70b-chat.ggmlv3.q4_K_M.bin --color -c 4096 --temp 0.7 --repeat_penalty 1.1 -n -1 -p "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\nWrite a story about llamas[/INST]"

# ./main -m llama-2-70b.ggmlv3.q4_0.bin -gqa 8 -t 13 -p "Llamas are"
