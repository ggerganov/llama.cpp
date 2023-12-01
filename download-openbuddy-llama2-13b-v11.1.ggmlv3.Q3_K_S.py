# https://modelscope.cn/models/Xorbits/OpenBuddy-Llama2-13B-v11.1-GGML/summary

# 下载
from modelscope.hub.file_download import model_file_download
model_dir = model_file_download(model_id='Xorbits/OpenBuddy-Llama2-13B-v11.1-GGML',file_path='openbuddy-llama2-13b-v11.1.ggmlv3.Q3_K_S.bin',cache_dir="../models")


python convert-llama-ggml-to-gguf.py --input ../models/ggmls/openbuddy-llama2-13b-v11.1.ggmlv3.Q3_K_S.bin --output ../models/ggmls/openbuddy-llama2-13b-v11.1.ggmlv3.Q3_K_S.gguf

#推理
# ./main -m ../models/ggmls/openbuddy-llama2-13b-v11.1.ggmlv3.Q3_K_S.gguf -n 128 -p "展示上个季度所有销售额超过 10000 美元的订单,写出SQL" -t 2 -ngl 10
# ./main -m ../models/ggmls/openbuddy-llama2-13b-v11.1.ggmlv3.Q3_K_S.gguf -n 256 -p "小丽有3个兄弟, 他们各有2个姐妹, 问小丽有几个姐妹" -t 2 -ngl 10
# ./main -m ../models/ggmls/openbuddy-llama2-13b-v11.1.ggmlv3.Q3_K_S.gguf  -ngl 10  -n 256 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-baichuan.txt

