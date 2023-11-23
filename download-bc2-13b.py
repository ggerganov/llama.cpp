import torch
from modelscope import snapshot_download, Model
model_dir = snapshot_download("baichuan-inc/Baichuan2-13B-Chat", revision='v1.0.3')
model = Model.from_pretrained(model_dir, device_map="balanced", trust_remote_code=True, torch_dtype=torch.float16)
messages = []
messages.append({"role": "user", "content": "讲解一下“温故而知新”"})
response = model(messages)
print(response)
messages = response['history'].copy()
messages.append({"role": "user", "content": "背诵一下将进酒"})
response = model(messages)
print(response)

#python convert.py /root/.cache/modelscope/hub/baichuan-inc/Baichuan2-13B-Chat-4bits/
# ./main -m ggml-model-q4_0.gguf -n 128 -p "莫勇开头写一首藏头诗"

