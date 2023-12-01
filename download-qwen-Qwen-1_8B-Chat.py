
import torch
from modelscope import AutoTokenizer, snapshot_download
from modelscope import AutoModelForCausalLM

model_dir = snapshot_download('qwen/Qwen-1_8B-Chat',cache_dir="../models")
model_dir = snapshot_download('Qwen/Qwen-1_8B-Chat-Int4',cache_dir="../models")

from modelscope import AutoTokenizer, AutoModelForCausalLM, snapshot_download
tokenizer = AutoTokenizer.from_pretrained("../models/qwen/Qwen-1_8B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "../models/qwen/Qwen-1_8B-Chat",
    device_map="cpu",
    trust_remote_code=True
).eval()
response, history = model.chat(tokenizer, "你好", history=None)
print(response)
response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)
print(response)
response, _ = model.chat(tokenizer, "你好呀", history=None, system="请用二次元可爱语气和我说话")
print(response)
response, _ = model.chat(tokenizer, "My colleague works diligently", history=None, system="You will write beautiful compliments according to needs")
print(response)

