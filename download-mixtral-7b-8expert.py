# # transformers>=4.36 (build from source)
# import torch
# from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download

# model = AutoModelForCausalLM.from_pretrained('AI-ModelScope/mixtral-7b-8expert', low_cpu_mem_usage=True, 
#                                              device_map="auto", trust_remote_code=True)
# tok = AutoTokenizer.from_pretrained('AI-ModelScope/mixtral-7b-8expert')
# x = tok.encode("The mistral wind in is a phenomenon ", return_tensors="pt").cuda()
# x = model.generate(x, max_new_tokens=128).cpu()
# print(tok.batch_decode(x))

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/mixtral-7b-8expert',cache_dir="../models")