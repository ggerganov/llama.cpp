import os, time
import tempfile
import json
import torch
import argparse
import transformers
from transformers import LlamaTokenizer, LlamaConfig, LlamaForCausalLM
from peft import PeftModel, LoraConfig, LoraModel

# args
parser = argparse.ArgumentParser()
# The original base model checkpoint dir
parser.add_argument("--model_path", type=str, default='llama-7b-hf')
# The finetuned lora model checkpoint dir
parser.add_argument("--lora_path",type=str, default='lora')
# The output dir
parser.add_argument("--out_path", type=str, default='lora-merged')

args = parser.parse_args()



print(f">>> load model from {args.model_path} and lora from {args.lora_path}....")

tokenizer = LlamaTokenizer.from_pretrained(args.model_path)

#transformer loaded. load  model.

model = LlamaForCausalLM.from_pretrained(
    args.model_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
    
)


#peft loaded. load lora.
model = PeftModel.from_pretrained(
    model,
    args.lora_path,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

print(f">>> merging lora...")

#Why 'LlamaForCausalLM' object has no attribute 'merge_and_unload' ????????
#okay, it works, i don't know why it didn't.

model = model.merge_and_unload()
model.save_pretrained(args.out_path)

