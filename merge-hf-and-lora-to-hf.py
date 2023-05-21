import json
import torch
import argparse
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel

# args with description.
parser = argparse.ArgumentParser(
    prog="Merge HF file with Lora\n",
    description="Please locate HF format model path with pytorch_*.bin inside, lora path with adapter_config.json and adapter_model.bin.",
)

# The original base model checkpoint dir
parser.add_argument(
    "--model_path",
    type=str,
    default="decapoda-research/llama-7b-hf",
    help="Directory contain original HF model",
)
# The finetuned lora model checkpoint dir
parser.add_argument(
    "--lora_path",
    type=str,
    default="decapoda-research/lora",
    help="Directory contain Lora ",
)
# The output dir
parser.add_argument(
    "--out_path",
    type=str,
    default="decapoda-research/lora-merged",
    help="Directory store merged HF model",
)

args = parser.parse_args()

print(f">>> load model from {args.model_path} and lora from {args.lora_path}....")

# transformer loaded. load and save Tokenizer.
tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
tokenizer.save_pretrained(args.out_path)

# load model.
model = LlamaForCausalLM.from_pretrained(
    args.model_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)


# peft loaded. load lora.
model = PeftModel.from_pretrained(
    model,
    args.lora_path,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

print(f">>> merging lora...")

# Using Peft function to merge Lora.
model = model.merge_and_unload()
model.save_pretrained(args.out_path)
