import torch
import argparse
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel

# args with description.
parser = argparse.ArgumentParser(
    description="Merge HF model with LoRA",
)

# The original base model checkpoint dir
parser.add_argument(
    "--model",
    type=str,
    default="decapoda-research/llama-7b-hf",
    help="HF model name or path",
)
# The finetuned lora model checkpoint dir
parser.add_argument(
    "--lora",
    type=str,
    required=True,
    help="LoRA model name or path",
)
# The output dir
parser.add_argument(
    "--out",
    type=str,
    required=True,
    help="Directory store merged HF model",
)

args = parser.parse_args()

print(f">>> load model from {args.model} and lora from {args.lora}....")

# transformer loaded. load and save Tokenizer.
tokenizer = LlamaTokenizer.from_pretrained(args.model)
tokenizer.save_pretrained(args.out)

# load model.
model = LlamaForCausalLM.from_pretrained(
    args.model,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

# peft loaded. load lora.
model = PeftModel.from_pretrained(
    model,
    args.lora,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

print(f">>> merging lora...")

# Using Peft function to merge Lora.
model = model.merge_and_unload()
model.save_pretrained(args.out)
