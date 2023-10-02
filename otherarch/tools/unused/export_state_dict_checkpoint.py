# this specific file adapted from https://github.com/tloen/alpaca-lora/blob/main/export_state_dict_checkpoint.py
# under Apache 2.0 license https://raw.githubusercontent.com/tloen/alpaca-lora/main/LICENSE
# todo: adapt to revert HF formats back to original PTH formats so ggml can convert them.

import json
import os

import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: E402

BASE_MODEL = os.environ.get("BASE_MODEL", None)
assert (
    BASE_MODEL
), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=decapoda-research/llama-7b-hf`"  # noqa: E501

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

lora_model = PeftModel.from_pretrained(
    base_model,
    "tloen/alpaca-lora-7b",
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

# merge weights
for layer in lora_model.base_model.model.model.layers:
    layer.self_attn.q_proj.merge_weights = True
    layer.self_attn.v_proj.merge_weights = True

lora_model.train(False)

lora_model_sd = lora_model.state_dict()

params = {
    "dim": 4096,
    "multiple_of": 256,
    "n_heads": 32,
    "n_layers": 32,
    "norm_eps": 1e-06,
    "vocab_size": -1,
}
n_layers = params["n_layers"]
n_heads = params["n_heads"]
dim = params["dim"]
dims_per_head = dim // n_heads
base = 10000.0
inv_freq = 1.0 / (
    base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head)
)


def permute(w):
    return (
        w.view(n_heads, dim // n_heads // 2, 2, dim)
        .transpose(1, 2)
        .reshape(dim, dim)
    )


def unpermute(w):
    return (
        w.view(n_heads, 2, dim // n_heads // 2, dim)
        .transpose(1, 2)
        .reshape(dim, dim)
    )


def translate_state_dict_key(k):  # noqa: C901
    k = k.replace("base_model.model.", "")
    if k == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    elif k == "model.norm.weight":
        return "norm.weight"
    elif k == "lm_head.weight":
        return "output.weight"
    elif k.startswith("model.layers."):
        layer = k.split(".")[2]
        if k.endswith(".self_attn.q_proj.weight"):
            return f"layers.{layer}.attention.wq.weight"
        elif k.endswith(".self_attn.k_proj.weight"):
            return f"layers.{layer}.attention.wk.weight"
        elif k.endswith(".self_attn.v_proj.weight"):
            return f"layers.{layer}.attention.wv.weight"
        elif k.endswith(".self_attn.o_proj.weight"):
            return f"layers.{layer}.attention.wo.weight"
        elif k.endswith(".mlp.gate_proj.weight"):
            return f"layers.{layer}.feed_forward.w1.weight"
        elif k.endswith(".mlp.down_proj.weight"):
            return f"layers.{layer}.feed_forward.w2.weight"
        elif k.endswith(".mlp.up_proj.weight"):
            return f"layers.{layer}.feed_forward.w3.weight"
        elif k.endswith(".input_layernorm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"):
            return f"layers.{layer}.ffn_norm.weight"
        elif k.endswith("rotary_emb.inv_freq") or "lora" in k:
            return None
        else:
            print(layer, k)
            raise NotImplementedError
    else:
        print(k)
        raise NotImplementedError


new_state_dict = {}
for k, v in lora_model_sd.items():
    new_k = translate_state_dict_key(k)
    if new_k is not None:
        if "wq" in new_k or "wk" in new_k:
            new_state_dict[new_k] = unpermute(v)
        else:
            new_state_dict[new_k] = v

os.makedirs("./ckpt", exist_ok=True)

torch.save(new_state_dict, "./ckpt/consolidated.00.pth")

with open("./ckpt/params.json", "w") as f:
    json.dump(params, f)