# Author: github.com/ductai199x
import argparse
import os
import struct

import numpy as np
import torch
from numba import njit
from tqdm.auto import tqdm


def read_header(fin):
    values = struct.unpack("i" * 9, fin.read(4 * 9))
    _, _, vocab_size, dim, multiple_of, n_heads, n_layers, rot, ftype = values
    return {
        "vocab_size": vocab_size,
        "dim": dim,
        "multiple_of": multiple_of,
        "n_heads": n_heads,
        "n_layers": n_layers,
    }, ftype


def read_tokens(fin, vocab_size):
    tokens = []
    for _ in range(vocab_size):
        text_len = struct.unpack("i", fin.read(4))[0]
        text_bytes = fin.read(text_len)
        try:
            text = text_bytes.decode()
        except UnicodeDecodeError:
            text = text_bytes.decode(errors="replace")
        score = struct.unpack("f", fin.read(4))[0]
        tokens.append((text, score))
    return tokens


@njit
def dequantize_weights_numba(fin_data, n_rows, n_cols):
    qk = 32
    nb = n_cols // qk
    bs = 4 + (qk // 2)

    weights = np.zeros((n_rows, n_cols), dtype=np.float32)
    data_pos = 0

    for row in range(n_rows):
        for block in range(nb):
            d = np.frombuffer(fin_data[data_pos : data_pos + 4], dtype=np.float32)[0]
            data_pos += 4
            packed_values = fin_data[data_pos : data_pos + (qk // 2)]
            data_pos += qk // 2

            for i in range(qk // 2):
                packed_value = packed_values[i]
                v0 = np.float32((packed_value & 0b00001111) - 8) * d
                v1 = np.float32((packed_value >> 4) - 8) * d

                weights[row, block * qk + 2 * i] = v0
                weights[row, block * qk + 2 * i + 1] = v1

    return weights


def dequantize_weights(fin, n_rows, n_cols):
    qk = 32
    nb = n_cols // qk
    data_size = n_rows * n_cols // 2 + n_rows * nb * 4
    fin_data = fin.read(data_size)
    return dequantize_weights_numba(fin_data, n_rows, n_cols)


def read_variables(fin):
    model = {}
    pbar = tqdm(total=os.path.getsize(fin.name), unit="B", unit_scale=True, desc="Reading variables")
    while True:
        start_pos = fin.tell()
        try:
            n_dims, name_length, ftype_cur = struct.unpack("iii", fin.read(4 * 3))
        except struct.error:
            break

        shape = tuple(struct.unpack("i" * n_dims, fin.read(4 * n_dims)))
        shape = shape[::-1]
        name = fin.read(name_length).decode()

        # ensure tensor data is aligned
        tensor_data_offset = fin.tell()
        tensor_data_offset = (tensor_data_offset + 31) & -32
        fin.seek(tensor_data_offset)

        if ftype_cur == 2:
            # 4-bit quantized weights
            dtype = np.uint8
            data = dequantize_weights(fin, shape[0], shape[1])
            data = data.reshape(shape)
        elif ftype_cur == 0:
            dtype = np.float32
            data_size = np.prod(shape)
            data = np.fromfile(fin, dtype=dtype, count=data_size).reshape(shape)
        elif ftype_cur == 1:
            dtype = np.float16
            data_size = np.prod(shape)
            data = np.fromfile(fin, dtype=dtype, count=data_size).reshape(shape)

        model[name] = torch.tensor(data, dtype=torch.float32 if dtype == np.float32 else torch.float16)

        pbar.update(fin.tell() - start_pos)

    return model


def convert_to_hf_format(model, hparams):
    # This works for llama 7B, need to test with other models
    n_layers = hparams["n_layers"]
    n_heads = hparams["n_heads"]
    dim = hparams["dim"]
    dims_per_head = dim // n_heads
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

    # permute for sliced rotary
    def permute(w):
        return w.view(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)

    state_dict = {}
    for layer_i in range(n_layers):
        state_dict.update(
            {
                f"model.layers.{layer_i}.self_attn.q_proj.weight": permute(
                    model[f"layers.{layer_i}.attention.wq.weight"]
                ),
                f"model.layers.{layer_i}.self_attn.k_proj.weight": permute(
                    model[f"layers.{layer_i}.attention.wk.weight"]
                ),
                f"model.layers.{layer_i}.self_attn.v_proj.weight": model[
                    f"layers.{layer_i}.attention.wv.weight"
                ],
                f"model.layers.{layer_i}.self_attn.o_proj.weight": model[
                    f"layers.{layer_i}.attention.wo.weight"
                ],
                f"model.layers.{layer_i}.mlp.gate_proj.weight": model[
                    f"layers.{layer_i}.feed_forward.w1.weight"
                ],
                f"model.layers.{layer_i}.mlp.down_proj.weight": model[
                    f"layers.{layer_i}.feed_forward.w2.weight"
                ],
                f"model.layers.{layer_i}.mlp.up_proj.weight": model[
                    f"layers.{layer_i}.feed_forward.w3.weight"
                ],
                f"model.layers.{layer_i}.input_layernorm.weight": model[
                    f"layers.{layer_i}.attention_norm.weight"
                ],
                f"model.layers.{layer_i}.post_attention_layernorm.weight": model[
                    f"layers.{layer_i}.ffn_norm.weight"
                ],
            }
        )
        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
    state_dict.update(
        {
            "model.embed_tokens.weight": model["tok_embeddings.weight"],
            "model.norm.weight": model["norm.weight"],
            "lm_head.weight": model["output.weight"],
        }
    )

    return state_dict


def chat(model, hparams, llama_dir):
    from transformers import (GenerationConfig, LlamaForCausalLM,
                              LlamaTokenizer, StoppingCriteria,
                              StoppingCriteriaList)
    from transformers.models.llama.configuration_llama import LlamaConfig

    class StoppingCriteriaSub(StoppingCriteria):
        def __init__(self):
            super().__init__()

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops=[]):
            print(tokenizer.decode(input_ids[0]), end="", flush=True)
            if input_ids[0][-1] == 13:
                return True

            return False

    config = LlamaConfig(
        vocab_size=hparams["vocab_size"],
        dim=hparams["dim"],
        num_hidden_layers=hparams["n_layers"],
        num_attention_heads=hparams["n_heads"],
    )

    llama = LlamaForCausalLM(config=config)
    llama.load_state_dict(state_dict=model, strict=True)
    tokenizer = LlamaTokenizer.from_pretrained(llama_dir)

    device = torch.device("cpu")
    llama = llama.to(device)

    ctx = """You are AI.
This is a dialog, where User interacts with AI. AI is helpful, kind, obedient, honest, respectful, direct, concise, should try to protect User's privacy, and knows its own limits. Also, AI must answer User and AI cannot stop the conversation by itself.
User: Hello, AI.
AI: Hello! How can I assist you today?
"""
    print(ctx.rstrip("\n"))
    while True:
        print("-" * 60)
        prompt = input("User: ")
        if ctx != "":
            ctx = f"{ctx}User: {prompt}\n"
        else:
            ctx = f"{prompt}\nAI:"

        ctx = (ctx[-1920:]) if len(ctx) >= 2048 else ctx

        print("-" * 60)
        if len(ctx.strip()) > 0:
            input_ids = tokenizer(ctx, return_tensors="pt")["input_ids"].to(device)
            generation_config = GenerationConfig(
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1764,
            )
            with torch.no_grad():
                generation_output = llama.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=2048,
                    do_sample=True,
                    stopping_criteria=StoppingCriteriaList([StoppingCriteriaSub()]),
                )
            s = generation_output.sequences[0]
            decoded = tokenizer.decode(s)
            ctx = f"{decoded}\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", "-i", type=str, required=True, help="The input directory containing the ggml files."
    )
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        required=True,
        help="The prefix of the ggml files (ggml-model-f16 or ggml-model-q4_0).",
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Whether to save the model in the Hugging Face format. (default: False)",
    )
    parser.add_argument(
        "--chat", "-c", action="store_true", help="Whether to open a chat with the model. (default: False)"
    )
    args = parser.parse_args()

    llama_dir = os.path.abspath(f"{args.input_dir}/../")

    ggml_files = sorted(
        [f"{args.input_dir}/{f}" for f in os.listdir(args.input_dir) if f.startswith(args.prefix)]
    )

    fin = open(ggml_files[0], "rb")
    hparams, ftype = read_header(fin)
    tokens = read_tokens(fin, hparams["vocab_size"])
    model = read_variables(fin)

    for f in tqdm(ggml_files[1:]):
        fin = open(f, "rb")
        read_header(fin)
        read_tokens(fin, hparams["vocab_size"])
        model.update(read_variables(fin))

    if args.hf:
        model = convert_to_hf_format(model, hparams)

    pth_ckpt = {
        "state_dict": model,
        "hparams": hparams,
        "tokens": tokens,
    }

    torch.save(pth_ckpt, f"{args.input_dir}/{args.prefix}-to-torch.pth")

    if args.chat:
        if not args.hf:
            model = convert_to_hf_format(model, hparams)
        chat(model, hparams, llama_dir)


if __name__ == "__main__":
    main()
