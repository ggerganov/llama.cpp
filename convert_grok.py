"""
Convert Grok-1 weights to GGUF format.

Example invocation:

python -m convert_grok -i path/to/grok-1/ckpt-0 --vocab_dir path/to/grok -o grok.bin -t q4_0 --experts 1,2

To run:

./build/bin/main -m grok.bin -p "The answer to life the universe and everything is" -s 1 -n 3 -ngl 1
"""

import argparse
import mmap
import os
import pathlib
import pickletools
import sys
import time

import ml_dtypes
import numpy as np
import torch

try:
    from tabulate import tabulate
except ModuleNotFoundError:
    pass

from convert import SentencePieceVocab

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(pathlib.Path(__file__).parent / "gguf-py"))

import gguf

GGML_QK8_0 = 32
GGML_QK4_0 = 32
GGML_QK4_1 = 32


# Heuristic to avoid having to fully parse pickle files.
FP32_SHAPES = {805306368: (131072, 6144), 6144: (6144,), 49152: (6144, 8)}
BF16_SHAPES = {
    262144: (8, 1, 32768),
    393216: (8, 8, 6144),
    1024: (1, 1024),
    49152: (8, 6144),
    6144: (1, 6144),
}


class AttributeDict(dict):
    def __getattr__(self, key):
        return self.__getitem__(key) if key in self else super().__getattr__(key)

    __setattr__ = dict.__setitem__


def _genops(data):
    view = memoryview(data)

    code2op = {ord(d.code): d for d in pickletools.opcodes}
    dataops = {
        "BINBYTES": pickletools.read_uint4,
        "BINBYTES8": pickletools.read_uint8,
    }

    while True:
        pos = data.tell()
        code = data.read_byte()
        opcode = code2op[code]

        arg = None
        if opcode.arg is not None:
            if opcode.name not in dataops:
                arg = opcode.arg.reader(data)
            else:
                size = dataops[opcode.name](data)
                p = data.tell()
                arg = np.frombuffer(view[p : p + size], dtype=np.uint8)
                data.seek(size, 1)

        yield opcode, arg, pos
        if code == ord(b"."):
            break


def genops(fn):
    """Yield (opcode, arg, pos) from for a pickle file.

    Uses mmap to avoid copies of binary data (e.g., np and JAX arrays)."""
    with open(fn, "rb") as f:
        yield from _genops(mmap.mmap(f.fileno(), length=0, flags=mmap.MAP_PRIVATE))


def get_weights(fn):
    """Returns tensor/array data in Grok pickle files, zero copy."""

    arrays = []
    for unused_opcode, arg, unused_pos in genops(fn):
        if isinstance(arg, np.ndarray):
            arrays.append(arg)

    if len(arrays) == 1:
        # Plain numpy array.
        array = arrays[0].view(np.float32)
        array = array.reshape(FP32_SHAPES[array.size])
        return array, None
    elif len(arrays) == 2:
        weight, scales = arrays

        scales = scales.view(ml_dtypes.bfloat16)
        scales = scales.reshape(BF16_SHAPES[scales.size])

        weight = weight.view(np.int8)
        shape = list(scales.shape)
        shape[-2] = -1
        weight = weight.reshape(shape)
        return weight, scales

    assert len(arrays) in (1, 2)


def quantize_q8_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q8_0 in ggml.c
    assert tensor.shape[1] % GGML_QK8_0 == 0
    tensor = tensor.view(-1, GGML_QK8_0)
    scale = tensor.abs().max(dim=-1, keepdim=True).values / ((1 << 7) - 1)
    tensor = (tensor / scale).round().clamp(min=-128, max=127).char()
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), tensor), dim=-1)
    return tensor


def quantize_q4_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q4_0 in ggml.c
    assert tensor.shape[1] % GGML_QK4_0 == 0
    tensor = tensor.reshape(-1, GGML_QK4_0)
    abs_max_indices = tensor.abs().max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    scale = max_values / -8
    tensor = (tensor / scale + 8).round().clamp(min=0, max=15).char()
    # compress two int4 weights into a int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), tensor), dim=-1)
    return tensor


def quantize_q4_1(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q4_1 in ggml.c
    assert tensor.shape[1] % GGML_QK4_1 == 0
    tensor = tensor.view(-1, GGML_QK4_1)
    abs_max_indices = tensor.max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    abs_min_indices = tensor.min(dim=-1, keepdim=True).indices
    min_values = torch.take_along_dim(tensor, abs_min_indices, dim=-1)
    scale = (max_values - min_values) / 15
    tensor = ((tensor - min_values) / scale).round().clamp(min=0, max=15).char()
    # compress two int4 weights into a int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)
    # add scale into each block
    tensor = torch.cat(
        (scale.half().view(torch.int8), min_values.half().view(torch.int8), tensor), dim=-1
    )
    return tensor


def maybe_quantize_tensor(tensor, ggml_type):
    assert tensor.dtype == torch.float32

    if ggml_type == gguf.GGMLQuantizationType.F32:
        return tensor.float()
    elif ggml_type == gguf.GGMLQuantizationType.F16:
        return tensor.half()
    elif ggml_type == gguf.GGMLQuantizationType.Q8_0:
        return quantize_q8_0(tensor)
    elif ggml_type == gguf.GGMLQuantizationType.Q4_0:
        return quantize_q4_0(tensor)
    elif ggml_type == gguf.GGMLQuantizationType.Q4_1:
        return quantize_q4_1(tensor)
    else:
        raise NotImplementedError(f"Cannot quantize tensor of dtype {tensor.dtype} ({ggml_type})")


def get_dtype_and_ggml_type(tensor, ggml_type):
    if tensor.ndim == 2:
        if tensor.shape[1] % GGML_QK8_0 == 0:
            return np.int8, ggml_type
        else:
            return np.float16, gguf.GGMLQuantizationType.F16
    else:
        # 1d weight: convert it to float32
        assert tensor.ndim == 1
        return np.float32, gguf.GGMLQuantizationType.F32


def dump_state_dict(f, weight_names, model_files, ggml_type, config):
    keys2names = {}
    meta_tensors = {}
    weight_scales = {}

    # First operate on meta tensors to find shapes and dtypes for GGUF header.
    for name, fn in model_files:
        weight, scales = get_weights(fn)
        meta_state_dict = convert_weight(name, weight, scales, config.experts, device="meta")
        weight_scales[name] = (weight, scales)
        for key in meta_state_dict.keys():
            keys2names[key] = name

        meta_tensors.update(meta_state_dict)

    for key in weight_names:
        meta_tensor = meta_tensors[key]
        dtype, tensor_ggml_type = get_dtype_and_ggml_type(meta_tensor, ggml_type)
        quantized_meta_tensor = maybe_quantize_tensor(meta_tensor, tensor_ggml_type)
        f.add_tensor_info(
            key, list(meta_tensor.shape), dtype, quantized_meta_tensor.nbytes, tensor_ggml_type
        )
    print("Loaded", len(meta_tensors), "files")

    f.write_header_to_file()
    f.write_kv_data_to_file()
    f.write_ti_data_to_file()

    cache = {}
    tensor_info = []

    for key in weight_names:
        if key not in cache:
            name = keys2names[key]
            weight, scales = weight_scales.pop(name)
            state_dict = convert_weight(name, weight, scales, config.experts)
            permute_tensors(state_dict, config)
            cache.update(state_dict)
        tensor = cache.pop(key)
        _, tensor_ggml_type = get_dtype_and_ggml_type(tensor, ggml_type)
        tensor = maybe_quantize_tensor(tensor, tensor_ggml_type)

        array = tensor.numpy()
        print(
            f"dumping {key}: {tensor_ggml_type.name}/{array.dtype}, {array.shape}, {array.nbytes} bytes"
        )
        f.write_tensor_data(array)

        tensor_info.append((key, tensor.shape, tensor_ggml_type.name))

    try:
        print(tabulate(tensor_info, headers=["name", "shape", "dtype"], tablefmt="psql"))
    except NameError:
        pass

    if len(tensor_info) != len(weight_names):
        print("Warning: not all tensors are converted")


def from_numpy(array):
    """Like torch.from_numpy, but handle ml_dtypes.bfloat16 too."""

    if array.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(array.view(np.uint8)).view(torch.bfloat16)
    return torch.from_numpy(array)


def convert_weight(tensor_name, weight, scales, experts, dtype=torch.float32, device=None):
    # copied from https://gist.github.com/chu-tianxiang/ec310e15d56949fd0f351cb5f65ee7a1
    result = {}

    weight = from_numpy(weight).to(device=device, dtype=dtype)
    if scales is not None:
        scale = from_numpy(scales).to(device=device, dtype=dtype)
        # row parallel layers have sharded scale
        if len(scale.shape) >= 2 and scale.shape[-2] != 1:
            scale = scale[..., None, :]
            weight = weight.view(*weight.shape[:-2], 8, -1, weight.shape[-1])
            weight = (weight * scale).view(*weight.shape[:-3], -1, weight.shape[-1])
        else:
            weight = weight * scale

    # Transpose linear matrix
    if len(weight.shape) >= 2 and "token_embd" not in tensor_name:
        weight = weight.transpose(-1, -2)

    if tensor_name.endswith("ffn_gate_inp.weight"):
        result[tensor_name] = weight[experts]  # gather.
    elif "experts" not in tensor_name:
        result[tensor_name] = weight
    else:
        # split moe
        for i, expert in enumerate(experts):
            key = tensor_name.replace("experts", str(i))
            result[key] = weight[expert]

    return result


def permute_tensors(state_dict, config):
    def permute(weights, n_head):
        return (
            weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )

    for name, tensor in state_dict.items():
        if name.endswith("attn_k.weight"):
            state_dict[name] = permute(tensor, config.num_key_value_heads)
        elif name.endswith("attn_q.weight"):
            state_dict[name] = permute(tensor, config.num_attention_heads)


def extract_vocabulary_from_model(vocab):
    tokens = []
    scores = []
    toktypes = []

    for text, score, toktype in vocab.all_tokens():
        tokens.append(text)
        scores.append(score)
        toktypes.append(toktype)

    assert len(tokens) == vocab.vocab_size

    return tokens, scores, toktypes


def get_weight_names(config):
    weight_names = ["token_embd.weight"]
    for i in range(config.num_hidden_layers):
        for j in range(config.num_experts):
            weight_names += [
                f"blk.{i}.ffn_gate.{j}.weight",
                f"blk.{i}.ffn_down.{j}.weight",
                f"blk.{i}.ffn_up.{j}.weight",
            ]

        weight_names += [
            f"blk.{i}.attn_k.weight",
            f"blk.{i}.attn_output.weight",
            f"blk.{i}.attn_q.weight",
            f"blk.{i}.attn_v.weight",
            f"blk.{i}.attn_norm.weight",
            f"blk.{i}.attn_output_norm.weight",
            f"blk.{i}.ffn_norm.weight",
            f"blk.{i}.layer_output_norm.weight",
            f"blk.{i}.ffn_gate_inp.weight",
        ]

    weight_names += ["output_norm.weight"]

    return weight_names


def convert_grok(args, vocab, ggml_type):
    start = time.time()

    def ffn_size(emb_size, widening_factor):
        _ffn_size = int(widening_factor * emb_size) * 2 // 3
        _ffn_size = _ffn_size + (8 - _ffn_size) % 8  # ensure it's a multiple of 8
        return _ffn_size

    config = {
        "vocab_size": 128 * 1024,
        "hidden_act": "gelu",
        "pad_token_id": 0,
        "eos_token_id": 2,
        "max_position_embeddings": 8192,
        "output_multiplier_scale": 0.5773502691896257,
        "embedding_multiplier_scale": 78.38367176906169,
        "hidden_size": 48 * 128,
        "intermediate_size": -1,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "num_hidden_layers": 64,  # Change to 1 for quicker debugging.
        "num_selected_experts": 2,
        "rope_theta": 10000,
        "attn_output_multiplier": 0.08838834764831845,
        "rms_norm_eps": 1e-5,
    }

    config = AttributeDict(config)

    config.intermediate_size = ffn_size(config.hidden_size, 8)

    config.experts = list(range(8))
    if args.experts != "":
        config.experts = [int(x, 0) for x in args.experts.split(",")]

    config.num_experts = len(config.experts)

    assert config.num_experts >= 2, "need at least 2 experts"
    print("experts to export:", config.experts)

    # Contents of in Grok-1 pickle files, in order. Weights with "experts" will be split later.
    tensor_names = [
        "token_embd.weight",
        "output_norm.weight",
    ]
    for i in range(config.num_hidden_layers):
        tensor_names += [
            f"blk.{i}.ffn_gate.experts.weight",
            f"blk.{i}.ffn_down.experts.weight",
            f"blk.{i}.ffn_up.experts.weight",
            f"blk.{i}.attn_k.weight",
            f"blk.{i}.attn_output.weight",
            f"blk.{i}.attn_q.weight",
            f"blk.{i}.attn_v.weight",
            f"blk.{i}.attn_norm.weight",
            f"blk.{i}.attn_output_norm.weight",
            f"blk.{i}.ffn_norm.weight",
            f"blk.{i}.layer_output_norm.weight",
            f"blk.{i}.ffn_gate_inp.weight",
        ]

    tensor_map = [(name, f"{args.input}/tensor{i:05}_000") for i, name in enumerate(tensor_names)]
    f = gguf.GGUFWriter(args.save_path, "grok", endianess=gguf.GGUFEndian.LITTLE)

    f.add_name("grok")
    f.add_vocab_size(config.vocab_size)
    f.add_context_length(config.max_position_embeddings)
    f.add_embedding_length(config.hidden_size)
    f.add_block_count(config.num_hidden_layers)
    f.add_feed_forward_length(config.intermediate_size)
    f.add_rope_dimension_count(config.hidden_size // config.num_attention_heads)
    f.add_head_count(config.num_attention_heads)
    f.add_head_count_kv(config.num_key_value_heads)

    f.add_expert_count(config.num_experts)
    f.add_expert_used_count(config.num_selected_experts)
    f.add_layer_norm_rms_eps(config.rms_norm_eps)

    f.add_rope_freq_base(config.rope_theta)

    f.add_tokenizer_model("llama")
    # Extract model vocabulary for model conversion
    tokens, scores, toktypes = extract_vocabulary_from_model(vocab)
    f.add_token_list(tokens)
    f.add_token_scores(scores)
    f.add_token_types(toktypes)

    weight_names = get_weight_names(config)
    dump_state_dict(f, weight_names, tensor_map, ggml_type, config)
    f.close()

    delta = time.time() - start

    print(f"grok GGUF model saved to {args.save_path}. Total time {delta:.2f} sec")


def load_vocab(path):
    def load_spm(p):
        print(f"Loading vocab file {p}")
        return SentencePieceVocab(p)

    # Be extra-friendly and accept either a file or a directory.  Also, if it's
    # a directory, it might be the model directory, and tokenizer.model might
    # be in the parent of that.
    if path.is_dir():
        path2 = path / "tokenizer.model"
        # Use `.parent` instead of /.. to handle the symlink case better.
        path3 = path.parent / "tokenizer.model"

        if path2.exists():
            return load_spm(path2)
        elif path3.exists():
            return load_spm(path3)

    raise FileNotFoundError(
        f"Could not find tokenizer.model in {path} or its parent; "
        "if it's in another directory, pass the directory as --vocab-dir"
    )


def main():
    parser = argparse.ArgumentParser("convert_grok")
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--save_path", type=pathlib.Path)
    parser.add_argument(
        "-t", "--type", type=str, default="q8_0", choices=["f32", "f16", "q8_0", "q4_0", "q4_1"]
    )
    parser.add_argument("--vocab_dir", type=str, default="")
    parser.add_argument("--experts", type=str, default="")
    args = parser.parse_args()

    vocab = load_vocab(pathlib.Path(args.vocab_dir) if args.vocab_dir else pathlib.Path(args.input))
    ggml_type = gguf.GGMLQuantizationType[args.type.upper()]
    convert_grok(args, vocab, ggml_type)


if __name__ == "__main__":
    main()
