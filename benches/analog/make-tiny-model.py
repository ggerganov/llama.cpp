#!/usr/bin/env python3
# Builds a tiny random llama model with a byte-level vocabulary, used to compare the analog
# backend against the CPU backend end to end (loading, evaluation, perplexity).
#
# The weights are random, so the model produces nonsense - only its behaviour matters here.
#
# Usage:
#   python3 benches/analog/make-tiny-model.py --out /tmp/tiny-analog.gguf
#
# The vocabulary consists of the three SPM special tokens and the 256 byte tokens, so that any
# input text can be tokenized without a trained vocabulary.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))

from gguf import GGUFWriter, GGMLQuantizationType, TokenType  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="build a tiny random llama model for backend testing")
    parser.add_argument("--out", type=Path, default=Path("tiny-analog.gguf"))
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-ff", type=int, default=128)
    parser.add_argument("--n-ctx", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--type", default="f32", choices=["f32", "f16", "q8_0", "q4_0"])
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    n_embd  = args.n_embd
    n_layer = args.n_layer
    n_head  = args.n_head
    n_ff    = args.n_ff

    assert n_embd % n_head == 0

    # vocabulary: <unk>, <s>, </s> and the 256 byte tokens
    tokens: list[str] = ["<unk>", "<s>", "</s>"]
    types:  list[int] = [TokenType.UNKNOWN, TokenType.CONTROL, TokenType.CONTROL]

    for i in range(256):
        tokens.append(f"<0x{i:02X}>")
        types.append(TokenType.BYTE)

    n_vocab = len(tokens)

    writer = GGUFWriter(args.out, "llama")

    writer.add_name("tiny analog backend test")
    writer.add_context_length(args.n_ctx)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_dimension_count(n_embd // n_head)
    writer.add_file_type(0)

    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_scores([0.0]*n_vocab)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)

    def add(name: str, shape: tuple[int, ...], scale: float) -> None:
        # the shapes are given in ggml order (ne[0] first), while the GGUF writer stores them
        # reversed with respect to the array shape
        tensor = (rng.standard_normal(shape[::-1])*scale).astype(np.float32)

        if args.type == "f16":
            tensor = tensor.astype(np.float16)
            dtype = GGMLQuantizationType.F16
        elif args.type == "f32":
            dtype = GGMLQuantizationType.F32
        else:
            # the quantized types are produced by the quantization tool
            dtype = GGMLQuantizationType.F32

        writer.add_tensor(name, tensor, raw_dtype=dtype)

    add("token_embd.weight", (n_embd, n_vocab), 0.02)
    add("output_norm.weight", (n_embd,), 0.1)
    add("output.weight",      (n_embd, n_vocab), 0.02)

    for i in range(n_layer):
        add(f"blk.{i}.attn_norm.weight", (n_embd,), 0.1)
        add(f"blk.{i}.attn_q.weight",    (n_embd, n_embd), 0.02)
        add(f"blk.{i}.attn_k.weight",    (n_embd, n_embd), 0.02)
        add(f"blk.{i}.attn_v.weight",    (n_embd, n_embd), 0.02)
        add(f"blk.{i}.attn_output.weight", (n_embd, n_embd), 0.02)
        add(f"blk.{i}.ffn_norm.weight",  (n_embd,), 0.1)
        add(f"blk.{i}.ffn_gate.weight",  (n_embd, n_ff), 0.02)
        add(f"blk.{i}.ffn_down.weight",  (n_ff, n_embd), 0.02)
        add(f"blk.{i}.ffn_up.weight",    (n_embd, n_ff), 0.02)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"wrote {args.out} (n_embd {n_embd}, n_layer {n_layer}, n_ff {n_ff}, vocab {n_vocab}, {args.type})")


if __name__ == "__main__":
    main()
