#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import tempfile
from enum import IntEnum, auto
from io import BufferedWriter
from pathlib import Path
from typing import IO, Any, BinaryIO, Callable, Sequence

import numpy as np

#
# constants
#

GGUF_MAGIC             = 0x46554747
GGUF_VERSION           = 3
GGUF_DEFAULT_ALIGNMENT = 32


# general
KEY_GENERAL_ARCHITECTURE         = "general.architecture"
KEY_GENERAL_QUANTIZATION_VERSION = "general.quantization_version"
KEY_GENERAL_ALIGNMENT            = "general.alignment"
KEY_GENERAL_NAME                 = "general.name"
KEY_GENERAL_AUTHOR               = "general.author"
KEY_GENERAL_URL                  = "general.url"
KEY_GENERAL_DESCRIPTION          = "general.description"
KEY_GENERAL_LICENSE              = "general.license"
KEY_GENERAL_SOURCE_URL           = "general.source.url"
KEY_GENERAL_SOURCE_HF_REPO       = "general.source.huggingface.repository"
KEY_GENERAL_FILE_TYPE            = "general.file_type"

# LLM
KEY_CONTEXT_LENGTH        = "{arch}.context_length"
KEY_EMBEDDING_LENGTH      = "{arch}.embedding_length"
KEY_BLOCK_COUNT           = "{arch}.block_count"
KEY_FEED_FORWARD_LENGTH   = "{arch}.feed_forward_length"
KEY_USE_PARALLEL_RESIDUAL = "{arch}.use_parallel_residual"
KEY_TENSOR_DATA_LAYOUT    = "{arch}.tensor_data_layout"

# attention
KEY_ATTENTION_HEAD_COUNT        = "{arch}.attention.head_count"
KEY_ATTENTION_HEAD_COUNT_KV     = "{arch}.attention.head_count_kv"
KEY_ATTENTION_MAX_ALIBI_BIAS    = "{arch}.attention.max_alibi_bias"
KEY_ATTENTION_CLAMP_KQV         = "{arch}.attention.clamp_kqv"
KEY_ATTENTION_LAYERNORM_EPS     = "{arch}.attention.layer_norm_epsilon"
KEY_ATTENTION_LAYERNORM_RMS_EPS = "{arch}.attention.layer_norm_rms_epsilon"

# RoPE
KEY_ROPE_DIMENSION_COUNT = "{arch}.rope.dimension_count"
KEY_ROPE_FREQ_BASE       = "{arch}.rope.freq_base"
KEY_ROPE_SCALE_LINEAR    = "{arch}.rope.scale_linear"

# tokenization
KEY_TOKENIZER_MODEL      = "tokenizer.ggml.model"
KEY_TOKENIZER_LIST       = "tokenizer.ggml.tokens"
KEY_TOKENIZER_TOKEN_TYPE = "tokenizer.ggml.token_type"
KEY_TOKENIZER_SCORES     = "tokenizer.ggml.scores"
KEY_TOKENIZER_MERGES     = "tokenizer.ggml.merges"
KEY_TOKENIZER_BOS_ID     = "tokenizer.ggml.bos_token_id"
KEY_TOKENIZER_EOS_ID     = "tokenizer.ggml.eos_token_id"
KEY_TOKENIZER_UNK_ID     = "tokenizer.ggml.unknown_token_id"
KEY_TOKENIZER_SEP_ID     = "tokenizer.ggml.seperator_token_id"
KEY_TOKENIZER_PAD_ID     = "tokenizer.ggml.padding_token_id"
KEY_TOKENIZER_HF_JSON    = "tokenizer.huggingface.json"
KEY_TOKENIZER_RWKV       = "tokenizer.rwkv.world"


#
# recommended mapping of model tensor names for storage in gguf
#


class MODEL_ARCH(IntEnum):
    LLAMA         : int = auto()
    FALCON        : int = auto()
    BAICHUAN      : int = auto()
    GPT2          : int = auto()
    GPTJ          : int = auto()
    GPTNEOX       : int = auto()
    MPT           : int = auto()
    STARCODER     : int = auto()
    PERSIMMON     : int = auto()
    REFACT        : int = auto()
    BERT          : int = auto()
    BLOOM         : int = auto()


class MODEL_TENSOR(IntEnum):
    TOKEN_EMBD      : int = auto()
    TOKEN_EMBD_NORM : int = auto()
    TOKEN_TYPES     : int = auto()
    POS_EMBD        : int = auto()
    OUTPUT          : int = auto()
    OUTPUT_NORM     : int = auto()
    ROPE_FREQS      : int = auto()
    ATTN_Q          : int = auto()
    ATTN_K          : int = auto()
    ATTN_V          : int = auto()
    ATTN_QKV        : int = auto()
    ATTN_OUT        : int = auto()
    ATTN_NORM       : int = auto()
    ATTN_NORM_2     : int = auto()
    ATTN_ROT_EMBD   : int = auto()
    FFN_GATE        : int = auto()
    FFN_DOWN        : int = auto()
    FFN_UP          : int = auto()
    FFN_NORM        : int = auto()
    ATTN_Q_NORM     : int = auto()
    ATTN_K_NORM     : int = auto()


MODEL_ARCH_NAMES: dict[MODEL_ARCH, str] = {
    MODEL_ARCH.LLAMA:          "llama",
    MODEL_ARCH.FALCON:         "falcon",
    MODEL_ARCH.BAICHUAN:       "baichuan",
    MODEL_ARCH.GPT2:           "gpt2",
    MODEL_ARCH.GPTJ:           "gptj",
    MODEL_ARCH.GPTNEOX:        "gptneox",
    MODEL_ARCH.MPT:            "mpt",
    MODEL_ARCH.STARCODER:      "starcoder",
    MODEL_ARCH.PERSIMMON:      "persimmon",
    MODEL_ARCH.REFACT:         "refact",
    MODEL_ARCH.BERT:           "bert",
    MODEL_ARCH.BLOOM:          "bloom",
}

TENSOR_NAMES: dict[MODEL_TENSOR, str] = {
    MODEL_TENSOR.TOKEN_EMBD:      "token_embd",
    MODEL_TENSOR.TOKEN_EMBD_NORM: "token_embd_norm",
    MODEL_TENSOR.TOKEN_TYPES:     "token_types",
    MODEL_TENSOR.POS_EMBD:        "position_embd",
    MODEL_TENSOR.OUTPUT_NORM:     "output_norm",
    MODEL_TENSOR.OUTPUT:          "output",
    MODEL_TENSOR.ROPE_FREQS:      "rope_freqs",
    MODEL_TENSOR.ATTN_NORM:       "blk.{bid}.attn_norm",
    MODEL_TENSOR.ATTN_NORM_2:     "blk.{bid}.attn_norm_2",
    MODEL_TENSOR.ATTN_QKV:        "blk.{bid}.attn_qkv",
    MODEL_TENSOR.ATTN_Q:          "blk.{bid}.attn_q",
    MODEL_TENSOR.ATTN_K:          "blk.{bid}.attn_k",
    MODEL_TENSOR.ATTN_V:          "blk.{bid}.attn_v",
    MODEL_TENSOR.ATTN_OUT:        "blk.{bid}.attn_output",
    MODEL_TENSOR.ATTN_ROT_EMBD:   "blk.{bid}.attn_rot_embd",
    MODEL_TENSOR.ATTN_Q_NORM:     "blk.{bid}.attn_q_norm",
    MODEL_TENSOR.ATTN_K_NORM:     "blk.{bid}.attn_k_norm",
    MODEL_TENSOR.FFN_NORM:        "blk.{bid}.ffn_norm",
    MODEL_TENSOR.FFN_GATE:        "blk.{bid}.ffn_gate",
    MODEL_TENSOR.FFN_DOWN:        "blk.{bid}.ffn_down",
    MODEL_TENSOR.FFN_UP:          "blk.{bid}.ffn_up",
}

MODEL_TENSORS: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    MODEL_ARCH.LLAMA: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.GPTNEOX: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.FALCON: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_NORM_2,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BAICHUAN: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.STARCODER: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BERT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_TYPES,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.MPT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.GPTJ: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.PERSIMMON: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.REFACT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BLOOM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.GPT2: [
        # TODO
    ],
    # TODO
}

# tensors that will not be serialized
MODEL_TENSOR_SKIP: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    MODEL_ARCH.LLAMA: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.BAICHUAN: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.PERSIMMON: [
        MODEL_TENSOR.ROPE_FREQS,
    ]
}


class TensorNameMap:
    mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {
        # Token embeddings
        MODEL_TENSOR.TOKEN_EMBD: (
            "gpt_neox.embed_in",                        # gptneox
            "transformer.wte",                          # gpt2 gpt-j mpt refact
            "transformer.word_embeddings",              # falcon
            "word_embeddings",                          # bloom
            "model.embed_tokens",                       # llama-hf
            "tok_embeddings",                           # llama-pth
            "embeddings.word_embeddings",               # bert
            "language_model.embedding.word_embeddings", # persimmon
        ),

        # Token type embeddings
        MODEL_TENSOR.TOKEN_TYPES: (
            "embeddings.token_type_embeddings",  # bert
        ),

        # Normalization of token embeddings
        MODEL_TENSOR.TOKEN_EMBD_NORM: (
            "word_embeddings_layernorm",  # bloom
        ),

        # Position embeddings
        MODEL_TENSOR.POS_EMBD: (
            "transformer.wpe",                 # gpt2
            "embeddings.position_embeddings",  # bert
        ),

        # Output
        MODEL_TENSOR.OUTPUT: (
            "embed_out",                # gptneox
            "lm_head",                  # gpt2 mpt falcon llama-hf baichuan
            "output",                   # llama-pth bloom
            "word_embeddings_for_head", # persimmon
        ),

        # Output norm
        MODEL_TENSOR.OUTPUT_NORM: (
            "gpt_neox.final_layer_norm",              # gptneox
            "transformer.ln_f",                       # gpt2 gpt-j falcon
            "model.norm",                             # llama-hf baichuan
            "norm",                                   # llama-pth
            "embeddings.LayerNorm",                   # bert
            "transformer.norm_f",                     # mpt
            "ln_f",                                   # refact bloom
            "language_model.encoder.final_layernorm", # persimmon
        ),

        # Rope frequencies
        MODEL_TENSOR.ROPE_FREQS: (
            "rope.freqs", # llama-pth
        ),
    }

    block_mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {
        # Attention norm
        MODEL_TENSOR.ATTN_NORM: (
            "gpt_neox.layers.{bid}.input_layernorm",               # gptneox
            "transformer.h.{bid}.ln_1",                            # gpt2 gpt-j refact
            "transformer.blocks.{bid}.norm_1",                     # mpt
            "transformer.h.{bid}.input_layernorm",                 # falcon7b
            "h.{bid}.input_layernorm",                             # bloom
            "transformer.h.{bid}.ln_mlp",                          # falcon40b
            "model.layers.{bid}.input_layernorm",                  # llama-hf
            "layers.{bid}.attention_norm",                         # llama-pth
            "encoder.layer.{bid}.attention.output.LayerNorm",      # bert
            "language_model.encoder.layers.{bid}.input_layernorm", # persimmon
        ),

        # Attention norm 2
        MODEL_TENSOR.ATTN_NORM_2: (
            "transformer.h.{bid}.ln_attn", # falcon40b
        ),

        # Attention query-key-value
        MODEL_TENSOR.ATTN_QKV: (
            "gpt_neox.layers.{bid}.attention.query_key_value",                    # gptneox
            "transformer.h.{bid}.attn.c_attn",                                    # gpt2
            "transformer.blocks.{bid}.attn.Wqkv",                                 # mpt
            "transformer.h.{bid}.self_attention.query_key_value",                 # falcon
            "h.{bid}.self_attention.query_key_value",                             # bloom
            "language_model.encoder.layers.{bid}.self_attention.query_key_value", # persimmon
        ),

        # Attention query
        MODEL_TENSOR.ATTN_Q: (
            "model.layers.{bid}.self_attn.q_proj",       # llama-hf
            "layers.{bid}.attention.wq",                 # llama-pth
            "encoder.layer.{bid}.attention.self.query",  # bert
            "transformer.h.{bid}.attn.q_proj",           # gpt-j
        ),

        # Attention key
        MODEL_TENSOR.ATTN_K: (
            "model.layers.{bid}.self_attn.k_proj",     # llama-hf
            "layers.{bid}.attention.wk",               # llama-pth
            "encoder.layer.{bid}.attention.self.key",  # bert
            "transformer.h.{bid}.attn.k_proj",         # gpt-j
        ),

        # Attention value
        MODEL_TENSOR.ATTN_V: (
            "model.layers.{bid}.self_attn.v_proj",       # llama-hf
            "layers.{bid}.attention.wv",                 # llama-pth
            "encoder.layer.{bid}.attention.self.value",  # bert
            "transformer.h.{bid}.attn.v_proj",           # gpt-j
        ),

        # Attention output
        MODEL_TENSOR.ATTN_OUT: (
            "gpt_neox.layers.{bid}.attention.dense",                   # gptneox
            "transformer.h.{bid}.attn.c_proj",                         # gpt2 refact
            "transformer.blocks.{bid}.attn.out_proj",                  # mpt
            "transformer.h.{bid}.self_attention.dense",                # falcon
            "h.{bid}.self_attention.dense",                            # bloom
            "model.layers.{bid}.self_attn.o_proj",                     # llama-hf
            "layers.{bid}.attention.wo",                               # llama-pth
            "encoder.layer.{bid}.attention.output.dense",              # bert
            "transformer.h.{bid}.attn.out_proj",                       # gpt-j
            "language_model.encoder.layers.{bid}.self_attention.dense" # persimmon
        ),

        # Rotary embeddings
        MODEL_TENSOR.ATTN_ROT_EMBD: (
            "model.layers.{bid}.self_attn.rotary_emb.inv_freq",  # llama-hf
            "layers.{bid}.attention.inner_attention.rope.freqs", # llama-pth
        ),

        # Feed-forward norm
        MODEL_TENSOR.FFN_NORM: (
            "gpt_neox.layers.{bid}.post_attention_layernorm",               # gptneox
            "transformer.h.{bid}.ln_2",                                     # gpt2 refact
            "h.{bid}.post_attention_layernorm",                             # bloom
            "transformer.blocks.{bid}.norm_2",                              # mpt
            "model.layers.{bid}.post_attention_layernorm",                  # llama-hf
            "layers.{bid}.ffn_norm",                                        # llama-pth
            "encoder.layer.{bid}.output.LayerNorm",                         # bert
            "language_model.encoder.layers.{bid}.post_attention_layernorm", # persimmon
        ),

        # Feed-forward up
        MODEL_TENSOR.FFN_UP: (
            "gpt_neox.layers.{bid}.mlp.dense_h_to_4h",               # gptneox
            "transformer.h.{bid}.mlp.c_fc",                          # gpt2
            "transformer.blocks.{bid}.ffn.up_proj",                  # mpt
            "transformer.h.{bid}.mlp.dense_h_to_4h",                 # falcon
            "h.{bid}.mlp.dense_h_to_4h",                             # bloom
            "model.layers.{bid}.mlp.up_proj",                        # llama-hf refact
            "layers.{bid}.feed_forward.w3",                          # llama-pth
            "encoder.layer.{bid}.intermediate.dense",                # bert
            "transformer.h.{bid}.mlp.fc_in",                         # gpt-j
            "language_model.encoder.layers.{bid}.mlp.dense_h_to_4h", # persimmon
        ),

        # Feed-forward gate
        MODEL_TENSOR.FFN_GATE: (
            "model.layers.{bid}.mlp.gate_proj", # llama-hf refact
            "layers.{bid}.feed_forward.w1",     # llama-pth
        ),

        # Feed-forward down
        MODEL_TENSOR.FFN_DOWN: (
            "gpt_neox.layers.{bid}.mlp.dense_4h_to_h",               # gptneox
            "transformer.h.{bid}.mlp.c_proj",                        # gpt2 refact
            "transformer.blocks.{bid}.ffn.down_proj",                # mpt
            "transformer.h.{bid}.mlp.dense_4h_to_h",                 # falcon
            "h.{bid}.mlp.dense_4h_to_h",                             # bloom
            "model.layers.{bid}.mlp.down_proj",                      # llama-hf
            "layers.{bid}.feed_forward.w2",                          # llama-pth
            "encoder.layer.{bid}.output.dense",                      # bert
            "transformer.h.{bid}.mlp.fc_out",                        # gpt-j
            "language_model.encoder.layers.{bid}.mlp.dense_4h_to_h", # persimmon
        ),

        MODEL_TENSOR.ATTN_Q_NORM: (
            "language_model.encoder.layers.{bid}.self_attention.q_layernorm",
        ),

        MODEL_TENSOR.ATTN_K_NORM: (
            "language_model.encoder.layers.{bid}.self_attention.k_layernorm",
        ),

        MODEL_TENSOR.ROPE_FREQS: (
            "language_model.encoder.layers.{bid}.self_attention.rotary_emb.inv_freq", # persimmon
        )
    }

    mapping: dict[str, tuple[MODEL_TENSOR, str]]

    def __init__(self, arch: MODEL_ARCH, n_blocks: int):
        self.mapping = {}
        for tensor, keys in self.mappings_cfg.items():
            if tensor not in MODEL_TENSORS[arch]:
                continue
            tensor_name = TENSOR_NAMES[tensor]
            self.mapping[tensor_name] = (tensor, tensor_name)
            for key in keys:
                self.mapping[key] = (tensor, tensor_name)
        for bid in range(n_blocks):
            for tensor, keys in self.block_mappings_cfg.items():
                if tensor not in MODEL_TENSORS[arch]:
                    continue
                tensor_name = TENSOR_NAMES[tensor].format(bid = bid)
                self.mapping[tensor_name] = (tensor, tensor_name)
                for key in keys:
                    key = key.format(bid = bid)
                    self.mapping[key] = (tensor, tensor_name)

    def get_type_and_name(self, key: str, try_suffixes: Sequence[str] = ()) -> tuple[MODEL_TENSOR, str] | None:
        result = self.mapping.get(key)
        if result is not None:
            return result
        for suffix in try_suffixes:
            if key.endswith(suffix):
                result = self.mapping.get(key[:-len(suffix)])
                if result is not None:
                    return (result[0], result[1] + suffix)
        return None

    def get_name(self, key: str, try_suffixes: Sequence[str] = ()) -> str | None:
        result = self.get_type_and_name(key, try_suffixes = try_suffixes)
        if result is None:
            return None
        return result[1]

    def get_type(self, key: str, try_suffixes: Sequence[str] = ()) -> MODEL_TENSOR | None:
        result = self.get_type_and_name(key, try_suffixes = try_suffixes)
        if result is None:
            return None
        return result[0]

    def __getitem__(self, key: str) -> str:
        try:
            return self.mapping[key][1]
        except KeyError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self.mapping

    def __repr__(self) -> str:
        return repr(self.mapping)

def get_tensor_name_map(arch: MODEL_ARCH, n_blocks: int) -> TensorNameMap:
    return TensorNameMap(arch, n_blocks)

class TokenType(IntEnum):
    NORMAL       = 1
    UNKNOWN      = 2
    CONTROL      = 3
    USER_DEFINED = 4
    UNUSED       = 5
    BYTE         = 6

#
# implementation
#


class GGMLQuantizationType(IntEnum):
    F32  = 0
    F16  = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15

class GGUFEndian(IntEnum):
    LITTLE = 0
    BIG = 1


class GGUFValueType(IntEnum):
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12

    @staticmethod
    def get_type(val):
        if isinstance(val, str) or isinstance(val, bytes) or isinstance(val, bytearray):
            return GGUFValueType.STRING
        elif isinstance(val, list):
            return GGUFValueType.ARRAY
        elif isinstance(val, float):
            return GGUFValueType.FLOAT32
        elif isinstance(val, bool):
            return GGUFValueType.BOOL
        elif isinstance(val, int):
            return GGUFValueType.INT32
        # TODO: need help with 64-bit types in Python
        else:
            print("Unknown type: "+str(type(val)))
            sys.exit()


class GGUFWriter:
    fout: BufferedWriter
    arch: str
    offset_tensor = 0
    data_alignment = GGUF_DEFAULT_ALIGNMENT
    kv_data = b""
    kv_data_count = 0
    ti_data = b""
    ti_data_count = 0
    use_temp_file: bool
    temp_file: tempfile.SpooledTemporaryFile[bytes] | None = None
    tensors: list[tuple[np.ndarray[Any, Any], int]]

    @property
    def pack_prefix(self):
        if self.endianess==GGUFEndian.LITTLE:
            return "<"
        else:
            return ">"

    def __init__(self, path: os.PathLike[str] | str, arch: str, use_temp_file = True, endianess=GGUFEndian.LITTLE):
        self.fout = open(path, "wb")
        self.arch = arch
        self.endianess = endianess
        self._simple_value_packing = {
            GGUFValueType.UINT8:   f"{self.pack_prefix}B",
            GGUFValueType.INT8:    f"{self.pack_prefix}b",
            GGUFValueType.UINT16:  f"{self.pack_prefix}H",
            GGUFValueType.INT16:   f"{self.pack_prefix}h",
            GGUFValueType.UINT32:  f"{self.pack_prefix}I",
            GGUFValueType.INT32:   f"{self.pack_prefix}i",
            GGUFValueType.FLOAT32: f"{self.pack_prefix}f",
            GGUFValueType.UINT64:  f"{self.pack_prefix}Q",
            GGUFValueType.INT64:   f"{self.pack_prefix}q",
            GGUFValueType.FLOAT64: f"{self.pack_prefix}d",
            GGUFValueType.BOOL:    "?" ,
        }
        self.add_architecture()
        self.use_temp_file = use_temp_file
        self.tensors = []
        endianess_str = "Big Endian" if self.endianess == GGUFEndian.BIG else "Little Endian"
        print(f"This gguf file is for {endianess_str} only")

    def write_header_to_file(self):
        self.fout.write(struct.pack("<I", GGUF_MAGIC))
        self.fout.write(struct.pack(f"{self.pack_prefix}I", GGUF_VERSION))
        self.fout.write(struct.pack(f"{self.pack_prefix}Q", self.ti_data_count))
        self.fout.write(struct.pack(f"{self.pack_prefix}Q", self.kv_data_count))
        self.flush()
#        print("tensors " + str(self.ti_data_count) + " kv " + str(self.kv_data_count))

    def write_kv_data_to_file(self):
        self.fout.write(self.kv_data)
        self.flush()

    def write_ti_data_to_file(self):
        self.fout.write(self.ti_data)
        self.flush()

    def add_key(self, key: str):
        self.add_val(key, GGUFValueType.STRING, add_vtype=False)

    def add_uint8(self, key: str, val: int):
        self.add_key(key)
        self.add_val(val, GGUFValueType.UINT8)

    def add_int8(self, key: str, val: int):
        self.add_key(key)
        self.add_val(val, GGUFValueType.INT8)

    def add_uint16(self, key: str, val: int):
        self.add_key(key)
        self.add_val(val, GGUFValueType.UINT16)

    def add_int16(self, key: str, val: int):
        self.add_key(key)
        self.add_val(val, GGUFValueType.INT16)

    def add_uint32(self, key: str, val: int):
        self.add_key(key)
        self.add_val(val, GGUFValueType.UINT32)

    def add_int32(self, key: str, val: int):
        self.add_key(key)
        self.add_val(val, GGUFValueType.INT32)

    def add_float32(self, key: str, val: float):
        self.add_key(key)
        self.add_val(val, GGUFValueType.FLOAT32)

    def add_uint64(self, key: str, val: int):
        self.add_key(key)
        self.add_val(val, GGUFValueType.UINT64)

    def add_int64(self, key: str, val: int):
        self.add_key(key)
        self.add_val(val, GGUFValueType.INT64)

    def add_float64(self, key: str, val: float):
        self.add_key(key)
        self.add_val(val, GGUFValueType.FLOAT64)

    def add_bool(self, key: str, val: bool):
        self.add_key(key)
        self.add_val(val, GGUFValueType.BOOL)

    def add_string(self, key: str, val: str):
        if len(val) == 0:
            return
        self.add_key(key)
        self.add_val(val, GGUFValueType.STRING)

    def add_array(self, key: str, val: Sequence[Any]):
        if not isinstance(val, Sequence):
            raise ValueError("Value must be a sequence for array type")

        self.add_key(key)
        self.add_val(val, GGUFValueType.ARRAY)

    def add_val(self, val: Any, vtype: GGUFValueType | None = None, add_vtype: bool = True):
        if vtype is None:
            vtype = GGUFValueType.get_type(val)

        if add_vtype:
            self.kv_data += struct.pack(f"{self.pack_prefix}I", vtype)
            self.kv_data_count += 1

        pack_fmt = self._simple_value_packing.get(vtype)
        if pack_fmt is not None:
            self.kv_data += struct.pack(pack_fmt, val)
        elif vtype == GGUFValueType.STRING:
            encoded_val = val.encode("utf8") if isinstance(val, str) else val
            self.kv_data += struct.pack(f"{self.pack_prefix}Q", len(encoded_val))
            self.kv_data += encoded_val
        elif vtype == GGUFValueType.ARRAY and isinstance(val, Sequence) and len(val) > 0:
            ltype = GGUFValueType.get_type(val[0])
            if not all(GGUFValueType.get_type(i) is ltype for i in val[1:]):
                raise ValueError("All items in a GGUF array should be of the same type")
            self.kv_data += struct.pack(f"{self.pack_prefix}I", ltype)
            self.kv_data += struct.pack(f"{self.pack_prefix}Q", len(val))
            for item in val:
                self.add_val(item, add_vtype=False)
        else:
            raise ValueError("Invalid GGUF metadata value type or value")

    @staticmethod
    def ggml_pad(x: int, n: int) -> int:
        return ((x + n - 1) // n) * n

    def add_tensor_info(self, name: str, tensor_shape: Sequence[int], tensor_dtype: np.dtype[np.float16] | np.dtype[np.float32], tensor_nbytes: int, raw_dtype: GGMLQuantizationType | None = None):
        assert raw_dtype is not None or tensor_dtype in (np.float32, np.float16), "Only F32 and F16 tensors are supported for now"

        encoded_name = name.encode("utf8")
        self.ti_data += struct.pack(f"{self.pack_prefix}Q", len(encoded_name))
        self.ti_data += encoded_name
        n_dims = len(tensor_shape)
        self.ti_data += struct.pack(f"{self.pack_prefix}I", n_dims)
        for i in range(n_dims):
            self.ti_data += struct.pack(f"{self.pack_prefix}Q", tensor_shape[n_dims - 1 - i])
        if raw_dtype is None:
            dtype = GGMLQuantizationType.F32 if tensor_dtype == np.float32 else GGMLQuantizationType.F16
        else:
            dtype = raw_dtype
        self.ti_data += struct.pack(f"{self.pack_prefix}I", dtype)
        self.ti_data += struct.pack(f"{self.pack_prefix}Q", self.offset_tensor)
        self.offset_tensor += GGUFWriter.ggml_pad(tensor_nbytes, self.data_alignment)
        self.ti_data_count += 1

    def add_tensor(self, name: str, tensor: np.ndarray[Any, Any], raw_shape: Sequence[int] | None = None, raw_dtype: GGMLQuantizationType | None = None):
        if self.endianess == GGUFEndian.BIG:
            tensor.byteswap(inplace=True)
        if self.use_temp_file and self.temp_file is None:
            fp = tempfile.SpooledTemporaryFile(mode="w+b", max_size=256*1024*1024)
            fp.seek(0)
            self.temp_file = fp

        shape: Sequence[int] = raw_shape if raw_shape is not None else tensor.shape
        self.add_tensor_info(name, shape, tensor.dtype, tensor.nbytes, raw_dtype = raw_dtype)

        pad = GGUFWriter.ggml_pad(tensor.nbytes, self.data_alignment) - tensor.nbytes

        if  self.temp_file is None:
            self.tensors.append((tensor, pad))
            return

        tensor.tofile(self.temp_file)

        if pad != 0:
            self.temp_file.write(bytes([0] * pad))

    def write_padding(self, fp: BinaryIO, n: int, align: int | None = None):
        pad = GGUFWriter.ggml_pad(n, align if align is not None else self.data_alignment) - n
        if pad != 0:
            fp.write(bytes([0] * pad))

    def write_tensor_data(self, tensor: np.ndarray[Any, Any]):
        if self.endianess==GGUFEndian.BIG:
            tensor.byteswap(inplace=True)
        self.write_padding(self.fout, self.fout.tell())
        tensor.tofile(self.fout)
        self.write_padding(self.fout, tensor.nbytes)

    def write_tensors_to_file(self):
        self.write_ti_data_to_file()

        self.write_padding(self.fout, self.fout.tell())

        if self.temp_file is None:
            for (currtensor, currpad) in self.tensors:
                currtensor.tofile(self.fout)
                if currpad != 0:
                    self.fout.write(bytes([0] * currpad))
            return

        self.temp_file.seek(0)

        shutil.copyfileobj(self.temp_file, self.fout)
        self.flush()
        self.temp_file.close()

    def flush(self):
        self.fout.flush()

    def close(self):
        self.fout.close()

    def add_architecture(self):
        self.add_string(KEY_GENERAL_ARCHITECTURE, self.arch)

    def add_author(self, author: str):
        self.add_string(KEY_GENERAL_AUTHOR, author)

    def add_tensor_data_layout(self, layout: str):
        self.add_string(KEY_TENSOR_DATA_LAYOUT.format(arch=self.arch), layout)

    def add_url(self, url: str):
        self.add_string(KEY_GENERAL_URL, url)

    def add_description(self, description: str):
        self.add_string(KEY_GENERAL_DESCRIPTION, description)

    def add_source_url(self, url: str):
        self.add_string(KEY_GENERAL_SOURCE_URL, url)

    def add_source_hf_repo(self, repo: str):
        self.add_string(KEY_GENERAL_SOURCE_HF_REPO, repo)

    def add_file_type(self, ftype: int):
        self.add_uint32(KEY_GENERAL_FILE_TYPE, ftype)

    def add_name(self, name: str):
        self.add_string(KEY_GENERAL_NAME, name)

    def add_quantization_version(self, quantization_version: GGMLQuantizationType):
        self.add_uint32(
            KEY_GENERAL_QUANTIZATION_VERSION, quantization_version)

    def add_custom_alignment(self, alignment: int):
        self.data_alignment = alignment
        self.add_uint32(KEY_GENERAL_ALIGNMENT, alignment)

    def add_context_length(self, length: int):
        self.add_uint32(
            KEY_CONTEXT_LENGTH.format(arch=self.arch), length)

    def add_embedding_length(self, length: int):
        self.add_uint32(
            KEY_EMBEDDING_LENGTH.format(arch=self.arch), length)

    def add_block_count(self, length: int):
        self.add_uint32(
            KEY_BLOCK_COUNT.format(arch=self.arch), length)

    def add_feed_forward_length(self, length: int):
        self.add_uint32(
            KEY_FEED_FORWARD_LENGTH.format(arch=self.arch), length)

    def add_parallel_residual(self, use: bool):
        self.add_bool(
            KEY_USE_PARALLEL_RESIDUAL.format(arch=self.arch), use)

    def add_head_count(self, count: int):
        self.add_uint32(
            KEY_ATTENTION_HEAD_COUNT.format(arch=self.arch), count)

    def add_head_count_kv(self, count: int):
        self.add_uint32(
            KEY_ATTENTION_HEAD_COUNT_KV.format(arch=self.arch), count)

    def add_max_alibi_bias(self, bias: float):
        self.add_float32(
            KEY_ATTENTION_MAX_ALIBI_BIAS.format(arch=self.arch), bias)

    def add_clamp_kqv(self, value: float):
        self.add_float32(
            KEY_ATTENTION_CLAMP_KQV.format(arch=self.arch), value)

    def add_layer_norm_eps(self, value: float):
        self.add_float32(
            KEY_ATTENTION_LAYERNORM_EPS.format(arch=self.arch), value)

    def add_layer_norm_rms_eps(self, value: float):
        self.add_float32(
            KEY_ATTENTION_LAYERNORM_RMS_EPS.format(arch=self.arch), value)

    def add_rope_dimension_count(self, count: int):
        self.add_uint32(
            KEY_ROPE_DIMENSION_COUNT.format(arch=self.arch), count)

    def add_rope_freq_base(self, value: float):
        self.add_float32(KEY_ROPE_FREQ_BASE.format(arch=self.arch), value)

    def add_rope_scale_linear(self, value: float):
        self.add_float32(KEY_ROPE_SCALE_LINEAR.format(arch=self.arch), value)

    def add_tokenizer_model(self, model: str):
        self.add_string(KEY_TOKENIZER_MODEL, model)

    def add_token_list(self, tokens: Sequence[str] | Sequence[bytes] | Sequence[bytearray]):
        self.add_array(KEY_TOKENIZER_LIST, tokens)

    def add_token_merges(self, merges: Sequence[str] | Sequence[bytes] | Sequence[bytearray]):
        self.add_array(KEY_TOKENIZER_MERGES, merges)

    def add_token_types(self, types: Sequence[TokenType] | Sequence[int]):
        self.add_array(KEY_TOKENIZER_TOKEN_TYPE, types)

    def add_token_scores(self, scores: Sequence[float]):
        self.add_array(KEY_TOKENIZER_SCORES, scores)

    def add_bos_token_id(self, id: int):
        self.add_uint32(KEY_TOKENIZER_BOS_ID, id)

    def add_eos_token_id(self, id: int):
        self.add_uint32(KEY_TOKENIZER_EOS_ID, id)

    def add_unk_token_id(self, id: int):
        self.add_uint32(KEY_TOKENIZER_UNK_ID, id)

    def add_sep_token_id(self, id: int):
        self.add_uint32(KEY_TOKENIZER_SEP_ID, id)

    def add_pad_token_id(self, id: int):
        self.add_uint32(KEY_TOKENIZER_PAD_ID, id)


class SpecialVocab:
    load_merges: bool = False
    merges: list[str] = []
    special_token_types: tuple[str, ...] = ('bos', 'eos', 'unk', 'sep', 'pad')
    special_token_ids: dict[str, int] = {}
    n_vocab: int | None = None

    def __init__(
        self, path: str | os.PathLike[str], load_merges: bool = False,
        special_token_types: tuple[str, ...] | None = None,
        n_vocab: int | None = None,
    ):
        self.special_token_ids = {}
        self.n_vocab = n_vocab
        self.load_merges = load_merges
        if special_token_types is not None:
            self.special_token_types = special_token_types
        self._load(Path(path))

    def _load(self, path: Path) -> None:
        if not self._try_load_from_tokenizer_json(path):
            self._try_load_from_config_json(path)

    def _set_special_token(self, typ: str, tid: Any):
        if not isinstance(tid, int) or tid < 0:
            return
        if self.n_vocab is None or tid < self.n_vocab:
            self.special_token_ids[typ] = tid
            return
        print(f'gguf: WARNING: Special token type {typ}, id {tid} out of range, must be under {self.n_vocab} - skipping',
            file = sys.stderr)


    def _try_load_from_tokenizer_json(self, path: Path) -> bool:
        tokenizer_file = path / 'tokenizer.json'
        if not tokenizer_file.is_file():
            return False
        with open(tokenizer_file, encoding = 'utf-8') as f:
            tokenizer = json.load(f)
        if self.load_merges:
            merges = tokenizer.get('model', {}).get('merges')
            if isinstance(merges, list) and len(merges) > 0 and isinstance(merges[0], str):
                self.merges = merges
        tokenizer_config_file = path / 'tokenizer_config.json'
        added_tokens = tokenizer.get('added_tokens')
        if added_tokens is None or not tokenizer_config_file.is_file():
            return True
        with open(tokenizer_config_file, encoding = 'utf-8') as f:
            tokenizer_config = json.load(f)
        for typ in self.special_token_types:
            entry = tokenizer_config.get(f'{typ}_token')
            if isinstance(entry, str):
                tc_content = entry
            elif isinstance(entry, dict):
                entry_content = entry.get('content')
                if not isinstance(entry_content, str):
                    continue
                tc_content = entry_content
            else:
                continue
            # We only need the first match here.
            maybe_token_id = next((
                atok.get('id') for atok in added_tokens
                if atok.get('content') == tc_content), None)
            self._set_special_token(typ, maybe_token_id)
        return True

    def _try_load_from_config_json(self, path: Path) -> bool:
        config_file = path / 'config.json'
        if not config_file.is_file():
            return False
        with open(config_file, encoding = 'utf-8') as f:
            config = json.load(f)
        for typ in self.special_token_types:
            self._set_special_token(typ, config.get(f'{typ}_token_id'))
        return True

    def add_to_gguf(self, gw: GGUFWriter, quiet: bool = False) -> None:
        if len(self.merges) > 0:
            if not quiet:
                print(f'gguf: Adding {len(self.merges)} merge(s).')
            gw.add_token_merges(self.merges)
        for typ, tokid in self.special_token_ids.items():
            handler: Callable[[int], None] | None = getattr(gw, f'add_{typ}_token_id', None)
            if handler is None:
                print(f'gguf: WARNING: No handler for special token type {typ} with id {tokid} - skipping', file = sys.stderr)
                continue
            if not quiet:
                print(f'gguf: Setting special token type {typ} to {tokid}')
            handler(tokid)

    def __repr__(self) -> str:
        return f'<SpecialVocab with {len(self.merges)} merges and special tokens {self.special_token_ids or "unset"}>'


# Example usage:
if __name__ == "__main__":
    # Example usage with a file
    gguf_writer = GGUFWriter("example.gguf", "llama")

    gguf_writer.add_architecture()
    gguf_writer.add_block_count(12)
    gguf_writer.add_uint32("answer", 42)  # Write a 32-bit integer
    gguf_writer.add_float32("answer_in_float", 42.0)  # Write a 32-bit float
    gguf_writer.add_custom_alignment(64)

    tensor1 = np.ones((32,), dtype=np.float32) * 100.0
    tensor2 = np.ones((64,), dtype=np.float32) * 101.0
    tensor3 = np.ones((96,), dtype=np.float32) * 102.0

    gguf_writer.add_tensor("tensor1", tensor1)
    gguf_writer.add_tensor("tensor2", tensor2)
    gguf_writer.add_tensor("tensor3", tensor3)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()
