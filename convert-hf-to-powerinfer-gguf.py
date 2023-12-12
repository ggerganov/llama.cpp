#!/usr/bin/env python3

from __future__ import annotations
from abc import ABC, abstractmethod

import argparse
import contextlib
import json
import os
import re
import struct
import sys
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ContextManager, Iterator, Optional, cast

import numpy as np
import torch
import torch.nn as tnn

if TYPE_CHECKING:
    from torch import Tensor

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / "gguf-py"))
import gguf


###### MODEL DEFINITIONS ######


class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


class ReluMLP(tnn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ReluMLP, self).__init__()
        self.fc1 = tnn.Linear(input_dim, hidden_dim, bias=False)
        self.relu = tnn.ReLU()
        self.fc2 = tnn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def from_file(model_file: Path):
        model = torch.load(model_file, map_location="cpu")
        hidden_size, input_size = model.get("fc1.weight").shape
        output_size, _ = model.get("fc2.weight").shape
        mlp = ReluMLP(input_size, hidden_size, output_size)
        mlp.load_state_dict(model)
        return mlp


class Model(ABC):
    """Base class for model conversion"""

    def __init__(
        self,
        dir_model: Path,
        dir_mlp_pred: Path,
        ftype: int,
        fname_out: Path,
        is_big_endian: bool,
    ):
        self.dir_model = dir_model
        self.dir_mlp_pred = dir_mlp_pred
        self.ftype = ftype
        self.fname_out = fname_out
        self.is_big_endian = is_big_endian
        self.endianess = (
            gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        )
        self.is_safetensors = self._is_model_safetensors()
        self.num_parts = Model.count_model_parts(
            self.dir_model, ".safetensors" if self.is_safetensors else ".bin"
        )
        self.part_names = self._get_part_names()
        self.hparams = Model.load_hparams(self.dir_model)
        self.model_arch = self._get_model_architecture()
        self.gguf_writer = gguf.GGUFWriter(
            fname_out, gguf.MODEL_ARCH_NAMES[self.model_arch], endianess=self.endianess, use_temp_file = False
        )

    def set_vocab(self):
        self._set_vocab_gpt2()

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        for model_layer, part_name in self._get_mlp_part_layer_names():
            print(f"gguf: loading mlp part '{part_name}'")
            mlp_model = ReluMLP.from_file(self.dir_mlp_pred / part_name)
            for name, data in mlp_model.state_dict().items():
                yield f"blk.{model_layer}.{name}", data

        for part_name in self.part_names:
            print(f"gguf: loading model part '{part_name}'")
            ctx: ContextManager[Any]
            if self.is_safetensors:
                from safetensors import safe_open

                ctx = cast(
                    ContextManager[Any],
                    safe_open(self.dir_model / part_name, framework="pt", device="cpu"),
                )
            else:
                ctx = contextlib.nullcontext(
                    torch.load(self.dir_model / part_name, map_location="cpu")
                )

            with ctx as model_part:
                for name in model_part.keys():
                    data = (
                        model_part.get_tensor(name)
                        if self.is_safetensors
                        else model_part[name]
                    )
                    yield name, data

    @abstractmethod
    def set_gguf_parameters(self):
        pass
        # self.gguf_writer.add_name(self.dir_model.name)
        # self.gguf_writer.add_block_count(
        #     self.hparams.get(
        #         "n_layers",
        #         self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")),
        #     )
        # )
        # if (n_ctx := self.hparams.get("max_position_embeddings")) is not None:
        #     self.gguf_writer.add_context_length(n_ctx)
        # if (n_embd := self.hparams.get("hidden_size")) is not None:
        #     self.gguf_writer.add_embedding_length(n_embd)
        # if (n_ff := self.hparams.get("intermediate_size")) is not None:
        #     self.gguf_writer.add_feed_forward_length(n_ff)
        # if (n_head := self.hparams.get("num_attention_head")) is not None:
        #     self.gguf_writer.add_head_count(n_head)
        # self.gguf_writer.add_parallel_residual(
        #     self.hparams.get("use_parallel_residual", True)
        # )

    @abstractmethod
    def write_tensors(self):
        pass

    def write(self):
        self.write_tensors()
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file()
        self.gguf_writer.close()

    def write_vocab(self):
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.close()

    @staticmethod
    def count_model_parts(dir_model: Path, prefix: str) -> int:
        num_parts = 0
        for filename in os.listdir(dir_model):
            if filename.endswith(prefix):
                num_parts += 1

        return num_parts

    @staticmethod
    def load_hparams(dir_model):
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def from_model_architecture(model_architecture):
        if model_architecture in ("FalconForCausalLM", "RWForCausalLM"):
            return FalconModel
        if model_architecture == "LlamaForCausalLM":
            return LlamaModel

        raise NotImplementedError(f'Architecture "{model_architecture}" not supported!')

    def _is_model_safetensors(self) -> bool:
        return Model.count_model_parts(self.dir_model, ".safetensors") > 0

    def _get_mlp_part_layer_names(self):
        """Returns a generator of (index, name) for MLP predictors of each model layer"""
        n_mlp_parts = Model.count_model_parts(self.dir_mlp_pred, ".pt")
        return ((n, f"model_{n}.pt") for n in range(n_mlp_parts))

    def _get_part_names(self):
        if self.is_safetensors:
            if self.num_parts == 1:  # there's only one .safetensors file
                return ("model.safetensors",)
            return (
                f"model-{n:05}-of-{self.num_parts:05}.safetensors"
                for n in range(1, self.num_parts + 1)
            )

        if self.num_parts == 1:  # there's only one .bin file
            return ("pytorch_model.bin",)
        return (
            f"pytorch_model-{n:05}-of-{self.num_parts:05}.bin"
            for n in range(1, self.num_parts + 1)
        )

    def _get_model_architecture(self) -> gguf.MODEL_ARCH:
        arch = self.hparams["architectures"][0]
        if arch == "FalconForCausalLM":
            return gguf.MODEL_ARCH.FALCON
        if arch == "RWForCausalLM" or arch == "LlamaForCausalLM":
            return gguf.MODEL_ARCH.LLAMA

        raise NotImplementedError(f'Architecture "{arch}" not supported!')

    def _translate_tensor_key(
        self, key: str, try_suffixes=(".weight", ".bias")
    ) -> Optional[str]:
        block_count = self.hparams.get(
            "n_layers",
            self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")),
        )
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        arch_tensor_key = tensor_map.get_name(key, try_suffixes=try_suffixes)
        if arch_tensor_key is not None:
            return arch_tensor_key
        # check and handle ReluMLP layers
        mlp_match = re.match(r"^blk\.\d+\.fc\d\.weight$", key)
        if mlp_match:
            return mlp_match.group(0)
        return None

    def _set_vocab_gpt2(self):
        dir_model = self.dir_model
        hparams = self.hparams
        tokens: list[bytearray] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer  # type: ignore[attr-defined]

        tokenizer = AutoTokenizer.from_pretrained(dir_model)
        vocab_size = hparams.get("vocab_size", len(tokenizer.vocab))
        assert max(tokenizer.vocab.values()) < vocab_size

        reverse_vocab = {
            id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()
        }
        added_vocab = tokenizer.get_added_vocab()

        for i in range(vocab_size):
            if i not in reverse_vocab:
                pad_token = f"[PAD{i}]".encode("utf-8")
                tokens.append(bytearray(pad_token))
                toktypes.append(gguf.TokenType.USER_DEFINED)
            elif reverse_vocab[i] in added_vocab:
                tokens.append(reverse_vocab[i])
                if tokenizer.added_tokens_decoder[i].special:
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_sentencepiece(self):
        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model / "tokenizer.model"

        tokens: list[bytes] = []
        scores: list[float] = []
        toktypes: list[int] = []

        if not tokenizer_path.is_file():
            print(f"Error: Missing {tokenizer_path}", file=sys.stderr)
            sys.exit(1)

        tokenizer = SentencePieceProcessor(str(tokenizer_path))
        vocab_size = self.hparams.get("vocab_size", tokenizer.vocab_size())

        for token_id in range(vocab_size):
            piece = tokenizer.id_to_piece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.get_score(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.is_unknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.is_control(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.is_unused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.is_byte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        added_tokens_file = self.dir_model / "added_tokens.json"
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)

                for key in added_tokens_json:
                    tokens.append(key.encode("utf-8"))
                    scores.append(-1000.0)
                    toktypes.append(SentencePieceTokenTypes.USER_DEFINED)

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)


class LlamaModel(Model):
    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        self.gguf_writer.add_name("Llama")
        self.gguf_writer.add_context_length(2048)  # not in config.json
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(self.hparams["num_hidden_layers"])
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(
            self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        )
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_rope_freq_base(self.hparams["rope_theta"])
        self.gguf_writer.add_file_type(self.ftype)

    def write_tensors(self):
        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith(
                (
                    ".attention.masked_bias",
                    ".attention.bias",
                    ".attention.rotary_emb.inv_freq",
                )
            ):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = self._translate_tensor_key(name)
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            # We need to transpose the weight matrices for the FFN Down layers to support the
            # Axpy operation in PowerInfer. So we don't need to transpose them at runtime.
            if "ffn_down" in new_name:
                new_name = new_name.replace("ffn_down", "ffn_down_t")
                data = data.T

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if (
                self.ftype == 1
                and data_dtype == np.float32
                and name.endswith(".weight")
                and n_dims == 2
            ):
                data = data.astype(np.float16)

            print(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)


class FalconModel(Model):
    def set_gguf_parameters(self):
        block_count = self.hparams.get("num_hidden_layers")
        if block_count is None:
            block_count = self.hparams["n_layer"]  # old name

        n_head = self.hparams.get("num_attention_heads")
        if n_head is None:
            n_head = self.hparams["n_head"]  # old name

        n_head_kv = self.hparams.get("num_kv_heads")
        if n_head_kv is None:
            n_head_kv = self.hparams.get("n_head_kv", 1)  # old name

        self.gguf_writer.add_name("Falcon")
        self.gguf_writer.add_context_length(2048)  # not in config.json
        self.gguf_writer.add_tensor_data_layout("jploski")  # qkv tensor transform
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head_kv)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def write_tensors(self):
        n_head = self.hparams.get("num_attention_heads")
        if n_head is None:
            n_head = self.hparams["n_head"]  # old name

        n_head_kv = self.hparams.get("num_kv_heads")
        if n_head_kv is None:
            n_head_kv = self.hparams.get("n_head_kv", 1)  # old name

        head_dim = self.hparams["hidden_size"] // n_head

        for name, data_torch in self.get_tensors():
            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            # QKV tensor transform
            # The original query_key_value tensor contains n_head_kv "kv groups",
            # each consisting of n_head/n_head_kv query weights followed by one key
            # and one value weight (shared by all query heads in the kv group).
            # This layout makes it a big pain to work with in GGML.
            # So we rearrange them here,, so that we have n_head query weights
            # followed by n_head_kv key weights followed by n_head_kv value weights,
            # in contiguous fashion.
            # ref: https://github.com/jploski/ggml/blob/falcon40b/examples/falcon/convert-hf-to-ggml.py

            if "query_key_value" in name:
                qkv = data_torch.view(
                    n_head_kv, n_head // n_head_kv + 2, head_dim, head_dim * n_head
                )
                q = qkv[:, :-2].reshape(n_head * head_dim, head_dim * n_head)
                k = qkv[:, [-2]].reshape(n_head_kv * head_dim, head_dim * n_head)
                v = qkv[:, [-1]].reshape(n_head_kv * head_dim, head_dim * n_head)
                data_torch = torch.cat((q, k, v)).reshape_as(data_torch)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = self._translate_tensor_key(name)
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            # We need to transpose the weight matrices for the FFN Down layers to support the
            # Axpy operation in PowerInfer. So we don't need to transpose them at runtime.
            if "ffn_down" in new_name:
                new_name = new_name.replace("ffn_down", "ffn_down_t")
                data = data.T

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if (
                self.ftype == 1
                and data_dtype == np.float32
                and name.endswith(".weight")
                and n_dims == 2
            ):
                data = data.astype(np.float16)

            print(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)


###### CONVERSION LOGIC ######


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a huggingface model to a GGML compatible file"
    )
    parser.add_argument(
        "--vocab-only",
        action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        help="path to write to; default: based on input",
    )
    parser.add_argument(
        "--outtype",
        type=str,
        choices=["f32", "f16"],
        default="f16",
        help="output format - use f32 for float32, f16 for float16",
    )
    parser.add_argument(
        "--bigendian",
        action="store_true",
        help="model is executed on big endian machine",
    )
    parser.add_argument(
        "model",
        type=Path,
        help="directory containing model file",
    )
    parser.add_argument(
        "mlp_predictors",
        type=Path,
        help="directory containing MLP predictors for model",
    )

    return parser.parse_args()


args = parse_args()

dir_model = args.model
dir_mlp_pred = args.mlp_predictors
if not dir_model.is_dir():
    print(f"Error: {args.model} is not a directory", file=sys.stderr)
    sys.exit(1)
if not dir_mlp_pred.is_dir():
    print(f"Error: {args.mlp_predictors} is not a directory", file=sys.stderr)
    sys.exit(1)

ftype_map = {
    "f32": gguf.GGMLQuantizationType.F32,
    "f16": gguf.GGMLQuantizationType.F16,
}

if args.outfile is not None:
    fname_out = args.outfile
else:
    # output in the same directory as the model by default
    fname_out = dir_model / f"ggml-model-{args.outtype}.gguf"

print(f"Loading model: {dir_model.name}")

hparams = Model.load_hparams(dir_model)

model_class = Model.from_model_architecture(hparams["architectures"][0])
model_instance = model_class(
    dir_model, dir_mlp_pred, ftype_map[args.outtype], fname_out, args.bigendian
)

print("Set model parameters")
model_instance.set_gguf_parameters()

print("Set model tokenizer")
model_instance.set_vocab()

if args.vocab_only:
    print(f"Exporting model vocab to '{fname_out}'")
    model_instance.write_vocab()
else:
    print(f"Exporting model to '{fname_out}'")
    model_instance.write()

# post-process: write another unique file header to distinguish from the origianl GGUF file
with open(fname_out, "r+b") as fout:
    POWERINFER_MAGIC = int.from_bytes(b"PWRI", "little")
    fout.write(struct.pack("<I", POWERINFER_MAGIC))

print(f"Model successfully exported to '{fname_out}'")
