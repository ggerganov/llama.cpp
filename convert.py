#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import enum
import faulthandler
import functools
import itertools
import json
import math
import mmap
import pickle
import re
import signal
import struct
import sys
import time
import zipfile
from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Iterable, Literal, TypeVar

import numpy as np
from sentencepiece import SentencePieceProcessor

import os
if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

if TYPE_CHECKING:
    from typing import TypeAlias

if hasattr(faulthandler, 'register') and hasattr(signal, 'SIGUSR1'):
    faulthandler.register(signal.SIGUSR1)

NDArray: TypeAlias = 'np.ndarray[Any, Any]'

ARCH = gguf.MODEL_ARCH.LLAMA

DEFAULT_CONCURRENCY = 8
#
# data types
#

@dataclass(frozen=True)
class DataType:
    name: str
    dtype: np.dtype[Any]
    valid_conversions: list[str]

    def elements_to_bytes(self, n_elements: int) -> int:
        return n_elements * self.dtype.itemsize

@dataclass(frozen=True)
class UnquantizedDataType(DataType):
    pass

DT_F16  = UnquantizedDataType('F16', dtype = np.dtype(np.float16), valid_conversions = ['F32', 'Q8_0'])
DT_F32  = UnquantizedDataType('F32', dtype = np.dtype(np.float32), valid_conversions = ['F16', 'Q8_0'])
DT_I32  = UnquantizedDataType('I32', dtype = np.dtype(np.int16), valid_conversions = [])
DT_BF16 = UnquantizedDataType('BF16', dtype = np.dtype(np.uint16), valid_conversions = ['F32', 'F16', 'Q8_0'])

@dataclass(frozen=True)
class QuantizedDataType(DataType):
    block_size: int
    quantized_dtype: np.dtype[Any]
    ggml_type: gguf.GGMLQuantizationType

    def quantize(self, arr: NDArray) -> NDArray:
        raise NotImplementedError(f'Quantization for {self.name} not implemented')

    def elements_to_bytes(self, n_elements: int) -> int:
        assert n_elements % self.block_size == 0, f'Invalid number of elements {n_elements} for {self.name} with block size {self.block_size}'
        return self.quantized_dtype.itemsize * (n_elements // self.block_size)

@dataclass(frozen=True)
class Q8_0QuantizedDataType(QuantizedDataType):
    # Mini Q8_0 quantization in Python!
    def quantize(self, arr: NDArray) -> NDArray:
        assert arr.size % self.block_size == 0 and arr.size != 0, f'Bad array size {arr.size}'
        assert arr.dtype == np.float32, f'Bad array type {arr.dtype}'
        n_blocks = arr.size // self.block_size
        blocks = arr.reshape((n_blocks, self.block_size))
        # Much faster implementation of block quantization contributed by @Cebtenzzre
        def quantize_blocks_q8_0(blocks: NDArray) -> Iterable[tuple[Any, Any]]:
            d = abs(blocks).max(axis = 1) / np.float32(127)
            with np.errstate(divide = 'ignore'):
                qs = (blocks / d[:, None]).round()
            qs[d == 0] = 0
            yield from zip(d, qs)
        return np.fromiter(quantize_blocks_q8_0(blocks), count = n_blocks, dtype = self.quantized_dtype)

DT_Q8_0 = Q8_0QuantizedDataType('Q8_0',
    dtype = np.dtype(np.float32), valid_conversions = [],
    ggml_type = gguf.GGMLQuantizationType.Q8_0, block_size = 32,
    quantized_dtype = np.dtype([('d', '<f2'), ('qs', 'i1', (32,))]))

# Quantized types skipped here because they may also map to np.float32
NUMPY_TYPE_TO_DATA_TYPE: dict[np.dtype[Any], DataType] = {}
for dt in (DT_BF16, DT_F16, DT_F32, DT_I32):
    if dt.dtype in NUMPY_TYPE_TO_DATA_TYPE:
        raise ValueError(f'Invalid duplicate data type {dt}')
    NUMPY_TYPE_TO_DATA_TYPE[dt.dtype] = dt

SAFETENSORS_DATA_TYPES: dict[str, DataType] = {
    'BF16': DT_BF16,
    'F16': DT_F16,
    'F32': DT_F32,
    'I32': DT_I32,
}

# TODO: match this with `llama_ftype`
# TODO: rename to LLAMAFileType
# TODO: move to `gguf.py`
class GGMLFileType(enum.IntEnum):
    AllF32     = 0
    MostlyF16  = 1  # except 1d tensors
    MostlyQ8_0 = 7  # except 1d tensors

    def type_for_tensor(self, name: str, tensor: LazyTensor) -> DataType:
        dt = GGML_FILE_TYPE_TO_DATA_TYPE.get(self)
        if dt is None:
            raise ValueError(self)
        # 1D tensors are always F32.
        return dt if len(tensor.shape) > 1 else DT_F32

GGML_FILE_TYPE_TO_DATA_TYPE: dict[GGMLFileType, DataType] = {
    GGMLFileType.AllF32    : DT_F32,
    GGMLFileType.MostlyF16 : DT_F16,
    GGMLFileType.MostlyQ8_0: DT_Q8_0,
}

#
# hparams loading
#

@dataclass
class PredictorParams:
    sparse_threshold: float | None = None

    @staticmethod
    def loadPredictorJson(model: LazyModel, config_path: Path) -> PredictorParams:
        config = json.load(open(config_path))
        return PredictorParams(
            sparse_threshold = config.get("sparse_threshold"),
        )

    @staticmethod
    def load(model_plus: ModelPlus) -> PredictorParams:
        config_path   = model_plus.paths[0].parent / "config.json"

        if config_path.exists():
            params = PredictorParams.loadPredictorJson(model_plus.model, config_path)
        else:
            params = PredictorParams()

        return params

@dataclass
class Params:
    n_vocab:    int
    n_embd:     int
    n_layer:    int
    n_ctx:      int
    n_ff:       int
    n_head:     int
    n_head_kv:  int
    f_norm_eps: float

    rope_scaling_type: gguf.RopeScalingType | None = None
    f_rope_freq_base: float | None = None
    f_rope_scale: float | None = None
    n_orig_ctx: int | None = None
    rope_finetuned: bool | None = None

    ftype: GGMLFileType | None = None

    # path to the directory containing the model files
    path_model: Path | None = None

    # MLP predictor parameters
    predictor_params: PredictorParams = dataclasses.field(default_factory=PredictorParams)

    @staticmethod
    def guessed(model: LazyModel) -> Params:
        # try transformer naming first
        n_vocab, n_embd = model["model.embed_tokens.weight"].shape if "model.embed_tokens.weight" in model else model["tok_embeddings.weight"].shape

        # try transformer naming first
        if "model.layers.0.self_attn.q_proj.weight" in model:
            n_layer=next(i for i in itertools.count() if f"model.layers.{i}.self_attn.q_proj.weight" not in model)
        elif "model.layers.0.self_attn.W_pack.weight" in model:   # next: try baichuan naming
            n_layer=next(i for i in itertools.count() if f"model.layers.{i}.self_attn.W_pack.weight" not in model)
        else:
            n_layer=next(i for i in itertools.count() if f"layers.{i}.attention.wq.weight" not in model)

        if n_layer < 1:
            raise Exception("failed to guess 'n_layer'. This model is unknown or unsupported.\n"
                            "Suggestion: provide 'config.json' of the model in the same directory containing model files.")

        n_head = n_embd // 128 # guessed
        n_mult = 256           # guessed

        # TODO: verify this
        n_ff = int(2 * (4 * n_embd) / 3)
        n_ff = n_mult * ((n_ff + n_mult - 1) // n_mult)

        return Params(
            n_vocab    = n_vocab,
            n_embd     = n_embd,
            n_layer    = n_layer,
            n_ctx      = -1,
            n_ff       = n_ff,
            n_head     = n_head,
            n_head_kv  = n_head,
            f_norm_eps = 1e-5,
        )

    @staticmethod
    def loadHFTransformerJson(model: LazyModel, config_path: Path) -> Params:
        config = json.load(open(config_path))

        rope_scaling_type = f_rope_scale = n_orig_ctx = rope_finetuned = None
        rope_scaling = config.get("rope_scaling")

        if rope_scaling is not None and (typ := rope_scaling.get("type")):
            rope_factor = rope_scaling.get("factor")
            f_rope_scale = rope_factor
            if typ == "linear":
                rope_scaling_type = gguf.RopeScalingType.LINEAR
            elif typ == "yarn":
                rope_scaling_type = gguf.RopeScalingType.YARN
                n_orig_ctx = rope_scaling['original_max_position_embeddings']
                rope_finetuned = rope_scaling['finetuned']
            else:
                raise NotImplementedError(f'Unknown rope scaling type: {typ}')

        if "max_sequence_length" in config:
            n_ctx = config["max_sequence_length"]
        elif "max_position_embeddings" in config:
            n_ctx = config["max_position_embeddings"]
        else:
            raise Exception("failed to guess 'n_ctx'. This model is unknown or unsupported.\n"
                            "Suggestion: provide 'config.json' of the model in the same directory containing model files.")

        return Params(
            n_vocab           = config["vocab_size"],
            n_embd            = config["hidden_size"],
            n_layer           = config["num_hidden_layers"],
            n_ctx             = n_ctx,
            n_ff              = config["intermediate_size"],
            n_head            = (n_head := config["num_attention_heads"]),
            n_head_kv         = config.get("num_key_value_heads", n_head),
            f_norm_eps        = config["rms_norm_eps"],
            f_rope_freq_base  = config.get("rope_theta"),
            rope_scaling_type = rope_scaling_type,
            f_rope_scale      = f_rope_scale,
            n_orig_ctx        = n_orig_ctx,
            rope_finetuned    = rope_finetuned,
        )

    # LLaMA v2 70B params.json
    # {"dim": 8192, "multiple_of": 4096, "ffn_dim_multiplier": 1.3, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": -1}
    @staticmethod
    def loadOriginalParamsJson(model: LazyModel, config_path: Path) -> Params:
        config = json.load(open(config_path))

        # hack to determine LLaMA v1 vs v2 vs CodeLlama
        if config.get("rope_theta") == 1000000:
            # CodeLlama
            n_ctx = 16384
        elif config["norm_eps"] == 1e-05:
            # LLaMA v2
            n_ctx = 4096
        else:
            # LLaMA v1
            n_ctx = 2048

        return Params(
            n_vocab          = config.get("vocab_size", model["tok_embeddings.weight"].shape[0]),
            n_embd           = config["dim"],
            n_layer          = config["n_layers"],
            n_ctx            = n_ctx,
            n_ff             = model["layers.0.feed_forward.w1.weight"].shape[0],
            n_head           = (n_head := config["n_heads"]),
            n_head_kv        = config.get("n_kv_heads", n_head),
            f_norm_eps       = config["norm_eps"],
            f_rope_freq_base = config.get("rope_theta"),
        )

    @staticmethod
    def load(model_plus: ModelPlus) -> Params:
        hf_config_path   = model_plus.paths[0].parent / "config.json"
        orig_config_path = model_plus.paths[0].parent / "params.json"

        if hf_config_path.exists():
            params = Params.loadHFTransformerJson(model_plus.model, hf_config_path)
        elif orig_config_path.exists():
            params = Params.loadOriginalParamsJson(model_plus.model, orig_config_path)
        elif model_plus.format != 'none':
            params = Params.guessed(model_plus.model)
        else:
            raise ValueError('Cannot guess params when model format is none')

        params.path_model = model_plus.paths[0].parent

        return params


#
# vocab
#

class BpeVocab:
    def __init__(self, fname_tokenizer: Path, fname_added_tokens: Path | None) -> None:
        self.bpe_tokenizer = json.loads(open(str(fname_tokenizer), encoding="utf-8").read())
        added_tokens: dict[str, int]
        if fname_added_tokens is not None:
            # FIXME: Verify that added tokens here _cannot_ overlap with the main vocab.
            added_tokens = json.load(open(fname_added_tokens, encoding="utf-8"))
        else:
            # Fall back to trying to find the added tokens in tokenizer.json
            tokenizer_json_file = fname_tokenizer.parent / 'tokenizer.json'
            if not tokenizer_json_file.is_file():
                added_tokens = {}
            else:
                tokenizer_json = json.load(open(tokenizer_json_file, encoding="utf-8"))
                added_tokens = dict(
                    (item['content'], item['id'])
                    for item in tokenizer_json.get('added_tokens', [])
                    # Added tokens here can be duplicates of the main vocabulary.
                    if item['content'] not in self.bpe_tokenizer )

        vocab_size: int = len(self.bpe_tokenizer)
        expected_ids    = list(range(vocab_size, vocab_size + len(added_tokens)))
        actual_ids      = sorted(added_tokens.values())
        if expected_ids != actual_ids:
            expected_end_id = vocab_size + len(actual_ids) - 1
            raise Exception(f"Expected the {len(actual_ids)} added token ID(s) to be sequential in the range {vocab_size} - {expected_end_id}; got {actual_ids}")

        items = sorted(added_tokens.items(), key=lambda text_idx: text_idx[1])
        self.added_tokens_list    = [text for (text, idx) in items]
        self.vocab_size_base: int = vocab_size
        self.vocab_size: int      = self.vocab_size_base + len(self.added_tokens_list)
        self.fname_tokenizer      = fname_tokenizer
        self.fname_added_tokens   = fname_added_tokens

    def bpe_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        tokenizer = self.bpe_tokenizer
        from transformers.models.gpt2 import tokenization_gpt2
        reverse_vocab = {id: encoded_tok for encoded_tok, id in tokenizer.items()}

        for i, _ in enumerate(tokenizer):
            yield reverse_vocab[i], 0.0, gguf.TokenType.NORMAL

    def added_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        for text in self.added_tokens_list:
            score = -1000.0
            yield text.encode("utf-8"), score, gguf.TokenType.CONTROL

    def all_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        yield from self.bpe_tokens()
        yield from self.added_tokens()

    def __repr__(self) -> str:
        return f"<BpeVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"


class SentencePieceVocab:
    def __init__(self, fname_tokenizer: Path, fname_added_tokens: Path | None) -> None:
        self.sentencepiece_tokenizer = SentencePieceProcessor(str(fname_tokenizer))
        added_tokens: dict[str, int]
        if fname_added_tokens is not None:
            added_tokens = json.load(open(fname_added_tokens, encoding="utf-8"))
        else:
            added_tokens = {}

        vocab_size: int = self.sentencepiece_tokenizer.vocab_size()

        new_tokens       = {id: piece for piece, id in added_tokens.items() if id >= vocab_size}
        expected_new_ids = list(range(vocab_size, vocab_size + len(new_tokens)))
        actual_new_ids   = sorted(new_tokens.keys())

        if expected_new_ids != actual_new_ids:
            raise ValueError(f"Expected new token IDs {expected_new_ids} to be sequential; got {actual_new_ids}")

        # Token pieces that were added to the base vocabulary.
        self.added_tokens_list  = [new_tokens[id] for id in actual_new_ids]
        self.vocab_size_base    = vocab_size
        self.vocab_size         = self.vocab_size_base + len(self.added_tokens_list)
        self.fname_tokenizer    = fname_tokenizer
        self.fname_added_tokens = fname_added_tokens

    def sentencepiece_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        tokenizer = self.sentencepiece_tokenizer
        for i in range(tokenizer.vocab_size()):
            piece = tokenizer.id_to_piece(i)
            text: bytes = piece.encode("utf-8")
            score: float = tokenizer.get_score(i)

            toktype = gguf.TokenType.NORMAL
            if tokenizer.is_unknown(i):
                toktype = gguf.TokenType.UNKNOWN
            if tokenizer.is_control(i):
                toktype = gguf.TokenType.CONTROL

            # NOTE: I think added_tokens are user defined.
            # ref: https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto
            # if tokenizer.is_user_defined(i): toktype = gguf.TokenType.USER_DEFINED

            if tokenizer.is_unused(i):
                toktype = gguf.TokenType.UNUSED
            if tokenizer.is_byte(i):
                toktype = gguf.TokenType.BYTE

            yield text, score, toktype

    def added_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        for text in self.added_tokens_list:
            score = -1000.0
            yield text.encode("utf-8"), score, gguf.TokenType.USER_DEFINED

    def all_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        yield from self.sentencepiece_tokens()
        yield from self.added_tokens()

    def __repr__(self) -> str:
        return f"<SentencePieceVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"

Vocab: TypeAlias = 'BpeVocab | SentencePieceVocab'

#
# data loading
# TODO: reuse (probably move to gguf.py?)
#

def permute(weights: NDArray, n_head: int, n_head_kv: int) -> NDArray:
    #print( "permute debug " + str(weights.shape[0]) + " x " + str(weights.shape[1]) + " nhead " + str(n_head) + " nheadkv " + str(n_kv_head) )
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))


class Tensor(metaclass=ABCMeta):
    data_type: DataType

    @abstractmethod
    def astype(self, data_type: DataType) -> Tensor: ...
    @abstractmethod
    def permute(self, n_head: int, n_head_kv: int) -> Tensor: ...
    @abstractmethod
    def permute_part(self, n_part: int, n_head: int, n_head_kv: int) -> UnquantizedTensor: ...
    @abstractmethod
    def part(self, n_part: int) -> UnquantizedTensor: ...
    @abstractmethod
    def to_ggml(self) -> GGMLCompatibleTensor: ...


def bf16_to_fp32(bf16_arr: np.ndarray[Any, np.dtype[np.uint16]]) -> NDArray:
    assert bf16_arr.dtype == np.uint16, f"Input array should be of dtype uint16, but got {bf16_arr.dtype}"
    fp32_arr = bf16_arr.astype(np.uint32) << 16
    return fp32_arr.view(np.float32)


class UnquantizedTensor(Tensor):
    def __init__(self, ndarray: NDArray) -> None:
        assert isinstance(ndarray, np.ndarray)
        self.ndarray = ndarray
        self.data_type = NUMPY_TYPE_TO_DATA_TYPE[ndarray.dtype]

    def astype(self, data_type: DataType) -> Tensor:
        dtype = data_type.dtype
        if self.data_type == DT_BF16:
            self.ndarray = bf16_to_fp32(self.ndarray)
        return UnquantizedTensor(self.ndarray.astype(dtype))

    def to_ggml(self) -> UnquantizedTensor:
        return self

    def permute_part(self, n_part: int, n_head: int, n_head_kv: int) -> UnquantizedTensor:
        r = self.ndarray.shape[0] // 3
        return UnquantizedTensor(permute(self.ndarray[r * n_part : r * n_part + r, ...], n_head, n_head_kv))

    def part(self, n_part: int) -> UnquantizedTensor:
        r = self.ndarray.shape[0] // 3
        return UnquantizedTensor(self.ndarray[r * n_part : r * n_part + r, ...])

    def permute(self, n_head: int, n_head_kv: int) -> UnquantizedTensor:
        return UnquantizedTensor(permute(self.ndarray, n_head, n_head_kv))


def load_unquantized(lazy_tensor: LazyTensor, expected_dtype: Any = None, convert: bool = False) -> NDArray:
    tensor = lazy_tensor.load()
    assert isinstance(tensor, UnquantizedTensor)

    # double-check:
    actual_shape = list(tensor.ndarray.shape)
    assert actual_shape == lazy_tensor.shape, (actual_shape, lazy_tensor.shape)
    if expected_dtype is not None and expected_dtype != tensor.ndarray.dtype:
        if convert:
            tensor.ndarray = tensor.ndarray.astype(expected_dtype)
        else:
            raise ValueError(f'expected this tensor to have dtype {expected_dtype}, got {tensor.ndarray.dtype}')

    return tensor.ndarray


GGMLCompatibleTensor = UnquantizedTensor


@dataclass
class LazyTensor:
    _load: Callable[[], Tensor]
    shape: list[int]
    data_type: DataType
    description: str

    def load(self) -> Tensor:
        ret = self._load()
        # Should be okay if it maps to the same numpy type?
        assert ret.data_type == self.data_type or (self.data_type.dtype == ret.data_type.dtype), \
                (self.data_type, ret.data_type, self.description)
        return ret

    def astype(self, data_type: DataType) -> LazyTensor:
        self.validate_conversion_to(data_type)

        def load() -> Tensor:
            return self.load().astype(data_type)
        return LazyTensor(load, self.shape, data_type, f'convert({data_type}) {self.description}')
    
    def transposed(self) -> LazyTensor:
        def load() -> Tensor:
            loaded = self.load()
            assert isinstance(loaded, UnquantizedTensor), f'Cannot transpose {loaded}'
            loaded.ndarray = loaded.ndarray.T
            return loaded
        return LazyTensor(load, self.shape[::-1], self.data_type, f'transpose {self.description}')

    def validate_conversion_to(self, data_type: DataType) -> None:
        if data_type != self.data_type and data_type.name not in self.data_type.valid_conversions:
            raise ValueError(f'Cannot validate conversion from {self.data_type} to {data_type}.')


LazyModel: TypeAlias = 'dict[str, LazyTensor]'


@dataclass
class ModelPlus:
    model: LazyModel
    paths: list[Path]  # Where this was read from.
    format: Literal['ggml', 'torch', 'safetensors', 'none']
    vocab: Vocab | None  # For GGML models (which have vocab built in), the vocab.


def merge_sharded(models: list[LazyModel]) -> LazyModel:
    # Original LLaMA models have each file contain one part of each tensor.
    # Use a dict instead of a set to preserve order.
    names = {name: None for model in models for name in model}

    def convert(name: str) -> LazyTensor:
        lazy_tensors: list[LazyTensor] = [model[name] for model in models]
        if len(lazy_tensors) == 1:
            # only one file; don't go through this procedure since there might
            # be quantized tensors
            return lazy_tensors[0]
        if len(lazy_tensors[0].shape) == 1:
            # the tensor is just duplicated in every file
            return lazy_tensors[0]
        if name.startswith('tok_embeddings.') or \
           name.endswith('.attention.wo.weight') or \
           name.endswith('.feed_forward.w2.weight'):
            # split by columns
            axis = 1
        else:
            # split by rows
            axis = 0
        concatenated_shape = list(lazy_tensors[0].shape)
        concatenated_shape[axis] = sum(tensor.shape[axis] for tensor in lazy_tensors)

        def load() -> UnquantizedTensor:
            ndarrays = [load_unquantized(tensor) for tensor in lazy_tensors]
            concatenated: NDArray = np.concatenate(ndarrays, axis=axis)
            return UnquantizedTensor(concatenated)
        description = 'concatenated[[' + '] | ['.join(lt.description for lt in lazy_tensors) + ']]'
        return LazyTensor(load, concatenated_shape, lazy_tensors[0].data_type, description)
    return {name: convert(name) for name in names}


def merge_multifile_models(models_plus: list[ModelPlus]) -> ModelPlus:
    formats = set(mp.format for mp in models_plus)
    # assert len(formats) == 1, "different formats?"
    format = formats.pop()
    paths = [path for mp in models_plus for path in mp.paths]
    # Use the first non-None vocab, if any.
    try:
        vocab = next(mp.vocab for mp in models_plus if mp.vocab is not None)
    except StopIteration:
        vocab = None

    if any("model.embed_tokens.weight" in mp.model for mp in models_plus) or \
       any("model.layers.0.fc1.weight" in mp.model for mp in models_plus):
        # Transformers models put different tensors in different files, but
        # don't split indivdual tensors between files.
        model: LazyModel = {}
        for mp in models_plus:
            model.update(mp.model)
    else:
        model = merge_sharded([mp.model for mp in models_plus])

    return ModelPlus(model, paths, format, vocab)


def permute_lazy(lazy_tensor: LazyTensor, n_head: int, n_head_kv: int) -> LazyTensor:
    def load() -> Tensor:
        return lazy_tensor.load().permute(n_head, n_head_kv)
    return LazyTensor(load, lazy_tensor.shape, lazy_tensor.data_type, f'permute({n_head}, {n_head_kv}) ' + lazy_tensor.description)

def permute_part_lazy(lazy_tensor: LazyTensor, n_part: int, n_head: int, n_head_kv: int) -> LazyTensor:
    def load() -> Tensor:
        return lazy_tensor.load().permute_part(n_part, n_head, n_head_kv)
    s = lazy_tensor.shape.copy()
    s[0] = s[0] // 3
    return LazyTensor(load, s, lazy_tensor.data_type, f'permute({n_head}, {n_head_kv}) ' + lazy_tensor.description)

def part_lazy(lazy_tensor: LazyTensor, n_part: int) -> LazyTensor:
    def load() -> Tensor:
        return lazy_tensor.load().part(n_part)
    s = lazy_tensor.shape.copy()
    s[0] = s[0] // 3
    return LazyTensor(load, s, lazy_tensor.data_type, 'part ' + lazy_tensor.description)


# Functionality that simulates `torch.load` but where individual tensors are
# only loaded into memory on demand, not all at once.
# PyTorch can't do this natively as of time of writing:
# - https://github.com/pytorch/pytorch/issues/64327
# This allows us to de-shard without multiplying RAM usage, and also
# conveniently drops the PyTorch dependency (though we still need numpy).


@dataclass
class LazyStorageKind:
    data_type: DataType


@dataclass
class LazyStorage:
    load: Callable[[int, int], NDArray]
    kind: LazyStorageKind
    description: str


class LazyUnpickler(pickle.Unpickler):
    def __init__(self, fp: IO[bytes], data_base_path: str, zip_file: zipfile.ZipFile):
        super().__init__(fp)
        self.data_base_path = data_base_path
        self.zip_file = zip_file

    def persistent_load(self, pid: Any) -> Any:
        assert pid[0] == 'storage'
        assert isinstance(pid[1], LazyStorageKind)
        data_type = pid[1].data_type
        filename_stem = pid[2]
        filename = f'{self.data_base_path}/{filename_stem}'
        info = self.zip_file.getinfo(filename)

        def load(offset: int, elm_count: int) -> NDArray:
            dtype = data_type.dtype
            fp = self.zip_file.open(info)
            fp.seek(offset * dtype.itemsize)
            size = elm_count * dtype.itemsize
            data = fp.read(size)
            assert len(data) == size
            return np.frombuffer(data, dtype)
        description = f'storage data_type={data_type} path-in-zip={filename} path={self.zip_file.filename}'
        return LazyStorage(load=load, kind=pid[1], description=description)

    @staticmethod
    def lazy_rebuild_tensor_v2(storage: Any, storage_offset: Any, size: Any, stride: Any,
                               requires_grad: Any, backward_hooks: Any, metadata: Any = None) -> LazyTensor:
        assert isinstance(storage, LazyStorage)

        def load() -> UnquantizedTensor:
            elm_count = stride[0] * size[0]
            return UnquantizedTensor(storage.load(storage_offset, elm_count).reshape(size))
        description = f'pickled storage_offset={storage_offset} in {storage.description}'
        return LazyTensor(load, list(size), storage.kind.data_type, description)

    @staticmethod
    def rebuild_from_type_v2(func, new_type, args, state):
        return func(*args)

    CLASSES: dict[tuple[str, str], Any] = {
        # getattr used here as a workaround for mypy not being smart enough to detrmine
        # the staticmethods have a __func__ attribute.
        ('torch._tensor', '_rebuild_from_type_v2'): getattr(rebuild_from_type_v2, '__func__'),
        ('torch._utils', '_rebuild_tensor_v2'): getattr(lazy_rebuild_tensor_v2, '__func__'),
        ('torch', 'BFloat16Storage'): LazyStorageKind(DT_BF16),
        ('torch', 'HalfStorage'): LazyStorageKind(DT_F16),
        ('torch', 'FloatStorage'): LazyStorageKind(DT_F32),
        ('torch', 'IntStorage'): LazyStorageKind(DT_I32),
        ('torch', 'Tensor'): LazyTensor,
    }

    def find_class(self, module: str, name: str) -> Any:
        if not module.startswith('torch'):
            return super().find_class(module, name)
        return self.CLASSES[(module, name)]


def lazy_load_torch_file(outer_fp: IO[bytes], path: Path) -> ModelPlus:
    zf = zipfile.ZipFile(outer_fp)
    pickle_paths = [name for name in zf.namelist() if name.endswith('.pkl')]
    assert len(pickle_paths) == 1, pickle_paths
    pickle_fp = zf.open(pickle_paths[0], 'r')
    unpickler = LazyUnpickler(pickle_fp,
                              data_base_path=pickle_paths[0][:-4],
                              zip_file=zf)
    model = unpickler.load()
    as_dict = dict(model.items())
    return ModelPlus(model=as_dict, paths=[path], format='torch', vocab=None)


def lazy_load_safetensors_file(fp: IO[bytes], path: Path) -> ModelPlus:
    header_size, = struct.unpack('<Q', fp.read(8))
    header: dict[str, dict[str, Any]] = json.loads(fp.read(header_size))
    # Use mmap for the actual data to avoid race conditions with the file offset.
    mapped = memoryview(mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ))
    byte_buf = mapped[8 + header_size:]

    def convert(info: dict[str, Any]) -> LazyTensor:
        data_type = SAFETENSORS_DATA_TYPES[info['dtype']]
        numpy_dtype = data_type.dtype
        shape: list[int] = info['shape']
        begin, end = info['data_offsets']
        assert 0 <= begin <= end <= len(byte_buf)
        assert end - begin == math.prod(shape) * numpy_dtype.itemsize
        buf = byte_buf[begin:end]

        def load() -> UnquantizedTensor:
            return UnquantizedTensor(np.frombuffer(buf, dtype=numpy_dtype).reshape(shape))
        description = f'safetensors begin={begin} end={end} type={data_type} path={path}'
        return LazyTensor(load, shape, data_type, description)
    model = {name: convert(info) for (name, info) in header.items() if name != '__metadata__'}
    return ModelPlus(model=model, paths=[path], format='safetensors', vocab=None)


def must_read(fp: IO[bytes], length: int) -> bytes:
    ret = fp.read(length)
    if len(ret) < length:
        raise Exception("unexpectedly reached end of file")
    return ret


@functools.lru_cache(maxsize=None)
def lazy_load_file(path: Path) -> ModelPlus:
    fp = open(path, 'rb')
    first8 = fp.read(8)
    fp.seek(0)
    if first8[:2] == b'PK':
        # A zip file, i.e. PyTorch format
        return lazy_load_torch_file(fp, path)
    elif struct.unpack('<Q', first8)[0] < 16 * 1024 * 1024:
        # Probably safetensors
        return lazy_load_safetensors_file(fp, path)
    else:
        raise ValueError(f"unknown format: {path}")


In = TypeVar('In')
Out = TypeVar('Out')

def bounded_parallel_map(func: Callable[[In], Out], iterable: Iterable[In], concurrency: int, max_workers: int | None = None, use_processpool_executor: bool = False) -> Iterable[Out]:
    '''Parallel map, but with backpressure.  If the caller doesn't call `next`
    fast enough, this will stop calling `func` at some point rather than
    letting results pile up in memory.  Specifically, there is a max of one
    output value buffered per thread.'''
    if concurrency < 2:
        yield from map(func, iterable)
        # Not reached.
    iterable = iter(iterable)
    executor_class: type[ThreadPoolExecutor] | type[ProcessPoolExecutor]
    if use_processpool_executor:
        executor_class = ProcessPoolExecutor
    else:
        executor_class = ThreadPoolExecutor
    with executor_class(max_workers = max_workers) as executor:
        futures: list[concurrent.futures.Future[Out]] = []
        done = False
        for _ in range(concurrency):
            try:
                futures.append(executor.submit(func, next(iterable)))
            except StopIteration:
                done = True
                break

        while futures:
            result = futures.pop(0).result()
            while not done and len(futures) < concurrency:
                try:
                    futures.append(executor.submit(func, next(iterable)))
                except StopIteration:
                    done = True
                    break
            yield result

def check_vocab_size(params: Params, vocab: Vocab) -> None:
    if params.n_vocab != vocab.vocab_size:
        assert isinstance(vocab, BpeVocab) or isinstance(vocab, SentencePieceVocab)
        if params.n_vocab == vocab.vocab_size_base:
            print("Ignoring added_tokens.json since model matches vocab size without it.")
            vocab.added_tokens_list = []
            vocab.vocab_size = vocab.vocab_size_base
            return
        msg = f"Vocab size mismatch (model has {params.n_vocab}, but {vocab.fname_tokenizer}"
        if vocab.fname_added_tokens is not None:
            msg += f" combined with {vocab.fname_added_tokens}"
        msg += f" has {vocab.vocab_size})."
        if vocab.vocab_size < params.n_vocab < vocab.vocab_size + 20 and vocab.fname_added_tokens is None:
            msg += f"  Most likely you are missing added_tokens.json (should be in {vocab.fname_tokenizer.parent})."
        raise Exception(msg)


class OutputFile:
    def __init__(self, fname_out: Path, endianess:gguf.GGUFEndian=gguf.GGUFEndian.LITTLE) -> None:
        self.gguf = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[ARCH], endianess=endianess)

    def add_meta_arch(self, params: Params) -> None:
        name = "LLaMA"

        # TODO: better logic to determine model name
        if params.n_ctx == 4096:
            name = "LLaMA v2"
        elif params.path_model is not None:
            name = str(params.path_model.parent).split('/')[-1]

        self.gguf.add_name                (name)
        self.gguf.add_context_length      (params.n_ctx)
        self.gguf.add_embedding_length    (params.n_embd)
        self.gguf.add_block_count         (params.n_layer)
        self.gguf.add_feed_forward_length (params.n_ff)
        self.gguf.add_rope_dimension_count(params.n_embd // params.n_head)
        self.gguf.add_head_count          (params.n_head)
        self.gguf.add_head_count_kv       (params.n_head_kv)
        self.gguf.add_layer_norm_rms_eps  (params.f_norm_eps)

        if params.f_rope_freq_base is not None:
            self.gguf.add_rope_freq_base(params.f_rope_freq_base)

        if params.rope_scaling_type:
            assert params.f_rope_scale is not None
            self.gguf.add_rope_scaling_type(params.rope_scaling_type)
            self.gguf.add_rope_scaling_factor(params.f_rope_scale)

        if params.n_orig_ctx is not None:
            self.gguf.add_rope_scaling_orig_ctx_len(params.n_orig_ctx)

        if params.rope_finetuned is not None:
            self.gguf.add_rope_scaling_finetuned(params.rope_finetuned)

        if params.ftype is not None:
            self.gguf.add_file_type(params.ftype)

        if params.predictor_params.sparse_threshold is not None:
            self.gguf.add_sparse_threshold(params.predictor_params.sparse_threshold)

    def add_meta_vocab(self, vocab: Vocab) -> None:
        tokens = []
        scores = []
        toktypes = []
        # NOTE: `all_tokens` returns the base vocabulary and added tokens
        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        if isinstance(vocab, SentencePieceVocab):
            self.gguf.add_tokenizer_model("llama")
        elif isinstance(vocab, BpeVocab):
            self.gguf.add_tokenizer_model("gpt2")
        else:
            raise ValueError('Unknown vocab type: Not BpeVocab or SentencePieceVocab')
        self.gguf.add_token_list(tokens)
        self.gguf.add_token_scores(scores)
        self.gguf.add_token_types(toktypes)

    def add_meta_special_vocab(self, svocab: gguf.SpecialVocab) -> None:
        svocab.add_to_gguf(self.gguf)

    def add_tensor_info(self, name: str, tensor: LazyTensor) -> None:
        n_elements = int(np.prod(tensor.shape))
        raw_dtype = getattr(tensor.data_type, 'ggml_type', None)
        data_type = getattr(tensor.data_type, 'quantized_type', None) or tensor.data_type.dtype
        data_nbytes = tensor.data_type.elements_to_bytes(n_elements)
        self.gguf.add_tensor_info(name, tensor.shape, data_type, data_nbytes, raw_dtype = raw_dtype)

    def write_meta(self) -> None:
        self.gguf.write_header_to_file()
        self.gguf.write_kv_data_to_file()

    def write_tensor_info(self) -> None:
        self.gguf.write_ti_data_to_file()

    def close(self) -> None:
        self.gguf.close()

    @staticmethod
    def write_vocab_only(fname_out: Path, params: Params, vocab: Vocab, svocab: gguf.SpecialVocab, endianess:gguf.GGUFEndian=gguf.GGUFEndian.LITTLE) -> None:
        check_vocab_size(params, vocab)

        of = OutputFile(fname_out, endianess=endianess)

        # meta data
        of.add_meta_arch(params)
        of.add_meta_vocab(vocab)
        of.add_meta_special_vocab(svocab)

        of.write_meta()

        of.close()

    @staticmethod
    def do_item(item: tuple[str, LazyTensor]) -> tuple[DataType, NDArray]:
        name, lazy_tensor = item
        tensor = lazy_tensor.load().to_ggml()
        return (lazy_tensor.data_type, tensor.ndarray)

    @staticmethod
    def maybe_do_quantize(item: tuple[DataType, NDArray]) -> NDArray:
        dt, arr = item
        if not isinstance(dt, QuantizedDataType):
            return arr
        return dt.quantize(arr)

    @staticmethod
    def write_all(fname_out: Path, ftype: GGMLFileType, params: Params, model: LazyModel, vocab: Vocab, svocab: gguf.SpecialVocab, concurrency: int = DEFAULT_CONCURRENCY, endianess: gguf.GGUFEndian = gguf.GGUFEndian.LITTLE) -> None:
        check_vocab_size(params, vocab)

        of = OutputFile(fname_out, endianess=endianess)

        # meta data
        of.add_meta_arch(params)
        of.add_meta_vocab(vocab)
        of.add_meta_special_vocab(svocab)

        # tensor info
        for name, lazy_tensor in model.items():
            of.add_tensor_info(name, lazy_tensor)

        of.write_meta()
        of.write_tensor_info()

        # tensor data
        ndarrays_inner = bounded_parallel_map(OutputFile.do_item, model.items(), concurrency = concurrency)
        if ftype == GGMLFileType.MostlyQ8_0:
            ndarrays = bounded_parallel_map(OutputFile.maybe_do_quantize, ndarrays_inner, concurrency = concurrency, max_workers = concurrency, use_processpool_executor = True)
        else:
            ndarrays = map(OutputFile.maybe_do_quantize, ndarrays_inner)

        start = time.time()
        for i, ((name, lazy_tensor), ndarray) in enumerate(zip(model.items(), ndarrays)):
            elapsed = time.time() - start
            size = ' x '.join(f"{dim:6d}" for dim in lazy_tensor.shape)
            padi = len(str(len(model)))
            print(f"[{i+1:{padi}d}/{len(model)}] Writing tensor {name:38s} | size {size:16} | type {lazy_tensor.data_type.name:4} | T+{int(elapsed):4}")
            of.gguf.write_tensor_data(ndarray)

        of.close()

def pick_output_type(model: LazyModel, output_type_str: str | None) -> GGMLFileType:
    wq_type = model[gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.ATTN_Q].format(bid=0)+".weight"].data_type

    if output_type_str == "f32" or (output_type_str is None and wq_type == DT_F32):
        return GGMLFileType.AllF32
    if output_type_str == "f16" or (output_type_str is None and wq_type in (DT_F16, DT_BF16)):
        return GGMLFileType.MostlyF16
    if output_type_str == "q8_0":
        return GGMLFileType.MostlyQ8_0

    name_to_type = {name: lazy_tensor.data_type for (name, lazy_tensor) in model.items()}

    raise Exception(f"Unexpected combination of types: {name_to_type}")

def convert_to_output_type(model: LazyModel, output_type: GGMLFileType) -> LazyModel:
    return {name: tensor.astype(output_type.type_for_tensor(name, tensor))
            for (name, tensor) in model.items()}

def convert_model_names(model: LazyModel, params: Params) -> LazyModel:
    tmap = gguf.TensorNameMap(ARCH, params.n_layer)
    should_skip: set[gguf.MODEL_TENSOR] = set(gguf.MODEL_TENSOR_SKIP.get(ARCH, []))

    tmp = model

    # HF models permut or pack some of the tensors, so we need to undo that
    for i in itertools.count():
        if f"model.layers.{i}.self_attn.q_proj.weight" in model:
            print(f"Permuting layer {i}")
            tmp[f"model.layers.{i}.self_attn.q_proj.weight"] = permute_lazy(model[f"model.layers.{i}.self_attn.q_proj.weight"], params.n_head, params.n_head)
            tmp[f"model.layers.{i}.self_attn.k_proj.weight"] = permute_lazy(model[f"model.layers.{i}.self_attn.k_proj.weight"], params.n_head, params.n_head_kv)
           #tmp[f"model.layers.{i}.self_attn.v_proj.weight"] =              model[f"model.layers.{i}.self_attn.v_proj.weight"]
        elif f"model.layers.{i}.self_attn.W_pack.weight" in model:
            print(f"Unpacking and permuting layer {i}")
            tmp[f"model.layers.{i}.self_attn.q_proj.weight"] = permute_part_lazy(model[f"model.layers.{i}.self_attn.W_pack.weight"], 0, params.n_head, params.n_head)
            tmp[f"model.layers.{i}.self_attn.k_proj.weight"] = permute_part_lazy(model[f"model.layers.{i}.self_attn.W_pack.weight"], 1, params.n_head, params.n_head_kv)
            tmp[f"model.layers.{i}.self_attn.v_proj.weight"] = part_lazy        (model[f"model.layers.{i}.self_attn.W_pack.weight"], 2)
            del tmp[f"model.layers.{i}.self_attn.W_pack.weight"]
        else:
            break

    out: LazyModel = {}
    for name, lazy_tensor in model.items():
        tensor_type, name_new = tmap.get_type_and_name(name, try_suffixes = (".weight", ".bias")) or (None, None)
        if name_new is None:
            raise Exception(f"Unexpected tensor name: {name}")

        if tensor_type in should_skip:
            print(f"skipping tensor {name_new}")
            continue

        print(f"{name:48s} -> {name_new:40s} | {lazy_tensor.data_type.name:6s} | {lazy_tensor.shape}")
        out[name_new] = lazy_tensor

    return out

def postprocess_transpose(model: LazyModel) -> LazyModel:
    """Transpose ffn_down matrices for Axpy ops."""
    out: LazyModel = {}
    
    for name, lazy_tensor in model.items():
        if name.endswith(".ffn_down.weight"):
            out[name.replace("ffn_down", "ffn_down_t")] = lazy_tensor.transposed()
        else:
            out[name] = lazy_tensor
    
    return out

def nth_multifile_path(path: Path, n: int) -> Path | None:
    '''Given any path belonging to a multi-file model (e.g. foo.bin.1), return
    the nth path in the model.
    '''
    # Support the following patterns:
    patterns: list[tuple[str, str]] = [
        # - x.00.pth, x.01.pth, etc.
        (r'\.[0-9]{2}\.pth$', f'.{n:02}.pth'),
        # - x-00001-of-00002.bin, x-00002-of-00002.bin, etc.
        (r'-[0-9]{5}-of-(.*)$', fr'-{n:05}-of-\1'),
        # x.bin, x.bin.1, etc.
        (r'(\.[0-9]+)?$', r'\1' if n == 0 else fr'\1.{n}'),
        # x_0.pt, x_1.pt, etc.
        (r'(_[0-9]+)?\.pt$', fr'_{n}.pt'),
    ]
    for regex, replacement in patterns:
        if re.search(regex, path.name):
            new_path = path.with_name(re.sub(regex, replacement, path.name))
            if new_path.exists():
                return new_path
    return None


def find_multifile_paths(path: Path) -> list[Path]:
    '''Given any path belonging to a multi-file model (e.g. foo.bin.1), return
    the whole list of paths in the model.
    '''
    ret: list[Path] = []
    for i in itertools.count():
        nth_path = nth_multifile_path(path, i)
        if nth_path is None:
            break
        ret.append(nth_path)
    if not ret:
        # No matches.  This should only happen if the file was named, e.g.,
        # foo.0, and there was no file named foo.  Oh well, try to process it
        # as a single file.
        return [path]
    return ret


def load_some_model(path: Path) -> ModelPlus:
    '''Load a model of any supported format.'''
    # Be extra-friendly and accept either a file or a directory:
    if path.is_dir():
        # Check if it's a set of safetensors files first
        globs = ["model-00001-of-*.safetensors", "model.safetensors"]
        files = [file for glob in globs for file in path.glob(glob)]
        if not files:
            # Try the PyTorch patterns too, with lower priority
            globs = ["consolidated.00.pth", "pytorch_model-00001-of-*.bin", "*.pt", "pytorch_model.bin"]
            files = [file for glob in globs for file in path.glob(glob)]
        if not files:
            raise Exception(f"Can't find model in directory {path}")
        if len(files) > 1:
            raise Exception(f"Found multiple models in {path}, not sure which to pick: {files}")
        path = files[0]

    paths = find_multifile_paths(path)
    models_plus: list[ModelPlus] = []
    for path in paths:
        print(f"Loading model file {path}")
        models_plus.append(lazy_load_file(path))

    model_plus = merge_multifile_models(models_plus)
    return model_plus

def load_mlp_model(path: Path) -> ModelPlus:
    '''Load MLP models for sparse attention from directory.'''
    assert path.is_dir(), f"MLP model path {path} is not a directory"
    
    first_model_path = path / "model_0.pt"
    assert first_model_path.resolve(), f"MLP model path {path} does not contain model_0.pt"

    model_paths = find_multifile_paths(first_model_path)
    models_plus: list[ModelPlus] = []
    for model_path in model_paths:
        # find number in model_path
        model_layer = int(re.search(r'model_(\d+).pt', str(model_path)).group(1))
        print(f"Loading MLP model file {model_path}")
        mlp_model = lazy_load_file(model_path)
        mlp_model.model = {f"model.layers.{model_layer}.{name}": tensor for name, tensor in mlp_model.model.items()}
        models_plus.append(mlp_model)

    return merge_multifile_models(models_plus)


def load_vocab(path: Path, vocabtype: str | None) -> Vocab:
    # Be extra-friendly and accept either a file or a directory.  Also, if it's
    # a directory, it might be the model directory, and tokenizer.model might
    # be in the parent of that.
    if path.is_dir():
        vocab_file = "tokenizer.model"
        if vocabtype == 'bpe':
            vocab_file = "vocab.json"
        path2 = path / vocab_file
        # Use `.parent` instead of /.. to handle the symlink case better.
        path3 = path.parent / vocab_file
        if path2.exists():
            path = path2
        elif path3.exists():
            path = path3
        else:
            raise FileNotFoundError(
                f"Could not find {vocab_file} in {path} or its parent; "
                "if it's in another directory, pass the directory as --vocab-dir")

    print(f"Loading vocab file '{path}', type '{vocabtype}'")

    added_tokens_path = path.parent / "added_tokens.json"
    if vocabtype == "bpe":
        return BpeVocab(path, added_tokens_path if added_tokens_path.exists() else None)
    elif vocabtype == "spm":
        return SentencePieceVocab(path, added_tokens_path if added_tokens_path.exists() else None)
    else:
        raise ValueError(f"Unsupported vocabulary type {vocabtype}")


def default_outfile(model_paths: list[Path], file_type: GGMLFileType) -> Path:
    namestr = {
        GGMLFileType.AllF32:    "f32",
        GGMLFileType.MostlyF16: "f16",
        GGMLFileType.MostlyQ8_0:"q8_0",
    }[file_type]
    ret = model_paths[0].parent / f"ggml-model-{namestr}.gguf"
    if ret in model_paths:
        sys.stderr.write(
            f"Error: Default output path ({ret}) would overwrite the input. "
            "Please explicitly specify a path using --outfile.\n")
        sys.exit(1)
    return ret


def do_dump_model(model_plus: ModelPlus) -> None:
    print(f"model_plus.paths = {model_plus.paths!r}")
    print(f"model_plus.format = {model_plus.format!r}")
    print(f"model_plus.vocab = {model_plus.vocab!r}")
    for name, lazy_tensor in model_plus.model.items():
        print(f"{name}: shape={lazy_tensor.shape} type={lazy_tensor.data_type}; {lazy_tensor.description}")


def main(args_in: list[str] | None = None) -> None:
    output_choices = ["f32", "f16"]
    if np.uint32(1) == np.uint32(1).newbyteorder("<"):
        # We currently only support Q8_0 output on little endian systems.
        output_choices.append("q8_0")
    parser = argparse.ArgumentParser(description="Convert a LLaMa model to a GGML compatible file")
    parser.add_argument("--dump",        action="store_true",    help="don't convert, just show what's in the model")
    parser.add_argument("--dump-single", action="store_true",    help="don't convert, just show what's in a single model file")
    parser.add_argument("--vocab-only",  action="store_true",    help="extract only the vocab")
    parser.add_argument("--outtype",     choices=output_choices, help="output format - note: q8_0 may be very slow (default: f16 or f32 based on input)")
    parser.add_argument("--vocab-dir",   type=Path,              help="directory containing tokenizer.model, if separate from model file")
    parser.add_argument("--outfile",     type=Path,              help="path to write to; default: based on input")
    parser.add_argument("model",         type=Path,              help="directory containing model file, or model file itself (*.pth, *.pt, *.bin, *.safetensors)")
    parser.add_argument("mlp_model",     type=Path,              help="MLP model for sparse attention")
    parser.add_argument("--vocabtype",   choices=["spm", "bpe"], help="vocab format (default: spm)", default="spm")
    parser.add_argument("--ctx",         type=int,               help="model training context (default: based on input)")
    parser.add_argument("--concurrency", type=int,               help=f"concurrency used for conversion (default: {DEFAULT_CONCURRENCY})", default = DEFAULT_CONCURRENCY)
    parser.add_argument("--bigendian",   action="store_true",    help="model is executed on big endian machine")

    args = parser.parse_args(args_in)
    if args.dump_single:
        model_plus = lazy_load_file(args.model)
        do_dump_model(model_plus)
        return

    if not args.vocab_only:
        model_plus = load_some_model(args.model)
        params = Params.load(model_plus)
        mlp_predictor_plus = load_mlp_model(args.mlp_model)
        params.predictor_params = PredictorParams.load(mlp_predictor_plus)
        model_plus = merge_multifile_models([model_plus, mlp_predictor_plus])
    else:
        model_plus = ModelPlus(model = {}, paths = [args.model / 'dummy'], format = 'none', vocab = None)
        params = Params.load(model_plus)

    if args.dump:
        do_dump_model(model_plus)
        return
    endianess = gguf.GGUFEndian.LITTLE
    if args.bigendian:
        endianess = gguf.GGUFEndian.BIG

    if params.n_ctx == -1:
        if args.ctx is None:
            raise Exception("The model doesn't have a context size, and you didn't specify one with --ctx\n"
                            "Please specify one with --ctx:\n"
                            " - LLaMA v1: --ctx 2048\n"
                            " - LLaMA v2: --ctx 4096\n")
        params.n_ctx = args.ctx

    if args.outtype:
        params.ftype = {
            "f32": GGMLFileType.AllF32,
            "f16": GGMLFileType.MostlyF16,
            "q8_0": GGMLFileType.MostlyQ8_0,
        }[args.outtype]

    print(f"params = {params}")

    vocab: Vocab
    if args.vocab_only:
        if not args.outfile:
            raise ValueError("need --outfile if using --vocab-only")
        # FIXME: Try to respect vocab_dir somehow?
        vocab = load_vocab(args.vocab_dir or args.model, args.vocabtype)
        special_vocab = gguf.SpecialVocab(model_plus.paths[0].parent,
            load_merges = args.vocabtype == 'bpe',
            n_vocab = vocab.vocab_size)
        outfile = args.outfile
        OutputFile.write_vocab_only(outfile, params, vocab, special_vocab)
        print(f"Wrote {outfile}")
        return

    if model_plus.vocab is not None and args.vocab_dir is None:
        vocab = model_plus.vocab
    else:
        vocab_dir = args.vocab_dir if args.vocab_dir else model_plus.paths[0].parent
        vocab = load_vocab(vocab_dir, args.vocabtype)
    # FIXME: Try to respect vocab_dir somehow?
    special_vocab = gguf.SpecialVocab(model_plus.paths[0].parent,
        load_merges = args.vocabtype == 'bpe',
        n_vocab = vocab.vocab_size)

    model   = model_plus.model
    model   = convert_model_names(model, params)
    model   = postprocess_transpose(model)
    ftype   = pick_output_type(model, args.outtype)
    model   = convert_to_output_type(model, ftype)
    outfile = args.outfile or default_outfile(model_plus.paths, ftype)

    params.ftype = ftype
    print(f"Writing {outfile}, format {ftype}")

    OutputFile.write_all(outfile, ftype, params, model, vocab, special_vocab, concurrency = args.concurrency, endianess=endianess)
    print(f"Wrote {outfile}")

    # post-process: write another unique file header to distinguish from the origianl GGUF file
    with open(outfile, "r+b") as fout:
        POWERINFER_MAGIC = int.from_bytes(b"PWRI", "little")
        fout.write(struct.pack("<I", POWERINFER_MAGIC))


if __name__ == '__main__':
    main()
