#!/usr/bin/env python3

import gguf
import argparse
import concurrent.futures
import copy
import enum
import faulthandler
import functools
import io
import itertools
import json
import math
import mmap
import pickle
import re
import signal
import struct
import sys
import zipfile
import numpy as np

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TypeVar, Union)
from sentencepiece import SentencePieceProcessor  # type: ignore

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

if hasattr(faulthandler, 'register') and hasattr(signal, 'SIGUSR1'):
    faulthandler.register(signal.SIGUSR1)

NDArray: 'TypeAlias' = 'np.ndarray[Any, Any]'

ARCH=gguf.MODEL_ARCH.LLAMA
NAMES=gguf.MODEL_TENSOR_NAMES[ARCH]

#
# data types
#

@dataclass(frozen=True)
class UnquantizedDataType:
    name: str

DT_F16  = UnquantizedDataType('F16')
DT_F32  = UnquantizedDataType('F32')
DT_I32  = UnquantizedDataType('I32')
DT_BF16 = UnquantizedDataType('BF16')

DataType = Union[UnquantizedDataType]

DATA_TYPE_TO_NUMPY: Dict[DataType, 'np.dtype[Any]'] = {
    DT_BF16: np.dtype(np.uint16),
    DT_F16:  np.dtype(np.float16),
    DT_F32:  np.dtype(np.float32),
    DT_I32:  np.dtype(np.int32),
}

NUMPY_TYPE_TO_DATA_TYPE: Dict['np.dtype[Any]', DataType] = \
    {dtype: data_type for (data_type, dtype) in DATA_TYPE_TO_NUMPY.items()}

SAFETENSORS_DATA_TYPES: Dict[str, DataType] = {
    'BF16': DT_BF16,
    'F16': DT_F16,
    'F32': DT_F32,
    'I32': DT_I32,
}

# TODO: match this with `llama_ftype`
# TODO: rename to LLAMAFileType
# TODO: move to `gguf.py`
class GGMLFileType(enum.IntEnum):
    AllF32    = 0
    MostlyF16 = 1  # except 1d tensors

    def type_for_tensor(self, name: str, tensor: 'LazyTensor') -> DataType:
        if len(tensor.shape) == 1:
            # 1D tensors are always F32.
            return DT_F32
        elif self == GGMLFileType.AllF32:
            return DT_F32
        elif self == GGMLFileType.MostlyF16:
            return DT_F16
        else:
            raise ValueError(self)


#
# hparams loading
#

@dataclass
class Params:
    n_vocab:    int
    n_embd:     int
    n_mult:     int
    n_layer:    int
    n_ctx:      int
    n_ff:       int
    n_head:     int
    n_head_kv:  int
    f_norm_eps: float

    f_rope_freq_base: Optional[float] = None
    f_rope_scale: Optional[float] = None

    ftype: Optional[GGMLFileType] = None

    # path to the directory containing the model files
    path_model: Optional['Path'] = None

    @staticmethod
    def find_n_mult(n_ff: int, n_embd: int) -> int:
        # hardcoded magic range
        for n_mult in range(8192, 1, -1):
            calc_ff = (((8*n_embd) // 3 + n_mult - 1) // n_mult)*n_mult
            if calc_ff == n_ff:
                return n_mult
        raise Exception(f"failed to find n_mult for (n_ff={n_ff}, n_embd={n_embd}).")

    @staticmethod
    def guessed(model: 'LazyModel') -> 'Params':
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
            n_mult     = n_mult,
            n_layer    = n_layer,
            n_ctx      = -1,
            n_ff       = n_ff,
            n_head     = n_head,
            n_head_kv  = n_head,
            f_norm_eps = 1e-5,
        )

    @staticmethod
    def loadHFTransformerJson(model: 'LazyModel', config_path: 'Path') -> 'Params':
        config = json.load(open(config_path))

        n_vocab          = config["vocab_size"]
        n_embd           = config["hidden_size"]
        n_layer          = config["num_hidden_layers"]
        n_ff             = config["intermediate_size"]
        n_head           = config["num_attention_heads"]
        n_head_kv        = config["num_key_value_heads"] if "num_key_value_heads" in config else n_head
        f_norm_eps       = config["rms_norm_eps"]
        f_rope_freq_base = config["rope_theta"] if "rope_theta" in config else None

        if "rope_scaling" in config and config["rope_scaling"].get("type") == "linear":
            f_rope_scale = config["rope_scaling"].get("factor")
        else:
            f_rope_scale = None

        n_mult = Params.find_n_mult(n_ff, n_embd)

        if "max_sequence_length" in config:
            n_ctx = config["max_sequence_length"]
        elif "max_position_embeddings" in config:
            n_ctx = config["max_position_embeddings"]
        else:
            raise Exception("failed to guess 'n_ctx'. This model is unknown or unsupported.\n"
                            "Suggestion: provide 'config.json' of the model in the same directory containing model files.")

        return Params(
            n_vocab          = n_vocab,
            n_embd           = n_embd,
            n_mult           = n_mult,
            n_layer          = n_layer,
            n_ctx            = n_ctx,
            n_ff             = n_ff,
            n_head           = n_head,
            n_head_kv        = n_head_kv,
            f_norm_eps       = f_norm_eps,
            f_rope_freq_base = f_rope_freq_base,
            f_rope_scale     = f_rope_scale,
        )

    # LLaMA v2 70B params.json
    # {"dim": 8192, "multiple_of": 4096, "ffn_dim_multiplier": 1.3, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": -1
    @staticmethod
    def loadOriginalParamsJson(model: 'LazyModel', config_path: 'Path') -> 'Params':
        config = json.load(open(config_path))

        n_vocab          = config["vocab_size"] if "vocab_size" in config else -1
        n_embd           = config["dim"]
        n_layer          = config["n_layers"]
        n_mult           = config["multiple_of"]
        n_ff             = -1
        n_head           = config["n_heads"]
        n_head_kv        = config["n_kv_heads"] if "n_kv_heads" in config else n_head
        f_norm_eps       = config["norm_eps"]
        f_rope_freq_base = config["rope_theta"] if "rope_theta" in config else None

        # hack to determine LLaMA v1 vs v2 vs CodeLlama
        if f_rope_freq_base and f_rope_freq_base == 1000000:
            # CodeLlama
            n_ctx = 16384
        elif config["norm_eps"] == 1e-05:
            # LLaMA v2
            n_ctx = 4096
        else:
            # LLaMA v1
            n_ctx = 2048

        if n_vocab == -1:
            n_vocab = model["tok_embeddings.weight"].shape[0]

        if n_ff == -1:
            n_ff = model["layers.0.feed_forward.w1.weight"].shape[0]

        return Params(
            n_vocab          = n_vocab,
            n_embd           = n_embd,
            n_mult           = n_mult,
            n_layer          = n_layer,
            n_ctx            = n_ctx,
            n_ff             = n_ff,
            n_head           = n_head,
            n_head_kv        = n_head_kv,
            f_norm_eps       = f_norm_eps,
            f_rope_freq_base = f_rope_freq_base,
        )

    @staticmethod
    def load(model_plus: 'ModelPlus') -> 'Params':
        hf_config_path   = model_plus.paths[0].parent / "config.json"
        orig_config_path = model_plus.paths[0].parent / "params.json"

        if hf_config_path.exists():
            params = Params.loadHFTransformerJson(model_plus.model, hf_config_path)
        elif orig_config_path.exists():
            params = Params.loadOriginalParamsJson(model_plus.model, orig_config_path)
        else:
            params = Params.guessed(model_plus.model)

        params.path_model = model_plus.paths[0].parent

        return params


#
# vocab
#

class BpeVocab:
    def __init__(self, fname_tokenizer: Path, fname_added_tokens: Optional[Path]) -> None:
        self.bpe_tokenizer = json.loads(open(str(fname_tokenizer), encoding="utf-8").read())
        added_tokens: Dict[str, int]
        if fname_added_tokens is not None:
            added_tokens = json.load(open(fname_added_tokens, encoding="utf-8"))
        else:
            added_tokens = {}

        vocab_size: int = len(self.bpe_tokenizer)
        expected_ids    = list(range(vocab_size, vocab_size + len(added_tokens)))
        actual_ids      = sorted(added_tokens.values())
        if expected_ids != actual_ids:
            raise Exception(f"Expected added token IDs to be sequential and start at {len(added_tokens)}; got {actual_ids}")

        items = sorted(added_tokens.items(), key=lambda text_idx: text_idx[1])
        self.added_tokens_list    = [text for (text, idx) in items]
        self.vocab_size_base: int = vocab_size
        self.vocab_size: int      = self.vocab_size_base + len(self.added_tokens_list)
        self.fname_tokenizer      = fname_tokenizer
        self.fname_added_tokens   = fname_added_tokens

    def bpe_tokens(self) -> Iterable[Tuple[bytes, float, gguf.TokenType]]:
        tokenizer = self.bpe_tokenizer
        from transformers.models.gpt2 import tokenization_gpt2
        byte_encoder = tokenization_gpt2.bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}
        for i, item in enumerate(tokenizer):
            text: bytes = item.encode("utf-8")
            score: float = -i
            yield text, score, gguf.TokenType.USER_DEFINED

    def added_tokens(self) -> Iterable[Tuple[bytes, float, gguf.TokenType]]:
        for text in self.added_tokens_list:
            score = -1000.0
            yield text.encode("utf-8"), score, gguf.TokenType.USER_DEFINED

    def all_tokens(self) -> Iterable[Tuple[bytes, float, gguf.TokenType]]:
        yield from self.bpe_tokens()
        yield from self.added_tokens()

    def __repr__(self) -> str:
        return f"BpeVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"


class SentencePieceVocab:
    def __init__(self, fname_tokenizer: Path, fname_added_tokens: Optional[Path]) -> None:
        self.sentencepiece_tokenizer = SentencePieceProcessor(str(fname_tokenizer))
        added_tokens: Dict[str, int]
        if fname_added_tokens is not None:
            added_tokens = json.load(open(fname_added_tokens, encoding="utf-8"))
        else:
            added_tokens = {}

        vocab_size: int = self.sentencepiece_tokenizer.vocab_size()
        expected_ids = list(range(vocab_size, vocab_size + len(added_tokens)))
        actual_ids   = sorted(added_tokens.values())
        if expected_ids != actual_ids:
            raise Exception(f"Expected added token IDs to be sequential and start at {len(added_tokens)}; got {actual_ids}")

        items = sorted(added_tokens.items(), key=lambda text_idx: text_idx[1])
        self.added_tokens_list = [text for (text, idx) in items]
        self.vocab_size_base: int = vocab_size
        self.vocab_size: int = self.vocab_size_base + len(self.added_tokens_list)
        self.fname_tokenizer = fname_tokenizer
        self.fname_added_tokens = fname_added_tokens

    def sentencepiece_tokens(self) -> Iterable[Tuple[bytes, float, gguf.TokenType]]:
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

    def added_tokens(self) -> Iterable[Tuple[bytes, float, gguf.TokenType]]:
        for text in self.added_tokens_list:
            score = -1000.0
            yield text.encode("utf-8"), score, gguf.TokenType.USER_DEFINED

    def all_tokens(self) -> Iterable[Tuple[bytes, float, gguf.TokenType]]:
        yield from self.sentencepiece_tokens()
        yield from self.added_tokens()

    def __repr__(self) -> str:
        return f"<SentencePieceVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"

Vocab = Union[BpeVocab, SentencePieceVocab]


#
# data loading
# TODO: reuse (probably move to gguf.py?)
#

def permute(weights: NDArray, n_head: int, n_head_kv: int) -> NDArray:
    #print( "permute debug " + str(weights.shape[0]) + " x " + str(weights.shape[1]) + " nhead " + str(n_head) + " nheadkv " + str(n_kv_head) )
    if n_head_kv is not None and n_head != n_head_kv:
        n_head //= n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))


class Tensor(metaclass=ABCMeta):
    data_type: DataType

    @abstractmethod
    def astype(self, data_type: DataType) -> 'Tensor': ...
    @abstractmethod
    def permute(self, n_head: int, n_head_kv: int) -> 'Tensor': ...
    @abstractmethod
    def permute_part(self, n_part: int, n_head: int) -> 'UnquantizedTensor': ...
    @abstractmethod
    def part(self, n_part: int) -> 'UnquantizedTensor': ...
    @abstractmethod
    def to_ggml(self) -> 'GGMLCompatibleTensor': ...


def bf16_to_fp32(bf16_arr: np.ndarray) -> np.ndarray:
    assert bf16_arr.dtype == np.uint16, f"Input array should be of dtype uint16, but got {bf16_arr.dtype}"
    fp32_arr = bf16_arr.astype(np.uint32) << 16
    return fp32_arr.view(np.float32)


class UnquantizedTensor(Tensor):
    def __init__(self, ndarray: NDArray) -> None:
        assert isinstance(ndarray, np.ndarray)
        self.ndarray = ndarray
        self.data_type = NUMPY_TYPE_TO_DATA_TYPE[ndarray.dtype]

    def astype(self, data_type: DataType) -> Tensor:
        dtype = DATA_TYPE_TO_NUMPY[data_type]
        if self.data_type == DT_BF16:
            self.ndarray = bf16_to_fp32(self.ndarray)
        return UnquantizedTensor(self.ndarray.astype(dtype))

    def to_ggml(self) -> 'UnquantizedTensor':
        return self

    def permute_part(self, n_part: int, n_head: int) -> 'UnquantizedTensor':
        r = self.ndarray.shape[0] // 3
        return UnquantizedTensor(permute(self.ndarray[r * n_part : r * n_part + r, ...], n_head))

    def part(self, n_part: int) -> 'UnquantizedTensor':
        r = self.ndarray.shape[0] // 3
        return UnquantizedTensor(self.ndarray[r * n_part : r * n_part + r, ...])

    def permute(self, n_head: int, n_head_kv: int) -> 'UnquantizedTensor':
        return UnquantizedTensor(permute(self.ndarray, n_head, n_head_kv))


def load_unquantized(lazy_tensor: 'LazyTensor', expected_dtype: Any = None, convert: bool = False) -> NDArray:
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


GGMLCompatibleTensor = Union[UnquantizedTensor]


class DeferredPermutedTensor(Tensor):
    def __init__(self, base: Tensor, n_head: int, n_head_kv: int) -> None:
        self.base = base
        self.n_head = n_head
        self.data_type = self.base.data_type

    def astype(self, data_type: DataType) -> Tensor:
        return self.base.astype(data_type).permute(self.n_head, self.n_head_kv)

    def to_ggml(self) -> GGMLCompatibleTensor:
        return self.base.to_ggml().permute(self.n_head, self.n_head_kv)

    def permute(self, n_head: int, n_head_kv: int) -> Tensor:
        raise Exception("shouldn't permute twice")


@dataclass
class LazyTensor:
    _load: Callable[[], Tensor]
    shape: List[int]
    data_type: DataType
    description: str

    def load(self) -> Tensor:
        ret = self._load()
        assert ret.data_type == self.data_type, (self.data_type, ret.data_type, self.description)
        return ret

    def astype(self, data_type: DataType) -> 'LazyTensor':
        self.validate_conversion_to(data_type)

        def load() -> Tensor:
            return self.load().astype(data_type)
        return LazyTensor(load, self.shape, data_type, f'convert({data_type}) {self.description}')

    def validate_conversion_to(self, data_type: DataType) -> None:
        if data_type == self.data_type:
            return


LazyModel = Dict[str, LazyTensor]


@dataclass
class ModelPlus:
    model: LazyModel
    paths: List[Path]  # Where this was read from.
    format: Literal['ggml', 'torch', 'safetensors']
    vocab: Optional[Vocab]  # For GGML models (which have vocab built in), the vocab.


def merge_sharded(models: List[LazyModel]) -> LazyModel:
    # Original LLaMA models have each file contain one part of each tensor.
    # Use a dict instead of a set to preserve order.
    names = {name: None for model in models for name in model}

    def convert(name: str) -> LazyTensor:
        lazy_tensors: List[LazyTensor] = [model[name] for model in models]
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


def merge_multifile_models(models_plus: List[ModelPlus]) -> ModelPlus:
    formats = set(mp.format for mp in models_plus)
    assert len(formats) == 1, "different formats?"
    format = formats.pop()
    paths = [path for mp in models_plus for path in mp.paths]
    # Use the first non-None vocab, if any.
    try:
        vocab = next(mp.vocab for mp in models_plus if mp.vocab is not None)
    except StopIteration:
        vocab = None

    if any("model.embed_tokens.weight" in mp.model for mp in models_plus):
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

def permute_part_lazy(lazy_tensor: LazyTensor, n_part: int, n_head: int) -> LazyTensor:
    def load() -> Tensor:
        return lazy_tensor.load().permute_part(n_part, n_head)
    s = lazy_tensor.shape.copy()
    s[0] = s[0] // 3
    return LazyTensor(load, s, lazy_tensor.data_type, f'permute({n_head}) ' + lazy_tensor.description)

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
        filename = self.data_base_path + '/' + filename_stem
        info = self.zip_file.getinfo(filename)

        def load(offset: int, elm_count: int) -> NDArray:
            dtype = DATA_TYPE_TO_NUMPY.get(data_type)
            if dtype is None:
                raise Exception("tensor stored in unsupported format")
            fp = self.zip_file.open(info)
            fp.seek(offset * dtype.itemsize)
            size = elm_count * dtype.itemsize
            data = fp.read(size)
            assert len(data) == size
            return np.frombuffer(data, dtype)
        description = f'storage data_type={data_type} path-in-zip={filename} path={self.zip_file.filename}'
        return LazyStorage(load=load, kind=pid[1], description=description)

    # @staticmethod
    def lazy_rebuild_tensor_v2(storage: Any, storage_offset: Any, size: Any, stride: Any,
                               # pyright: ignore[reportSelfClsParameterName]
                               requires_grad: Any, backward_hooks: Any, metadata: Any = None) -> LazyTensor:
        assert isinstance(storage, LazyStorage)

        def load() -> UnquantizedTensor:
            elm_count = stride[0] * size[0]
            return UnquantizedTensor(storage.load(storage_offset, elm_count).reshape(size))
        description = f'pickled storage_offset={storage_offset} in {storage.description}'
        return LazyTensor(load, list(size), storage.kind.data_type, description)

    # @staticmethod
    def rebuild_from_type_v2(func, new_type, args, state):
        return func(*args)

    CLASSES: Dict[Any, Any] = {
        ('torch._tensor', '_rebuild_from_type_v2'): rebuild_from_type_v2,
        ('torch._utils', '_rebuild_tensor_v2'): lazy_rebuild_tensor_v2,
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
    header: Dict[str, Dict[str, Any]] = json.loads(fp.read(header_size))
    # Use mmap for the actual data to avoid race conditions with the file offset.
    mapped = memoryview(mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ))
    byte_buf = mapped[8 + header_size:]

    def convert(info: Dict[str, Any]) -> LazyTensor:
        data_type = SAFETENSORS_DATA_TYPES[info['dtype']]
        numpy_dtype = DATA_TYPE_TO_NUMPY[data_type]
        shape: List[int] = info['shape']
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

def bounded_parallel_map(func: Callable[[In], Out], iterable: Iterable[In], concurrency: int) -> Iterable[Out]:
    '''Parallel map, but with backpressure.  If the caller doesn't call `next`
    fast enough, this will stop calling `func` at some point rather than
    letting results pile up in memory.  Specifically, there is a max of one
    output value buffered per thread.'''
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures: List[concurrent.futures.Future[Out]] = []
        items_rev = list(iterable)[::-1]
        for i in range(min(concurrency, len(items_rev))):
            futures.append(executor.submit(func, items_rev.pop()))
        while futures:
            result = futures.pop(0).result()
            if items_rev:
                futures.append(executor.submit(func, items_rev.pop()))
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
    def __init__(self, fname_out: Path) -> None:
        self.gguf = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[ARCH])

    def add_meta_arch(self, params: Params) -> None:
        name = "LLaMA"
        if (params.n_ctx == 4096):
            name = "LLaMA v2"
            if params.path_model:
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

        if params.f_rope_freq_base:
            self.gguf.add_rope_freq_base(params.f_rope_freq_base)

        if params.f_rope_scale:
            self.gguf.add_rope_scale_linear(params.f_rope_scale)

        if params.ftype:
            self.gguf.add_file_type(params.ftype)

    def add_meta_vocab(self, vocab: Vocab) -> None:
        tokens = []
        scores = []
        toktypes = []
        # NOTE: `all_tokens` returns the the base vocabulary and added tokens
        # TODO: add special tokens?
        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        self.gguf.add_tokenizer_model("llama")
        self.gguf.add_token_list(tokens)
        self.gguf.add_token_scores(scores)
        self.gguf.add_token_types(toktypes)

    def add_tensor_info(self, name: str, tensor: LazyTensor) -> None:
        n_elements = 1
        for dim in tensor.shape:
            n_elements *= dim
        data_type = DATA_TYPE_TO_NUMPY[tensor.data_type]
        data_nbytes = n_elements * data_type.itemsize
        self.gguf.add_tensor_info(name, tensor.shape, data_type, data_nbytes)

    def write_meta(self) -> None:
        self.gguf.write_header_to_file()
        self.gguf.write_kv_data_to_file()

    def write_tensor_info(self) -> None:
        self.gguf.write_ti_data_to_file()

    def close(self) -> None:
        self.gguf.close()

    @staticmethod
    def write_vocab_only(fname_out: Path, params: Params, vocab: Vocab) -> None:
        check_vocab_size(params, vocab)

        of = OutputFile(fname_out)

        # meta data
        of.add_meta_arch(params)
        of.add_meta_vocab(vocab)
        of.write_meta()

        of.close()

    @staticmethod
    def write_all(fname_out: Path, params: Params, model: LazyModel, vocab: Vocab) -> None:
        check_vocab_size(params, vocab)

        of = OutputFile(fname_out)

        # meta data
        of.add_meta_arch(params)
        of.add_meta_vocab(vocab)

        # tensor info
        for name, lazy_tensor in model.items():
            of.add_tensor_info(name, lazy_tensor)

        of.write_meta()
        of.write_tensor_info()

        def do_item(item: Tuple[str, LazyTensor]) -> NDArray:
            name, lazy_tensor = item
            return lazy_tensor.load().to_ggml().ndarray

        # tensor data
        ndarrays = bounded_parallel_map(do_item, model.items(), concurrency=8)
        for i, ((name, lazy_tensor), ndarray) in enumerate(zip(model.items(), ndarrays)):
            size = ' x '.join(f"{dim:6d}" for dim in lazy_tensor.shape)
            padi = len(str(len(model)))
            print(f"[{i+1:{padi}d}/{len(model)}] Writing tensor {name:38s} | size {size:16} | type {lazy_tensor.data_type}")
            of.gguf.write_tensor_data(ndarray)

        of.close()

def pick_output_type(model: LazyModel, output_type_str: Optional[str]) -> GGMLFileType:
    wq_type = model[NAMES[gguf.MODEL_TENSOR.ATTN_Q].format(bid=0)+".weight"].data_type

    if output_type_str == "f32" or (output_type_str is None and wq_type == DT_F32):
        return GGMLFileType.AllF32
    if output_type_str == "f16" or (output_type_str is None and wq_type in (DT_F16, DT_BF16)):
        return GGMLFileType.MostlyF16

    name_to_type = {name: lazy_tensor.data_type for (name, lazy_tensor) in model.items()}

    raise Exception(f"Unexpected combination of types: {name_to_type}")

def convert_to_output_type(model: LazyModel, output_type: GGMLFileType) -> LazyModel:
    return {name: tensor.astype(output_type.type_for_tensor(name, tensor))
            for (name, tensor) in model.items()}

def convert_model_names(model: LazyModel, params: Params) -> LazyModel:
    tmap = gguf.get_tensor_name_map(ARCH, params.n_layer)

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
        else:
            break

    out: LazyModel = {}
    for name, lazy_tensor in model.items():
        name_new = name

        if name in tmap:
            name_new = tmap[name]
        elif name.endswith(".weight") and name[:-7] in tmap:
            name_new = tmap[name[:-7]] + ".weight"
        elif name.endswith(".bias") and name[:-5] in tmap:
            name_new = tmap[name[:-5]] + ".bias"
        else:
            raise Exception(f"Unexpected tensor name: {name}")

        if gguf.should_skip_tensor_TMP(ARCH, params.n_layer, name_new):
            print(f"skipping tensor {name_new}")
            continue
        else:
            print(f"{name:48s} -> {name_new:40s} | {lazy_tensor.data_type} | {lazy_tensor.shape}")
            out[name_new] = lazy_tensor

    return out

def nth_multifile_path(path: Path, n: int) -> Optional[Path]:
    '''Given any path belonging to a multi-file model (e.g. foo.bin.1), return
    the nth path in the model.
    '''
    # Support the following patterns:
    patterns: List[Tuple[str, str]] = [
        # - x.00.pth, x.01.pth, etc.
        (r'\.[0-9]{2}\.pth$', f'.{n:02}.pth'),
        # - x-00001-of-00002.bin, x-00002-of-00002.bin, etc.
        (r'-[0-9]{5}-of-(.*)$', fr'-{n:05}-of-\1'),
        # x.bin, x.bin.1, etc.
        (r'(\.[0-9]+)?$', r'\1' if n == 0 else fr'\1.{n}')
    ]
    for regex, replacement in patterns:
        if re.search(regex, path.name):
            new_path = path.with_name(re.sub(regex, replacement, path.name))
            if new_path.exists():
                return new_path
    return None


def find_multifile_paths(path: Path) -> List[Path]:
    '''Given any path belonging to a multi-file model (e.g. foo.bin.1), return
    the whole list of paths in the model.
    '''
    ret: List[Path] = []
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
        files = list(path.glob("model-00001-of-*.safetensors"))
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
    models_plus: List[ModelPlus] = []
    for path in paths:
        print(f"Loading model file {path}")
        models_plus.append(lazy_load_file(path))

    model_plus = merge_multifile_models(models_plus)
    return model_plus


def load_vocab(path: Path, vocabtype: Optional[str]) -> Union[BpeVocab, SentencePieceVocab]:
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


def default_outfile(model_paths: List[Path], file_type: GGMLFileType) -> Path:
    namestr = {
        GGMLFileType.AllF32:    "f32",
        GGMLFileType.MostlyF16: "f16",
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


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a LLaMa model to a GGML compatible file")
    parser.add_argument("--dump",        action="store_true",    help="don't convert, just show what's in the model")
    parser.add_argument("--dump-single", action="store_true",    help="don't convert, just show what's in a single model file")
    parser.add_argument("--vocab-only",  action="store_true",    help="extract only the vocab")
    parser.add_argument("--outtype",     choices=["f32", "f16"], help="output format (default: based on input)")
    parser.add_argument("--vocab-dir",   type=Path,              help="directory containing tokenizer.model, if separate from model file")
    parser.add_argument("--outfile",     type=Path,              help="path to write to; default: based on input")
    parser.add_argument("model",         type=Path,              help="directory containing model file, or model file itself (*.pth, *.pt, *.bin)")
    parser.add_argument("--vocabtype",   choices=["spm", "bpe"], help="vocab format (default: spm)", default="spm")
    parser.add_argument("--ctx",         type=int,               help="model training context (default: based on input)")
    args = parser.parse_args(args_in)

    if args.dump_single:
        model_plus = lazy_load_file(args.model)
        do_dump_model(model_plus)

    model_plus = load_some_model(args.model)

    params = Params.load(model_plus)
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
        }[args.outtype]

    print(f"params = {params}")

    vocab: Vocab
    if args.vocab_only:
        vocab = load_vocab(args.vocab_dir or args.model, args.vocabtype)
        assert args.outfile, "need --outfile if using --vocab-only"
        outfile = args.outfile
        OutputFile.write_vocab_only(outfile, params, vocab)
        print(f"Wrote {outfile}")
    else:
        if args.dump:
            do_dump_model(model_plus)
            return

        if model_plus.vocab is not None and args.vocab_dir is None:
            vocab = model_plus.vocab
        else:
            vocab_dir = args.vocab_dir if args.vocab_dir else model_plus.paths[0].parent
            vocab = load_vocab(vocab_dir, args.vocabtype)

        model   = model_plus.model
        model   = convert_model_names(model, params)
        ftype   = pick_output_type(model, args.outtype)
        model   = convert_to_output_type(model, ftype)
        outfile = args.outfile or default_outfile(model_plus.paths, ftype)

        params.ftype = ftype
        print(f"Writing {outfile}, format {ftype}")

        OutputFile.write_all(outfile, params, model, vocab)
        print(f"Wrote {outfile}")


if __name__ == '__main__':
    main()
