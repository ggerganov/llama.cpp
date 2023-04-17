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
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Literal, Optional, Sequence, Tuple, TypeVar, Union)

import numpy as np
from sentencepiece import SentencePieceProcessor  # type: ignore

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

if hasattr(faulthandler, 'register') and hasattr(signal, 'SIGUSR1'):
    faulthandler.register(signal.SIGUSR1)

NDArray: 'TypeAlias' = 'np.ndarray[Any, Any]'


@dataclass(frozen=True)
class UnquantizedDataType:
    name: str


DT_F16 = UnquantizedDataType('F16')
DT_F32 = UnquantizedDataType('F32')
DT_I32 = UnquantizedDataType('I32')
DT_BF16 = UnquantizedDataType('BF16')


@dataclass(frozen=True)
class QuantizedDataType:
    groupsize: int
    have_addends: bool
    have_g_idx: bool


DT_Q4_0 = QuantizedDataType(groupsize=32, have_addends=False, have_g_idx=False)
DT_Q4_1 = QuantizedDataType(groupsize=32, have_addends=True, have_g_idx=False)

DataType = Union[UnquantizedDataType, QuantizedDataType]

DATA_TYPE_TO_FTYPE: Dict[DataType, int] = {
    DT_F32: 0,
    DT_F16: 1,
    DT_Q4_0: 2,
    DT_Q4_1: 3,
}

FTYPE_TO_DATA_TYPE: Dict[int, DataType] = \
    {ftype: dtype for (dtype, ftype) in DATA_TYPE_TO_FTYPE.items()}

DATA_TYPE_TO_NUMPY: Dict[DataType, 'np.dtype[Any]'] = {
    DT_F16: np.dtype(np.float16),
    DT_F32: np.dtype(np.float32),
    DT_I32: np.dtype(np.int32),
}

NUMPY_TYPE_TO_DATA_TYPE: Dict['np.dtype[Any]', DataType] = \
    {dtype: data_type for (data_type, dtype) in DATA_TYPE_TO_NUMPY.items()}


class GGMLFileType(enum.Enum):
    AllF32 = 0
    MostlyF16 = 1  # except 1d tensors
    MostlyQ4_0 = 2  # except 1d tensors
    MostlyQ4_1 = 3  # except 1d tensors
    PerLayerIsQ4_1 = 4  # but tok_embeddings.weight and output.weight are F16

    def type_for_tensor(self, name: str, tensor: 'LazyTensor') -> DataType:
        if len(tensor.shape) == 1:
            # 1D tensors are always F32.
            return DT_F32
        elif self == GGMLFileType.AllF32:
            return DT_F32
        elif self == GGMLFileType.MostlyF16:
            return DT_F16
        elif self == GGMLFileType.MostlyQ4_0:
            return DT_Q4_0
        elif self == GGMLFileType.MostlyQ4_1:
            return DT_Q4_1
        elif self == GGMLFileType.PerLayerIsQ4_1:
            if name in ('output.weight', 'tok_embeddings.weight'):
                return DT_F16
            else:
                return DT_Q4_1
        else:
            raise ValueError(self)


def make_tensors_list() -> List[str]:
    ret = [
        'tok_embeddings.weight',
        'norm.weight',
        'output.weight',
    ]
    for i in range(80):  # maximum number of layer
        ret += [
            f'layers.{i}.attention.wq.weight',
            f'layers.{i}.attention.wk.weight',
            f'layers.{i}.attention.wv.weight',
            f'layers.{i}.attention.wo.weight',
            f'layers.{i}.attention_norm.weight',
            f'layers.{i}.feed_forward.w1.weight',
            f'layers.{i}.feed_forward.w2.weight',
            f'layers.{i}.feed_forward.w3.weight',
            f'layers.{i}.atttention_norm.weight',
            f'layers.{i}.ffn_norm.weight',
        ]
    return ret


TENSORS_LIST = make_tensors_list()
TENSORS_SET = set(TENSORS_LIST)


@dataclass
class Params:
    n_vocab: int
    n_embd: int
    n_mult: int
    n_head: int
    n_layer: int
    file_type: GGMLFileType

    @staticmethod
    def guessed(model: 'LazyModel', file_type: GGMLFileType) -> 'Params':
        n_vocab, n_embd = model["tok_embeddings.weight"].shape

        return Params(
            n_vocab=n_vocab,
            n_embd=n_embd,
            n_mult=256,
            n_head=n_embd // 128,
            n_layer=next(i for i in itertools.count() if f"layers.{i}.attention.wq.weight" not in model),
            file_type=file_type,
        )


class SentencePieceVocab:
    def __init__(self, fname_tokenizer: Path, fname_added_tokens: Optional[Path]) -> None:
        self.sentencepiece_tokenizer = SentencePieceProcessor(str(fname_tokenizer))
        added_tokens: Dict[str, int]
        if fname_added_tokens is not None:
            added_tokens = json.load(open(fname_added_tokens))
        else:
            added_tokens = {}
        vocab_size: int = self.sentencepiece_tokenizer.vocab_size()
        expected_ids = list(range(vocab_size, vocab_size + len(added_tokens)))
        actual_ids = sorted(added_tokens.values())
        if expected_ids != actual_ids:
            raise Exception(f"Expected added token IDs to be sequential and start at {len(added_tokens)}; got {actual_ids}")
        items = sorted(added_tokens.items(), key=lambda text_idx: text_idx[1])
        self.added_tokens_list = [text for (text, idx) in items]
        self.vocab_size_base: int = vocab_size
        self.vocab_size: int = self.vocab_size_base + len(self.added_tokens_list)
        self.fname_tokenizer = fname_tokenizer
        self.fname_added_tokens = fname_added_tokens

    def sentencepiece_tokens(self) -> Iterable[Tuple[bytes, float]]:
        tokenizer = self.sentencepiece_tokenizer
        for i in range(tokenizer.vocab_size()):
            text: bytes
            if tokenizer.is_unknown(i):
                text = " \u2047 ".encode("utf-8")
            elif tokenizer.is_control(i):
                text = b""
            elif tokenizer.is_byte(i):
                piece = tokenizer.id_to_piece(i)
                if len(piece) != 6:
                    raise Exception(f"Invalid token: {piece}")
                byte_value = int(piece[3:-1], 16)
                text = struct.pack("B", byte_value)
            else:
                text = tokenizer.id_to_piece(i).replace("\u2581", " ").encode("utf-8")
            score: float = tokenizer.get_score(i)
            yield text, score

    def added_tokens(self) -> Iterable[Tuple[bytes, float]]:
        for text in self.added_tokens_list:
            score = -1000.0
            yield text.encode("utf-8"), score

    def all_tokens(self) -> Iterable[Tuple[bytes, float]]:
        yield from self.sentencepiece_tokens()
        yield from self.added_tokens()

    def __repr__(self) -> str:
        return f"<SentencePieceVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"


class GGMLVocab:
    def __init__(self, tokens: List[Tuple[bytes, float]]):
        self.tokens = tokens
        self.vocab_size = len(tokens)

    def all_tokens(self) -> Iterable[Tuple[bytes, float]]:
        return self.tokens

    def __repr__(self) -> str:
        return f"<GGMLVocab with {self.vocab_size} tokens>"


Vocab = Union[SentencePieceVocab, GGMLVocab]


def permute(weights: NDArray, n_head: int) -> NDArray:
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                   .swapaxes(1, 2)
                   .reshape(weights.shape))


def dequantize_q4(qvalues_pack32: NDArray, scales: NDArray, addends: Optional[NDArray], g_idx: Optional[NDArray]) -> NDArray:
    # First reinterpret each row from a list of int32s containing 8 values each
    # to a list of uint8s containing 2 values each.
    qvalues_pack8 = qvalues_pack32.view(np.uint8)

    # Then split out the two values per int8 (which requires an actual
    # conversion because numpy doesn't natively support int4s).
    qvalues = np.zeros([qvalues_pack8.shape[0], qvalues_pack8.shape[1] * 2], dtype=np.uint8)
    qvalues[:, 0::2] = qvalues_pack8 & 0xf
    qvalues[:, 1::2] = qvalues_pack8 >> 4

    assert addends is None or addends.shape == scales.shape
    assert qvalues.shape[0] == scales.shape[0]
    assert qvalues.shape[1] % scales.shape[1] == 0
    if g_idx is None:
        repeat_count = qvalues.shape[1] // scales.shape[1]
        scales = scales[:, :, np.newaxis]
        if addends is not None:
            addends = addends[:, :, np.newaxis]
        # Reshape so that the below computation broadcasts over scales and addends:
        qvalues.shape = (qvalues.shape[0], scales.shape[1], int(repeat_count))
    else:
        # In this case the scale and addend is selected for each column by g_idx:
        assert addends is not None
        scales = scales[:, g_idx]
        addends = addends[:, g_idx]
    if addends is None:
        # Q4_0
        qvalues = qvalues.view(np.int8)
        qvalues -= 8
    # And do the actual 'value = scale * qvalue + addend' computation.
    values = scales * qvalues
    if addends is not None:
        values += addends
    if g_idx is None:
        values.shape = (values.shape[0], values.shape[1] * values.shape[2])
    return values


class Tensor(metaclass=ABCMeta):
    data_type: DataType

    @abstractmethod
    def astype(self, data_type: DataType) -> 'Tensor': ...
    @abstractmethod
    def permute(self, n_head: int) -> 'Tensor': ...
    @abstractmethod
    def to_ggml(self) -> 'GGMLCompatibleTensor': ...


class UnquantizedTensor(Tensor):
    def __init__(self, ndarray: NDArray) -> None:
        assert isinstance(ndarray, np.ndarray)
        self.ndarray = ndarray
        self.data_type = NUMPY_TYPE_TO_DATA_TYPE[ndarray.dtype]

    def astype(self, data_type: DataType) -> Tensor:
        dtype = DATA_TYPE_TO_NUMPY[data_type]
        return UnquantizedTensor(self.ndarray.astype(dtype))

    def to_ggml(self) -> 'UnquantizedTensor':
        return self

    def permute(self, n_head: int) -> 'UnquantizedTensor':
        return UnquantizedTensor(permute(self.ndarray, n_head))


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


class GGMLQuantizedTensor(Tensor):
    data_type: QuantizedDataType

    def __init__(self, ndarray: NDArray, shape: List[int], data_type: DataType) -> None:
        rows, columns = shape
        assert data_type in (DT_Q4_1, DT_Q4_0)  # for now
        assert isinstance(data_type, QuantizedDataType)  # redundant, but mypy complains without this
        assert columns % data_type.groupsize == 0
        words_in_block = 6 if data_type == DT_Q4_1 else 5
        self.ndarray = ndarray.view(dtype=np.uint32).reshape((rows, columns // data_type.groupsize, words_in_block))
        self.shape = shape[:]
        self.data_type = data_type

    def astype(self, data_type: DataType) -> Tensor:
        if data_type == self.data_type:
            return self
        scales = self.ndarray[:, :, 0].view(np.float32)
        if self.data_type.have_addends:
            addends = self.ndarray[:, :, 1].view(np.float32)
        else:
            addends = None
        qweights = self.ndarray[:, :, -4:].reshape([self.shape[0], self.shape[1] // 8])

        dq = dequantize_q4(qweights, scales, addends, g_idx=None)
        return UnquantizedTensor(dq).astype(data_type)

    def to_ggml(self) -> 'GGMLQuantizedTensor':
        return self

    def permute(self, n_head: int) -> 'GGMLQuantizedTensor':
        return GGMLQuantizedTensor(permute(self.ndarray, n_head), self.shape, self.data_type)


GGMLCompatibleTensor = Union[UnquantizedTensor, GGMLQuantizedTensor]


class DeferredPermutedTensor(Tensor):
    def __init__(self, base: Tensor, n_head: int) -> None:
        self.base = base
        self.n_head = n_head
        self.data_type = self.base.data_type

    def astype(self, data_type: DataType) -> Tensor:
        return self.base.astype(data_type).permute(self.n_head)

    def to_ggml(self) -> GGMLCompatibleTensor:
        return self.base.to_ggml().permute(self.n_head)

    def permute(self, n_head: int) -> Tensor:
        raise Exception("shouldn't permute twice")


class GPTQForLLaMaQuantizedTensor(Tensor):
    def __init__(self, model: 'LazyModel', namebase: str) -> None:
        qweight = load_unquantized(model[f"{namebase}.qweight"], np.int32)
        scales = load_unquantized(model[f"{namebase}.scales"], np.float32, convert=True)

        bias = model.get(f"{namebase}.bias")
        if bias is not None:
            # Q4_1 does not support bias; good thing the bias is always all zeros.
            assert not np.any(load_unquantized(bias))

        if f"{namebase}.zeros" in model:
            zeros = load_unquantized(model[f"{namebase}.zeros"], np.float32)
        else:
            qzeros = load_unquantized(model[f"{namebase}.qzeros"], np.int32)
            assert qzeros.dtype == np.int32
            zeros = dequantize_q4(qzeros, scales, scales, g_idx=None)
            assert zeros.dtype == np.float32

        assert zeros.shape == scales.shape

        # Output is transposed compared to the input, and addends have their sign flipped.
        # Scales and zeros similarly must be transposed but only for newer
        # versions of GPTQ-for-LLaMa; the older versions can be identified by
        # having shape (n_embd, 1).
        qweight = qweight.T
        if scales.shape[1] != 1:
            scales = scales.T
            zeros = zeros.T

        # Output also has signs flipped for the addends.
        self.qweight = qweight
        self.scales = scales
        self.addends = -zeros

        self.g_idx: Optional[NDArray]
        if f"{namebase}.g_idx" in model:
            self.g_idx = load_unquantized(model[f"{namebase}.g_idx"], np.int32)
            assert self.g_idx.shape == (qweight.shape[1] * 8,)
        else:
            self.g_idx = None

        self.shape = [self.qweight.shape[0], self.qweight.shape[1] * 8]
        self.data_type = QuantizedDataType(groupsize=self.groupsize(), have_addends=True,
                                           have_g_idx=(self.g_idx is not None))

    def inspect(self, row: int, col: int) -> None:
        '''For debugging.'''
        qweight = (self.qweight[row, col // 8] >> (4 * (col & 7))) & 0xf
        if self.g_idx is not None:
            group = self.g_idx[col]
        else:
            group = int(col // self.groupsize())
        scale = self.scales[row, group]
        addend = self.addends[row, group]
        with np.printoptions(precision=None, suppress=True):
            print(f'scale:{scale} addend:{addend} qweight:{qweight}')
            print('possible values:', np.arange(16) * scale + addend)
            print('actual value:', qweight * scale + addend)

    def astype(self, data_type: DataType) -> Tensor:
        if isinstance(data_type, QuantizedDataType):
            assert self.g_idx is None and data_type.have_addends is True and data_type.have_g_idx is False
            return self.regroup(data_type.groupsize)

        dequantized = dequantize_q4(np.ascontiguousarray(self.qweight), self.scales, self.addends, self.g_idx)
        return UnquantizedTensor(dequantized).astype(data_type)

    def groupsize(self) -> int:
        assert self.addends.shape == self.scales.shape
        assert self.shape[1] % self.scales.shape[1] == 0
        return self.shape[1] // self.scales.shape[1]

    def regroup(self, new_groupsize: int = 32) -> 'GPTQForLLaMaQuantizedTensor':
        # Old versions of GPTQ-for-LLaMa shared scales and addends between all the
        # columns in a row.  Newer versions share them between every set of N
        # columns in a row, where N is the `groupsize` parameter, usually 128.  The
        # output format shares them between every set of 32 columns.  To handle
        # this, duplicate scales and addends for every smaller group.
        # (In the above, 'row' and 'column' are in the sense of the output.)
        assert self.g_idx is None
        old_groupsize = self.groupsize()
        assert old_groupsize >= new_groupsize and old_groupsize % new_groupsize == 0, old_groupsize
        ret = copy.copy(self)
        ret.addends = self.addends.repeat(old_groupsize // new_groupsize, axis=1)
        ret.scales = self.scales.repeat(old_groupsize // new_groupsize, axis=1)
        ret.data_type = QuantizedDataType(groupsize=new_groupsize, have_addends=True, have_g_idx=False)
        return ret

    def permute(self, n_head: int) -> Tensor:
        return DeferredPermutedTensor(self, n_head)

    def to_ggml(self) -> GGMLQuantizedTensor:
        # The output format looks like this:
        # For each row:
        #   For each group of 32 columns:
        #     - addend (float32, 4 bytes)
        #     - scale (float32, 4 bytes)
        #     - weights (int4 * 32, 16 bytes)

        if self.groupsize() != 32:
            raise Exception("should have been regrouped before converting to ggml")

        # Since the output format is mixed between integers and floats, we have
        # to hackily view the floats as int32s just so numpy will let us
        # concatenate them.
        addends_view = self.addends.view(dtype=np.int32)[:, :, np.newaxis]
        scales_view = self.scales.view(dtype=np.int32)[:, :, np.newaxis]

        # Split into groups of 4 columns (i.e. 32 columns of quantized data):
        grouped = self.qweight.reshape([self.qweight.shape[0], self.qweight.shape[1] // 4, 4])

        # And concatenate:
        grouped = np.concatenate([scales_view, addends_view, grouped], axis=2, casting='no')

        return GGMLQuantizedTensor(grouped, self.shape, DT_Q4_1)


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
        if isinstance(data_type, QuantizedDataType):
            if not isinstance(self.data_type, QuantizedDataType):
                raise Exception(f"Can't turn an unquantized tensor into a quantized type ({data_type})")
            if self.data_type.have_g_idx:
                sys.stderr.write("Error: Input uses the newer GPTQ-for-LLaMa format (using g_idx), which is not yet natively supported by GGML.  For now you can still convert this model by passing `--outtype f16` to dequantize, but that will result in a much larger output file for no quality benefit.\n")
                sys.exit(1)
            assert not data_type.have_g_idx and self.data_type.have_addends and data_type.have_addends


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


def permute_lazy(lazy_tensor: LazyTensor, n_head: int) -> LazyTensor:
    def load() -> Tensor:
        return lazy_tensor.load().permute(n_head)
    return LazyTensor(load, lazy_tensor.shape, lazy_tensor.data_type, f'permute({n_head}) ' + lazy_tensor.description)


def convert_transformers_to_orig(model: LazyModel) -> LazyModel:
    out: LazyModel = {}
    out["tok_embeddings.weight"] = model["model.embed_tokens.weight"]
    out["norm.weight"] = model["model.norm.weight"]
    out["output.weight"] = model["lm_head.weight"]

    n_head = model["model.layers.0.self_attn.q_proj.weight"].shape[1] // 128
    for i in itertools.count():
        if f"model.layers.{i}.self_attn.q_proj.weight" not in model:
            break
        out[f"layers.{i}.attention.wq.weight"] = permute_lazy(model[f"model.layers.{i}.self_attn.q_proj.weight"], n_head)
        out[f"layers.{i}.attention.wk.weight"] = permute_lazy(model[f"model.layers.{i}.self_attn.k_proj.weight"], n_head)
        out[f"layers.{i}.attention.wv.weight"] = model[f"model.layers.{i}.self_attn.v_proj.weight"]
        out[f"layers.{i}.attention.wo.weight"] = model[f"model.layers.{i}.self_attn.o_proj.weight"]

        out[f"layers.{i}.feed_forward.w1.weight"] = model[f"model.layers.{i}.mlp.gate_proj.weight"]
        out[f"layers.{i}.feed_forward.w2.weight"] = model[f"model.layers.{i}.mlp.down_proj.weight"]
        out[f"layers.{i}.feed_forward.w3.weight"] = model[f"model.layers.{i}.mlp.up_proj.weight"]

        out[f"layers.{i}.attention_norm.weight"] = model[f"model.layers.{i}.input_layernorm.weight"]
        out[f"layers.{i}.ffn_norm.weight"] = model[f"model.layers.{i}.post_attention_layernorm.weight"]
    return out


def handle_quantization(model: LazyModel) -> LazyModel:
    '''Convert a model with entries for 'foo.qweight', 'foo.scales', etc.
    (which resolve to UnquantizedTensors with the raw data) to one with entries
    for 'foo.weight' (which resolve to QuantizedTensors).
    '''
    def convert(name: str) -> Tuple[str, LazyTensor]:
        if name.endswith(".qweight"):
            namebase = name.rsplit('.', 1)[0]
            orig_name = namebase + ".weight"

            lazy_tensor = model[name]
            assert len(lazy_tensor.shape) == 2
            real_shape = [lazy_tensor.shape[1], lazy_tensor.shape[0] * 8]

            # Calculate type.  This replicates the logic in
            # GPTQForLLaMaQuantizedTensor (which is executed when the modelis
            # actually loaded).
            lazy_scales = model[f"{namebase}.scales"]
            scales_width = 1 if lazy_scales.shape[1] == 1 else lazy_scales.shape[0]
            assert real_shape[1] % scales_width == 0
            groupsize = real_shape[1] // scales_width
            have_g_idx = f"{namebase}.g_idx" in model
            data_type = QuantizedDataType(groupsize=groupsize, have_addends=True, have_g_idx=have_g_idx)

            def load() -> Tensor:
                return GPTQForLLaMaQuantizedTensor(model, namebase)

            return (orig_name, LazyTensor(load, real_shape, data_type, '[quantized]'))
        else:
            return (name, model[name])
    return dict(convert(name) for name in model)

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

    def lazy_rebuild_tensor_v2(storage: Any, storage_offset: Any, size: Any, stride: Any,  # pyright: ignore[reportSelfClsParameterName]
                               requires_grad: Any, backward_hooks: Any, metadata: Any = None) -> LazyTensor:
        assert isinstance(storage, LazyStorage)

        def load() -> UnquantizedTensor:
            elm_count = stride[0] * size[0]
            return UnquantizedTensor(storage.load(storage_offset, elm_count).reshape(size))
        description = f'pickled storage_offset={storage_offset} in {storage.description}'
        return LazyTensor(load, list(size), storage.kind.data_type, description)

    CLASSES: Dict[Any, Any] = {
        ('torch._utils', '_rebuild_tensor_v2'): lazy_rebuild_tensor_v2,
        ('torch', 'BFloat16Storage'): LazyStorageKind(DT_BF16),
        ('torch', 'HalfStorage'): LazyStorageKind(DT_F16),
        ('torch', 'FloatStorage'): LazyStorageKind(DT_F32),
        ('torch', 'IntStorage'): LazyStorageKind(DT_I32),
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


SAFETENSORS_DATA_TYPES: Dict[str, DataType] = {
    'F16': DT_F16,
    'F32': DT_F32,
    'I32': DT_I32,
}


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
    model = {name: convert(info) for (name, info) in header.items()}
    return ModelPlus(model=model, paths=[path], format='safetensors', vocab=None)


def must_read(fp: IO[bytes], length: int) -> bytes:
    ret = fp.read(length)
    if len(ret) < length:
        raise Exception("unexpectedly reached end of file")
    return ret


def lazy_load_ggml_file(fp: io.BufferedReader, path: Path) -> ModelPlus:
    magic = must_read(fp, 4)[::-1]
    if magic in (b'ggmf', b'ggjt'):
        version, = struct.unpack("i", must_read(fp, 4))
        assert version == 1
    else:
        assert magic == b'ggml'
        version = None
    n_vocab, n_embd, n_mult, n_head, n_layer, rot, file_type = struct.unpack('<7i', must_read(fp, 28))

    tokens: List[Tuple[bytes, float]] = []
    for i in range(n_vocab):
        if i == 32000:
            # HACK: GPT4All messed with the format without changing the magic
            # number.  Specifically, they changed the vocab section to contain
            # `n_vocab - 1` tokens instead of `n_vocab` (i.e. omitting the
            # extra pad token).  Try to detect if we're reading a file like
            # this.
            orig_pos = fp.tell()
            fp.seek(20, io.SEEK_CUR)
            is_gpt4all = fp.read(21) == b'tok_embeddings.weight'
            fp.seek(orig_pos)
            if is_gpt4all:
                break

        length, = struct.unpack("i", must_read(fp, 4))
        text = must_read(fp, length)
        if magic != b'ggml':
            score, = struct.unpack("f", must_read(fp, 4))
            tokens.append((text, score))
    vocab = GGMLVocab(tokens) if magic != b'ggml' else None

    model: LazyModel = {}
    # Use mmap for the actual data to avoid race conditions with the file offset.
    off = fp.raw.tell()
    mapped = memoryview(mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ))
    fp.raw.seek(off) # needed on Windows

    def read_tensor() -> None:  # this is a function so that variables captured in `load` don't change
        shape_len, name_len, ftype = struct.unpack("iii", must_read(fp, 12))
        assert 0 <= shape_len <= 3
        shape: List[int] = list(struct.unpack(f"{shape_len}i", must_read(fp, 4 * shape_len)))
        shape = shape[::-1]
        name = must_read(fp, name_len).decode('utf-8')
        data_type = FTYPE_TO_DATA_TYPE[ftype]

        if magic == b'ggjt':
            fp.seek((fp.tell() + 31) & -32)

        if data_type == DT_Q4_1:
            # See GPTQForLLaMaQuantizedTensor.ggml_ndarray()
            size = 24 * (shape[1] // 32) * shape[0]
        elif data_type == DT_Q4_0:
            size = 20 * (shape[1] // 32) * shape[0]
        else:
            numpy_dtype = DATA_TYPE_TO_NUMPY[data_type]
            elm_count = math.prod(shape)
            size = elm_count * numpy_dtype.itemsize
        offset = fp.tell()
        buf = mapped[offset:offset+size]
        fp.seek(size, io.SEEK_CUR)

        def load() -> Tensor:
            if isinstance(data_type, QuantizedDataType):
                ndarray = np.frombuffer(buf, dtype=np.uint32)
                return GGMLQuantizedTensor(ndarray, shape, data_type)
            else:
                return UnquantizedTensor(np.frombuffer(buf, dtype=numpy_dtype).reshape(shape))
        description = f'ggml offset={offset} type={data_type} path={path}'
        model[name] = LazyTensor(load, shape, data_type, description)

    while fp.read(1) != b'':
        fp.seek(-1, io.SEEK_CUR)
        read_tensor()

    return ModelPlus(model=model, paths=[path], format='ggml', vocab=vocab)


@functools.lru_cache(maxsize=None)
def lazy_load_file(path: Path) -> ModelPlus:
    fp = open(path, 'rb')
    first8 = fp.read(8)
    fp.seek(0)
    if first8[:2] == b'PK':
        # A zip file, i.e. PyTorch format
        return lazy_load_torch_file(fp, path)
    elif first8[2:4] == b'gg':
        # GGML format
        return lazy_load_ggml_file(fp, path)
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
        # GGMLVocab comes from the same file as the model so shouldn't mismatch:
        assert isinstance(vocab, SentencePieceVocab)
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
        self.fout = open(fname_out, "wb")

    def write_file_header(self, params: Params) -> None:
        self.fout.write(b"ggjt"[::-1])  # magic
        values = [
            1,  # file version
            params.n_vocab,
            params.n_embd,
            params.n_mult,
            params.n_head,
            params.n_layer,
            params.n_embd // params.n_head,  # rot (obsolete)
            params.file_type.value,
        ]
        self.fout.write(struct.pack("i" * len(values), *values))

    def write_tensor_header(self, name: str, shape: Sequence[int], data_type: DataType) -> None:
        sname = name.encode('utf-8')
        self.fout.write(struct.pack("iii", len(shape), len(sname), DATA_TYPE_TO_FTYPE[data_type]))
        self.fout.write(struct.pack("i" * len(shape), *shape[::-1]))
        self.fout.write(sname)
        self.fout.seek((self.fout.tell() + 31) & -32)

    def write_vocab(self, vocab: Vocab) -> None:
        for text, score in vocab.all_tokens():
            self.fout.write(struct.pack("i", len(text)))
            self.fout.write(text)
            self.fout.write(struct.pack("f", score))

    @staticmethod
    def write_vocab_only(fname_out: Path, vocab: Vocab) -> None:
        of = OutputFile(fname_out)
        params = Params(n_vocab=vocab.vocab_size, n_embd=0, n_mult=0,
                        n_head=1, n_layer=0, file_type=GGMLFileType.AllF32)
        of = OutputFile(fname_out)
        of.write_file_header(params)
        of.write_vocab(vocab)
        of.fout.close()

    @staticmethod
    def write_all(fname_out: Path, params: Params, model: LazyModel, vocab: Vocab) -> None:
        check_vocab_size(params, vocab)
        of = OutputFile(fname_out)
        of.write_file_header(params)
        print("Writing vocab...")
        of.write_vocab(vocab)

        def do_item(item: Tuple[str, LazyTensor]) -> NDArray:
            name, lazy_tensor = item
            return lazy_tensor.load().to_ggml().ndarray

        ndarrays = bounded_parallel_map(do_item, model.items(), concurrency=8)
        for i, ((name, lazy_tensor), ndarray) in enumerate(zip(model.items(), ndarrays)):
            size = ' x '.join(f"{dim:6d}" for dim in lazy_tensor.shape)
            padi = len(str(len(model)))
            print(f"[{i+1:{padi}d}/{len(model)}] Writing tensor {name:38s} | size {size:16} | type {lazy_tensor.data_type}")
            of.write_tensor_header(name, lazy_tensor.shape, lazy_tensor.data_type)
            ndarray.tofile(of.fout)
        of.fout.close()


def pick_output_type(model: LazyModel, output_type_str: Optional[str]) -> GGMLFileType:
    wq_type = model["layers.0.attention.wq.weight"].data_type
    if output_type_str == "f32" or (output_type_str is None and wq_type == DT_F32):
        return GGMLFileType.AllF32
    if output_type_str == "f16" or (output_type_str is None and wq_type == DT_F16):
        return GGMLFileType.MostlyF16
    if output_type_str == "q4_1" or (output_type_str is None and isinstance(wq_type, QuantizedDataType) and
                                     wq_type.have_addends):
        if isinstance(model["output.weight"].data_type, QuantizedDataType):
            return GGMLFileType.MostlyQ4_1
        else:
            return GGMLFileType.PerLayerIsQ4_1
    if output_type_str == "q4_0" or (output_type_str is None and isinstance(wq_type, QuantizedDataType)):
        return GGMLFileType.MostlyQ4_0
    name_to_type = {name: lazy_tensor.data_type for (name, lazy_tensor) in model.items()}
    raise Exception(f"Unexpected combination of types: {name_to_type}")


def do_necessary_conversions(model: LazyModel) -> LazyModel:
    model = handle_quantization(model)

    if "lm_head.weight" in model:
        model = convert_transformers_to_orig(model)
    model = filter_and_sort_tensors(model)

    return model


def convert_to_output_type(model: LazyModel, output_type: GGMLFileType) -> LazyModel:
    return {name: tensor.astype(output_type.type_for_tensor(name, tensor))
            for (name, tensor) in model.items()}


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
        globs = ["consolidated.00.pth", "pytorch_model-00001-of-*.bin", "*.pt"]
        files = [file for glob in globs for file in path.glob(glob)]
        if not files:
            # Try GGML too, but with lower priority, since if both a non-GGML
            # model and a GGML model exist in the same directory, we assume the
            # latter was converted from the former.
            files = list(path.glob("ggml-model*.bin*"))
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


def filter_and_sort_tensors(model: LazyModel) -> LazyModel:
    return {name: model[name] for name in TENSORS_LIST if name in model}


def load_vocab(path: Path) -> SentencePieceVocab:
    # Be extra-friendly and accept either a file or a directory.  Also, if it's
    # a directory, it might be the model directory, and tokenizer.model might
    # be in the parent of that.
    if path.is_dir():
        path2 = path / "tokenizer.model"
        # Use `.parent` instead of /.. to handle the symlink case better.
        path3 = path.parent / "tokenizer.model"
        if path2.exists():
            path = path2
        elif path3.exists():
            path = path3
        else:
            raise FileNotFoundError(f"Could not find tokenizer.model in {path} or its parent; if it's in another directory, pass the directory as --vocab-dir")
    added_tokens_path = path.parent / "added_tokens.json"
    print(f"Loading vocab file {path}")
    return SentencePieceVocab(path, added_tokens_path if added_tokens_path.exists() else None)


def default_outfile(model_paths: List[Path], params: Params) -> Path:
    namestr = {
        GGMLFileType.AllF32: "f32",
        GGMLFileType.MostlyF16: "f16",
        GGMLFileType.MostlyQ4_0: "q4_0",
        GGMLFileType.MostlyQ4_1: "q4_1",
        GGMLFileType.PerLayerIsQ4_1: "q4_1",
    }[params.file_type]
    ret = model_paths[0].parent / f"ggml-model-{namestr}.bin"
    if ret in model_paths:
        sys.stderr.write(f"Error: Default output path ({ret}) would overwrite the input.  Please explicitly specify a path using --outfile.\n")
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
    parser.add_argument("--dump", action="store_true", help="don't convert, just show what's in the model")
    parser.add_argument("--dump-single", action="store_true", help="don't convert, just show what's in a single model file")
    parser.add_argument("--vocab-only", action="store_true", help="extract only the vocab")
    parser.add_argument("--outtype", choices=["f32", "f16", "q4_1", "q4_0"], help="output format (default: based on input)")
    parser.add_argument("--vocab-dir", type=Path, help="directory containing tokenizer.model, if separate from model file")
    parser.add_argument("--outfile", type=Path, help="path to write to; default: based on input")
    parser.add_argument("model", type=Path, help="directory containing model file, or model file itself (*.pth, *.pt, *.bin)")
    args = parser.parse_args(args_in)

    vocab: Vocab
    if args.dump_single:
        model_plus = lazy_load_file(args.model)
        do_dump_model(model_plus)
    elif args.vocab_only:
        vocab = load_vocab(args.vocab_dir or args.model)
        assert args.outfile, "need --outfile if using --vocab-only"
        outfile = args.outfile
        OutputFile.write_vocab_only(outfile, vocab)
        print(f"Wrote {outfile}")
    else:
        model_plus = load_some_model(args.model)
        if args.dump:
            do_dump_model(model_plus)
            return
        if model_plus.vocab is not None and args.vocab_dir is None:
            vocab = model_plus.vocab
        else:
            vocab_dir = args.vocab_dir if args.vocab_dir else model_plus.paths[0].parent
            vocab = load_vocab(vocab_dir)
        model = model_plus.model
        model = do_necessary_conversions(model)
        output_type = pick_output_type(model, args.outtype)
        model = convert_to_output_type(model, output_type)
        params = Params.guessed(model, output_type)
        outfile = args.outfile or default_outfile(model_plus.paths, params)
        OutputFile.write_all(outfile, params, model, vocab)
        print(f"Wrote {outfile}")


if __name__ == '__main__':
    main()
