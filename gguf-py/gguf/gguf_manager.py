from __future__ import annotations

import os
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Sequence, Mapping
from string import ascii_letters, digits
from argparse import Namespace
from math import ceil
from collections import deque

import numpy as np

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

from .constants import (
    GGMLQuantizationType,
    GGUFEndian,
    GGUFValueType,
    Keys,
    RopeScalingType,
    PoolingType,
    TokenType,
)
from .gguf_writer import GGUFWriter


SHARD_NAME_FORMAT = "{:s}-{:05d}-of-{:05d}.gguf"

LLM_KV_SPLIT_NO = "split.no"
LLM_KV_SPLIT_COUNT = "split.count"
LLM_KV_SPLIT_TENSORS_COUNT = "split.tensors.count"

SplitTensorsPerFile: TypeAlias = deque[tuple[os.PathLike[str], deque[tuple[str, Any]], GGUFWriter]] # [(outfile name, [(tensor name, tensor data)] for each tensor in file, filewriter)]
KVTempData: TypeAlias = dict[str, tuple[Any, GGUFValueType]] # {key: (value, type)}
TensorTempData: TypeAlias = tuple[str, np.ndarray[Any, Any], GGMLQuantizationType] # (tensor name, tensor data, tensor dtype), aka LazyModel


class SplitStyle(IntEnum):
    NONE = 0
    TENSORS = 1
    SIZE = 2


class SplitArguments:
    split: bool
    dry_run: bool
    small_first_shard: bool
    split_max_tensors: int
    split_max_size: int
    split_style: SplitStyle

    def __init__(self, args: Namespace = None) -> None:
        self.split = args.split if args else False
        self.split_max_tensors = args.split_max_tensors if args else 0
        self.split_max_size = SplitStrategy.split_str_to_n_bytes(args.split_max_size) if args and args.split_max_size else 0
        self.dry_run = args.dry_run if args else False
        self.small_first_shard = not args.large_first_shard if args else False
        self.split_style = SplitStyle.NONE if not self.split or not args \
            else SplitStyle.TENSORS if self.split_max_tensors \
            else SplitStyle.SIZE


class SplitStrategy(deque):
    data: SplitTensorsPerFile

    def __init__(self, fname_out: os.PathLike[str], model: list[TensorTempData], arch: str,
                 split_arguments: SplitArguments, use_temp_file: bool = True, endianess: GGUFEndian = GGUFEndian.LITTLE,
    ):
        super().__init__()

        if split_arguments.split_style == SplitStyle.NONE:
            self.append((fname_out, model, GGUFWriter(fname_out, arch, use_temp_file=use_temp_file, endianess=endianess)))

        elif split_arguments.split_style == SplitStyle.TENSORS:
            total_shards = ceil(len(model) / split_arguments.split_max_tensors) + split_arguments.small_first_shard
            shard_files = [fname_out.with_name(SHARD_NAME_FORMAT.format(fname_out.stem, i + 1, total_shards)) for i in range(total_shards)]

            if split_arguments.small_first_shard:
                self.append((shard_files[0], None, GGUFWriter(shard_files[0], arch, use_temp_file=use_temp_file, endianess=endianess)))

            for i, shard in enumerate(shard_files[split_arguments.small_first_shard:]):
                start = i * split_arguments.split_max_tensors
                stop = min((i + 1) * split_arguments.split_max_tensors, len(model))
                self.append((shard, model[start:stop], GGUFWriter(shard, arch, use_temp_file=use_temp_file, endianess=endianess)))

        elif split_arguments.split_style == SplitStyle.SIZE:
            shards = deque()

            # we have to determine the shards first to determine how many shards there will be in total - two passes
            for i, shard in enumerate(model):
                if i == 0:
                    shards.append([shard])
                    continue
                if SplitStrategy.get_tensor_size(shard[1]) + sum(SplitStrategy.get_tensor_size(t[1]) for t in shards[-1]) > split_arguments.split_max_size:
                    shards.append([shard])
                else:
                    shards[-1].append(shard)

            total_shards = len(shards) + split_arguments.small_first_shard
            shard_offset = 1

            if split_arguments.small_first_shard:
                outname = fname_out.with_name(SHARD_NAME_FORMAT.format(fname_out.stem, shard_offset, total_shards))
                self.append((outname, None, GGUFWriter(outname, arch, use_temp_file=use_temp_file, endianess=endianess)))
                shard_offset += 1

            for i, shard in enumerate(shards):
                outname = fname_out.with_name(SHARD_NAME_FORMAT.format(fname_out.stem, i + shard_offset, total_shards))
                self.append((outname, deque(shard), GGUFWriter(outname, arch, use_temp_file=use_temp_file, endianess=endianess)))

    @staticmethod
    def get_tensor_size(tensor) -> int:
        # we don't have the LazyTensor class here from convert.py but we can try
        try:
            return tensor.data_type.elements_to_bytes(np.prod(tensor.shape))
        except AttributeError: # numpy ndarray[Any, Any]
            return tensor.nbytes
        except: # this should never happen
            raise ValueError(f"Invalid tensor type: {type(tensor)}")
    
    @staticmethod
    def split_str_to_n_bytes(split_str: str) -> int:
        if split_str.endswith("K"):
            n = int(split_str[:-1]) * 1024
        elif split_str.endswith("M"):
            n = int(split_str[:-1]) * 1024 * 1024
        elif split_str.endswith("G"):
            n = int(split_str[:-1]) * 1024 * 1024 * 1024
        elif split_str.isnumeric():
            n = int(split_str)
        else:
            raise ValueError(f"Invalid split size: {split_str}, must be a number, optionally followed by K, M, or G")

        if n <= 0:
            raise ValueError(f"Invalid split size: {split_str}, must be positive")

        return n

    @staticmethod
    def format_n_bytes_to_str(num: int) -> str:
        num = float(num)
        for unit in ("", "K", "M", "G"):
            if abs(num) < 1024.0:
                return f"{num:3.1f}{unit}"
            num /= 1024.0
        return f"{num:.1f}T - over 1TB, --split recommended"


# ideally this has most of the same signatures as GGUFWriter so it's nearly a drop-in replacement
class GGUFManager:
    kv_data: KVTempData
    tensors: deque[TensorTempData]
    split_arguments: SplitArguments
    split_strategy: SplitStrategy
    dtype: GGMLQuantizationType

    def __init__(self, path: os.PathLike[str] | str, arch: str, split_arguments: SplitArguments,
                 use_temp_file: bool = True, endianess: GGUFEndian = GGUFEndian.LITTLE
        ) -> None:
        self.arch = arch
        self.path = path
        self.endianess = endianess
        self.offset_tensor = 0
        self.kv_data = {}
        self.tensors = deque()
        self.split_strategy = None
        self.total_shards = None
        self.total_tensors = None
        self.use_temp_file = use_temp_file
        self.split_arguments = split_arguments

        self.add_architecture()

    # have to consolidate because we need to know kv data count and tensor count before we can write the header
    # and we need to write tensor info before we can write metadata
    # these all kinda show up around the same places anyway so it's not a huge deal?
    def write_to_file(self, meta_only: bool = False) -> None:

        # here is the first place you can assume you have all tensors written and you can establish the size of the file - so logic goes here
        self.total_tensors = len(self.tensors)
        total_size = sum(SplitStrategy.get_tensor_size(tensor[1]) for tensor in self.tensors)

        if self.split_arguments.split_max_tensors and self.total_tensors < self.split_arguments.split_max_tensors:
            print("Model has fewer tensors than the split threshold, not splitting")
            self.split_style = SplitStyle.NONE

        if self.split_arguments.split_max_size and total_size < self.split_arguments.split_max_size:
            print("Model has smaller size than the split threshold, not splitting")
            self.split_style = SplitStyle.NONE

        self.split_strategy = SplitStrategy(self.path, self.tensors, self.arch, self.split_arguments,
                                            use_temp_file=self.use_temp_file, endianess=self.endianess)
        del self.tensors
        self.total_shards = len(self.split_strategy)

        # only the first shard needs all the KV data
        for key, (value, etype) in self.kv_data.items():
            self.split_strategy[0][2].add_key(key)
            self.split_strategy[0][2].add_val(value, etype)

        if self.split_arguments.split_style != SplitStyle.NONE:
            for i, (_, _, writer) in enumerate(self.split_strategy):
                writer.add_uint16(LLM_KV_SPLIT_NO, i)
                writer.add_uint16(LLM_KV_SPLIT_COUNT, self.total_shards)
                writer.add_int32(LLM_KV_SPLIT_TENSORS_COUNT, self.total_tensors)

        # metadata/vocab only can write and return here
        if meta_only:
            for i, (_, _, writer) in enumerate(self.split_strategy):
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
            return
        
        # tensor writing code starts here

        print("\nWriting the following files:")
        for (shard_path, shard_tensors, _) in self.split_strategy:
            size = SplitStrategy.format_n_bytes_to_str(sum(SplitStrategy.get_tensor_size(t[1]) for t in shard_tensors)) if shard_tensors else "negligible - metadata only"
            print(f"  {shard_path}: n_tensors = {len(shard_tensors) if shard_tensors else 0}, total_size = {size}")

        if self.split_arguments.dry_run:
            print("\nDry run, not writing files")
            # instantiating GGUFWriters creates files
            for name, _, _ in self.split_strategy:
                os.remove(name)
            return

        # run add_tensor_info, write data, then write_tensor_data - taken from convert.py
        running_total = self.total_tensors
        ct = 0
        while True:
            try:
                (_, tensors, writer) = self.split_strategy.popleft()
            except IndexError:
                break

            shard_num_tensors = len(tensors) if tensors else 0
            
            if tensors:
                while True:
                    try:
                        (name, tensor, dtype) = tensors.popleft()
                    except IndexError:
                        break
                    writer.add_tensor(name, tensor, raw_dtype=dtype)

                print(f"Writing to shard {ct + 1}/{self.total_shards} with {shard_num_tensors}/{running_total} remaining tensors (of {self.total_tensors} total)")
                running_total -= shard_num_tensors

            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=True)
            ct = ct + 1
            del tensors

    def add_uint8(self, key: str, val: int) -> None:
        self.kv_data[key] = (val, GGUFValueType.UINT8)

    def add_int8(self, key: str, val: int) -> None:
        self.kv_data[key] = (val, GGUFValueType.INT8)

    def add_uint16(self, key: str, val: int) -> None:
        self.kv_data[key] = (val, GGUFValueType.UINT16)

    def add_int16(self, key: str, val: int) -> None:
        self.kv_data[key] = (val, GGUFValueType.INT16)

    def add_uint32(self, key: str, val: int) -> None:
        self.kv_data[key] = (val, GGUFValueType.UINT32)

    def add_int32(self, key: str, val: int) -> None:
        self.kv_data[key] = (val, GGUFValueType.INT32)

    def add_float32(self, key: str, val: float) -> None:
        self.kv_data[key] = (val, GGUFValueType.FLOAT32)

    def add_uint64(self, key: str, val: int) -> None:
        self.kv_data[key] = (val, GGUFValueType.UINT64)

    def add_int64(self, key: str, val: int) -> None:
        self.kv_data[key] = (val, GGUFValueType.INT64)

    def add_float64(self, key: str, val: float) -> None:
        self.kv_data[key] = (val, GGUFValueType.FLOAT64)

    def add_bool(self, key: str, val: bool) -> None:
        self.kv_data[key] = (val, GGUFValueType.BOOL)

    def add_string(self, key: str, val: str) -> None:
        if not val:
            return
        self.kv_data[key] = (val, GGUFValueType.STRING)

    def add_array(self, key: str, val: Sequence[Any]) -> None:
        if not isinstance(val, Sequence):
            raise ValueError(f'Expected a sequence for {key}, got {type(val)}')
        self.kv_data[key] = (val, GGUFValueType.ARRAY)

    def add_tensor(
        self, name: str, tensor: np.ndarray[Any, Any], raw_shape: Sequence[int] | None = None,
        raw_dtype: GGMLQuantizationType | None = None,
    ) -> None:
        if self.endianess == GGUFEndian.BIG:
            tensor.byteswap(inplace=True)

        # TODO reimplement temp file
        #if self.use_temp_file and self.temp_file is None:
        #    fp = tempfile.SpooledTemporaryFile(mode="w+b", max_size=256 * 1024 * 1024)
        #    fp.seek(0)
        #    self.temp_file = fp

        self.tensors.append((name, tensor, raw_dtype))

        #if self.temp_file is None:
        #    self.tensors.append(tensor)
        #    return

        #tensor.tofile(self.temp_file)
        #self.write_padding(self.temp_file, tensor.nbytes)

    def close(self) -> None:
        for _, _, writer in self.split_strategy:
            writer.close()

    def add_architecture(self) -> None:
        self.add_string(Keys.General.ARCHITECTURE, self.arch)

    def add_author(self, author: str) -> None:
        self.add_string(Keys.General.AUTHOR, author)

    def add_version(self, version: str) -> None:
        self.add_string(Keys.General.VERSION, version)

    def add_tensor_data_layout(self, layout: str) -> None:
        self.add_string(Keys.LLM.TENSOR_DATA_LAYOUT.format(arch=self.arch), layout)

    def add_url(self, url: str) -> None:
        self.add_string(Keys.General.URL, url)

    def add_description(self, description: str) -> None:
        self.add_string(Keys.General.DESCRIPTION, description)

    def add_licence(self, licence: str) -> None:
        self.add_string(Keys.General.LICENSE, licence)

    def add_source_url(self, url: str) -> None:
        self.add_string(Keys.General.SOURCE_URL, url)

    def add_source_hf_repo(self, repo: str) -> None:
        self.add_string(Keys.General.SOURCE_HF_REPO, repo)

    def add_file_type(self, ftype: int) -> None:
        self.add_uint32(Keys.General.FILE_TYPE, ftype)

    def add_name(self, name: str) -> None:
        self.add_string(Keys.General.NAME, name)

    def add_quantization_version(self, quantization_version: GGMLQuantizationType) -> None:
        self.add_uint32(Keys.General.QUANTIZATION_VERSION, quantization_version)

    def add_custom_alignment(self, alignment: int) -> None:
        self.data_alignment = alignment
        self.add_uint32(Keys.General.ALIGNMENT, alignment)

    def add_vocab_size(self, size: int) -> None:
        self.add_uint32(Keys.LLM.VOCAB_SIZE.format(arch=self.arch), size)

    def add_context_length(self, length: int) -> None:
        self.add_uint32(Keys.LLM.CONTEXT_LENGTH.format(arch=self.arch), length)

    def add_embedding_length(self, length: int) -> None:
        self.add_uint32(Keys.LLM.EMBEDDING_LENGTH.format(arch=self.arch), length)

    def add_block_count(self, length: int) -> None:
        self.add_uint32(Keys.LLM.BLOCK_COUNT.format(arch=self.arch), length)

    def add_feed_forward_length(self, length: int) -> None:
        self.add_uint32(Keys.LLM.FEED_FORWARD_LENGTH.format(arch=self.arch), length)

    def add_parallel_residual(self, use: bool) -> None:
        self.add_bool(Keys.LLM.USE_PARALLEL_RESIDUAL.format(arch=self.arch), use)

    def add_head_count(self, count: int) -> None:
        self.add_uint32(Keys.Attention.HEAD_COUNT.format(arch=self.arch), count)

    def add_head_count_kv(self, count: int) -> None:
        self.add_uint32(Keys.Attention.HEAD_COUNT_KV.format(arch=self.arch), count)

    def add_key_length(self, length: int) -> None:
        self.add_uint32(Keys.Attention.KEY_LENGTH.format(arch=self.arch), length)

    def add_value_length(self, length: int) -> None:
        self.add_uint32(Keys.Attention.VALUE_LENGTH.format(arch=self.arch), length)

    def add_max_alibi_bias(self, bias: float) -> None:
        self.add_float32(Keys.Attention.MAX_ALIBI_BIAS.format(arch=self.arch), bias)

    def add_clamp_kqv(self, value: float) -> None:
        self.add_float32(Keys.Attention.CLAMP_KQV.format(arch=self.arch), value)

    def add_logit_scale(self, value: float) -> None:
        self.add_float32(Keys.LLM.LOGIT_SCALE.format(arch=self.arch), value)

    def add_expert_count(self, count: int) -> None:
        self.add_uint32(Keys.LLM.EXPERT_COUNT.format(arch=self.arch), count)

    def add_expert_used_count(self, count: int) -> None:
        self.add_uint32(Keys.LLM.EXPERT_USED_COUNT.format(arch=self.arch), count)

    def add_layer_norm_eps(self, value: float) -> None:
        self.add_float32(Keys.Attention.LAYERNORM_EPS.format(arch=self.arch), value)

    def add_layer_norm_rms_eps(self, value: float) -> None:
        self.add_float32(Keys.Attention.LAYERNORM_RMS_EPS.format(arch=self.arch), value)

    def add_causal_attention(self, value: bool) -> None:
        self.add_bool(Keys.Attention.CAUSAL.format(arch=self.arch), value)

    def add_pooling_type(self, value: PoolingType) -> None:
        self.add_uint32(Keys.LLM.POOLING_TYPE.format(arch=self.arch), value.value)

    def add_rope_dimension_count(self, count: int) -> None:
        self.add_uint32(Keys.Rope.DIMENSION_COUNT.format(arch=self.arch), count)

    def add_rope_freq_base(self, value: float) -> None:
        self.add_float32(Keys.Rope.FREQ_BASE.format(arch=self.arch), value)

    def add_rope_scaling_type(self, value: RopeScalingType) -> None:
        self.add_string(Keys.Rope.SCALING_TYPE.format(arch=self.arch), value.value)

    def add_rope_scaling_factor(self, value: float) -> None:
        self.add_float32(Keys.Rope.SCALING_FACTOR.format(arch=self.arch), value)

    def add_rope_scaling_orig_ctx_len(self, value: int) -> None:
        self.add_uint32(Keys.Rope.SCALING_ORIG_CTX_LEN.format(arch=self.arch), value)

    def add_rope_scaling_finetuned(self, value: bool) -> None:
        self.add_bool(Keys.Rope.SCALING_FINETUNED.format(arch=self.arch), value)

    def add_ssm_conv_kernel(self, value: int) -> None:
        self.add_uint32(Keys.SSM.CONV_KERNEL.format(arch=self.arch), value)

    def add_ssm_inner_size(self, value: int) -> None:
        self.add_uint32(Keys.SSM.INNER_SIZE.format(arch=self.arch), value)

    def add_ssm_state_size(self, value: int) -> None:
        self.add_uint32(Keys.SSM.STATE_SIZE.format(arch=self.arch), value)

    def add_ssm_time_step_rank(self, value: int) -> None:
        self.add_uint32(Keys.SSM.TIME_STEP_RANK.format(arch=self.arch), value)

    def add_tokenizer_model(self, model: str) -> None:
        self.add_string(Keys.Tokenizer.MODEL, model)

    def add_tokenizer_pre(self, pre: str) -> None:
        self.add_string(Keys.Tokenizer.PRE, pre)

    def add_token_list(self, tokens: Sequence[str] | Sequence[bytes] | Sequence[bytearray]) -> None:
        self.add_array(Keys.Tokenizer.LIST, tokens)

    def add_token_merges(self, merges: Sequence[str] | Sequence[bytes] | Sequence[bytearray]) -> None:
        self.add_array(Keys.Tokenizer.MERGES, merges)

    def add_token_types(self, types: Sequence[TokenType] | Sequence[int]) -> None:
        self.add_array(Keys.Tokenizer.TOKEN_TYPE, types)

    def add_token_type_count(self, value: int) -> None:
        self.add_uint32(Keys.Tokenizer.TOKEN_TYPE_COUNT, value)

    def add_token_scores(self, scores: Sequence[float]) -> None:
        self.add_array(Keys.Tokenizer.SCORES, scores)

    def add_bos_token_id(self, id: int) -> None:
        self.add_uint32(Keys.Tokenizer.BOS_ID, id)

    def add_eos_token_id(self, id: int) -> None:
        self.add_uint32(Keys.Tokenizer.EOS_ID, id)

    def add_unk_token_id(self, id: int) -> None:
        self.add_uint32(Keys.Tokenizer.UNK_ID, id)

    def add_sep_token_id(self, id: int) -> None:
        self.add_uint32(Keys.Tokenizer.SEP_ID, id)

    def add_pad_token_id(self, id: int) -> None:
        self.add_uint32(Keys.Tokenizer.PAD_ID, id)

    def add_cls_token_id(self, id: int) -> None:
        self.add_uint32(Keys.Tokenizer.CLS_ID, id)

    def add_mask_token_id(self, id: int) -> None:
        self.add_uint32(Keys.Tokenizer.MASK_ID, id)

    def add_add_bos_token(self, value: bool) -> None:
        self.add_bool(Keys.Tokenizer.ADD_BOS, value)

    def add_add_eos_token(self, value: bool) -> None:
        self.add_bool(Keys.Tokenizer.ADD_EOS, value)

    def add_add_space_prefix(self, value: bool) -> None:
        self.add_bool(Keys.Tokenizer.ADD_PREFIX, value)

    def add_chat_template(self, value: str | Sequence[Mapping[str, str]]) -> None:
        if isinstance(value, list):
            template_default = None
            template_names = set()

            for choice in value:
                name = choice.get('name', '')
                template = choice.get('template')

                # Allowing non-alphanumerical characters in template name is probably not a good idea, so filter it
                name = ''.join((c if c in ascii_letters + digits else '_' for c in name))

                if name and template is not None:
                    if name == 'default':
                        template_default = template
                    else:
                        template_names.add(name)
                        self.add_string(Keys.Tokenizer.CHAT_TEMPLATE_N.format(name=name), template)

            if template_names:
                self.add_array(Keys.Tokenizer.CHAT_TEMPLATES, list(template_names))

            if template_default is None:
                return

            value = template_default

        self.add_string(Keys.Tokenizer.CHAT_TEMPLATE, value)

    def add_prefix_token_id(self, id: int) -> None:
        self.add_uint32(Keys.Tokenizer.PREFIX_ID, id)

    def add_suffix_token_id(self, id: int) -> None:
        self.add_uint32(Keys.Tokenizer.SUFFIX_ID, id)

    def add_middle_token_id(self, id: int) -> None:
        self.add_uint32(Keys.Tokenizer.MIDDLE_ID, id)

    def add_eot_token_id(self, id: int) -> None:
        self.add_uint32(Keys.Tokenizer.EOT_ID, id)