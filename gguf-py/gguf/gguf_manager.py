from __future__ import annotations

import os
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Sequence
from argparse import Namespace
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

from .constants import (
    GGMLQuantizationType,
    GGUFEndian,
    GGUFValueType
)
from .gguf_writer import GGUFWriter, WriterState
from .constants import Keys


SHARD_NAME_FORMAT = "{:s}-{:05d}-of-{:05d}.gguf"
METADATA_ONLY_INDICATOR = -1

KVTempData: TypeAlias = dict[str, tuple[Any, GGUFValueType]] # {key: (value, type)}
TensorTempData: TypeAlias = tuple[str, np.ndarray[Any, Any], GGMLQuantizationType] # (tensor name, tensor data, tensor dtype)


@dataclass
class Shard:
    path: Path
    tensor_count: int
    size: int
    tensors: deque[TensorTempData]


class SplitStyle(IntEnum):
    NONE = 0
    TENSORS = 1
    SIZE = 2


class SplitArguments:
    def __init__(self, args: Namespace) -> None:
        self.split = args.split
        self.split_max_tensors = args.split_max_tensors if args.split else 0
        self.split_max_size = GGUFManager.split_str_to_n_bytes(args.split_max_size) if args.split and args.split_max_size else 0
        self.split_style = SplitStyle.NONE if not self.split \
            else SplitStyle.TENSORS if self.split_max_tensors \
            else SplitStyle.SIZE
        self.dry_run = args.dry_run
        self.small_first_shard = args.small_first_shard


class GGUFManager(GGUFWriter):
    kv_data: KVTempData
    split_arguments: SplitArguments
    shards: list[Shard]
    shard_writers: list[GGUFWriter]

    def __init__(self, path: os.PathLike[str] | str, arch: str, split_arguments: SplitArguments,
                 use_temp_file: bool = True, endianess: GGUFEndian = GGUFEndian.LITTLE
        ) -> None:
        # we intentionally don't call superclass constructor
        self.arch = arch
        self.path = Path(path)
        self.endianess = endianess
        self.kv_data = {}
        self.shards = []
        self.shard_writers = []
        self.total_tensors = 0
        self.use_temp_file = use_temp_file
        self.split_arguments = split_arguments
        self.recent_key = None
        self.state = WriterState.EMPTY

        if self.split_arguments.small_first_shard:
            self.shards.append(Shard(Path(), 0, METADATA_ONLY_INDICATOR, deque()))

    def init_shards(self) -> None:
        self.total_tensors = sum(shard.tensor_count for shard in self.shards)
        total_size = sum(shard.size for shard in self.shards)

        # check if we need to split
        if self.split_arguments.split_max_tensors and self.total_tensors < self.split_arguments.split_max_tensors:
            print("Model has fewer tensors than the split threshold, not splitting")
            self.split_style = SplitStyle.NONE

        if self.split_arguments.split_max_size and total_size < self.split_arguments.split_max_size:
            print("Model has smaller size than the split threshold, not splitting")
            self.split_style = SplitStyle.NONE

        # no shards are created when writing vocab so make one
        if not self.shards:
            self.shards.append(Shard(Path(), 0, METADATA_ONLY_INDICATOR, deque()))

        # format shard names
        if len(self.shards) == 1:
            self.shards[0].path = self.path
        else:
            for i in range(len(self.shards)):
                self.shards[i].path = self.path.with_name(SHARD_NAME_FORMAT.format(self.path.stem, i + 1, len(self.shards)))

        # print shard info
        print("\nWriting the following files:")
        for shard in self.shards:
            print(f"  {shard.path}: n_tensors = {shard.tensor_count}, total_size = {GGUFManager.format_n_bytes_to_str(shard.size)}")
        print()

        if self.split_arguments.dry_run:
            print("\nDry run, not writing files")
            exit()

        # we don't want to initialize GGUFWriters until now because they create files
        for i, shard in enumerate(self.shards):
            # add_architecture is used for consistency - examples/gguf_split doesn't add arch to all shards
            writer = GGUFWriter(shard.path, self.arch, use_temp_file=self.use_temp_file,
                                endianess=self.endianess, add_architecture=(i == 0))

            # only the first shard needs all the KV data
            if i == 0:
                for key, (value, etype) in self.kv_data.items():
                    writer.add_key(key)
                    writer.add_val(value, etype)

            # add split metadata unless it's one file - small first shard splits even with SplitStyle.NONE
            if self.split_arguments.split_style != SplitStyle.NONE or self.split_arguments.small_first_shard:
                writer.add_uint16(Keys.Split.LLM_KV_SPLIT_NO, i)
                writer.add_uint16(Keys.Split.LLM_KV_SPLIT_COUNT, len(self.shards))
                writer.add_int32(Keys.Split.LLM_KV_SPLIT_TENSORS_COUNT, self.total_tensors)

            # add tensors, deque popleft() ensures references to eager tensors are not kept
            while True:
                try:
                    (name, tensor, dtype) = shard.tensors.popleft()
                    writer.add_tensor(name, tensor, raw_dtype=dtype)
                except:
                    break

            self.shard_writers.append(writer)

    def write_header_to_file(self) -> None:
        if self.state is not WriterState.EMPTY:
            raise ValueError(f'Expected GGUFManager state to be EMPTY, got {self.state}')

        for writer in self.shard_writers:
            writer.write_header_to_file()

        self.state = WriterState.HEADER

    def write_kv_data_to_file(self) -> None:
        if self.state is not WriterState.HEADER:
            raise ValueError(f'Expected GGUFManager state to be HEADER, got {self.state}')
        
        for writer in self.shard_writers:
            writer.write_kv_data_to_file()

        self.state = WriterState.KV_DATA

    def write_tensors_to_file(self, *, progress: bool = False) -> None:
        if self.state is not WriterState.KV_DATA:
            raise ValueError(f'Expected GGUFManager state to be KV_DATA, got {self.state}')

        running_total = self.total_tensors
        for i in range(len(self.shard_writers)):
            writer = self.shard_writers[i]
            is_metadata = writer.ti_data_count == 0
            if is_metadata:
                print(f"Writing to shard {i + 1}/{len(self.shards)} with metadata only")
            else:
                print(f"Writing to shard {i + 1}/{len(self.shards)} with {writer.ti_data_count}/{running_total} remaining tensors (of {self.total_tensors} total)")
            running_total -= writer.ti_data_count
            writer.write_tensors_to_file(progress=(progress and not is_metadata))
            del writer

        self.state = WriterState.TI_DATA

    # override add_key, add_val to handle kv data separately
    def add_key(self, key: str) -> None:
        self.recent_key = key
    
    def add_val(self, val: Any, vtype: GGUFValueType | None = None, add_vtype: bool = True) -> None:
        if self.recent_key is None:
            raise ValueError("No key set for value")
        self.kv_data[self.recent_key] = (val, vtype)

    # need to handle arrays separately
    def add_array(self, key: str, val: Sequence[Any]) -> None:
        if not isinstance(val, Sequence):
            raise ValueError(f'Expected a sequence for {key}, got {type(val)}')
        self.kv_data[key] = (val, GGUFValueType.ARRAY)

    def add_tensor(
        self, name: str, tensor: np.ndarray[Any, Any], raw_shape: Sequence[int] | None = None,
        raw_dtype: GGMLQuantizationType | None = None,
    ) -> None:
        # we build splits as tensors are added so we need logic to figure out when to split
        # logic is all in the conditional because it short-circuits, otherwise accessing self.shards[-1] would throw an error

        # create a first shard to start it off
        if (len(self.shards) == self.split_arguments.small_first_shard \
            # or split when over tensor limit
            or (self.split_arguments.split_style == SplitStyle.TENSORS \
                and self.shards[-1].tensor_count >= self.split_arguments.split_max_tensors) \
            # or split when over size limit
            or (self.split_arguments.split_style == SplitStyle.SIZE \
                and self.shards[-1].size + GGUFManager.get_tensor_size(tensor) > self.split_arguments.split_max_size)):

            # we fill in the name later when we know how many shards there are
            self.shards.append(Shard(Path(), 1, GGUFManager.get_tensor_size(tensor), deque([(name, tensor, raw_dtype)])))
        else:
            self.shards[-1].tensor_count += 1
            self.shards[-1].size += GGUFManager.get_tensor_size(tensor)
            self.shards[-1].tensors.append((name, tensor, raw_dtype))

    def close(self) -> None:
        for writer in self.shard_writers:
            writer.close()

    @staticmethod
    def get_tensor_size(tensor) -> int:
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
        if num == METADATA_ONLY_INDICATOR:
            return "negligible - metadata only"
        fnum = float(num)
        for unit in ("", "K", "M", "G"):
            if abs(fnum) < 1024.0:
                return f"{fnum:3.1f}{unit}"
            fnum /= 1024.0
        return f"{fnum:.1f}T - over 1TB, --split recommended"