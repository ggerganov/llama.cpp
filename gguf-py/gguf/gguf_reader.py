from __future__ import annotations

import os
from collections import OrderedDict
from typing import TypeVar, NamedTuple

import numpy as np
import numpy.typing as npt

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Allow running file in package as a script.
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf.constants import (
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC,
    GGUF_VERSION,
    GGML_QUANT_SIZES,
    GGMLQuantizationType,
    GGUFValueType,
)

READER_SUPPORTED_VERSIONS = [2, GGUF_VERSION]


class ReaderField(NamedTuple):
    # Offset to start of this field.
    offset: int

    # Name of the field (not necessarily from file data).
    name: str

    # Data parts. Some types have multiple components, such as strings
    # that consist of a length followed by the string data.
    parts: [npt.NDArray] = []

    # Indexes into parts that we can call the actual data. For example
    # an array of strings will be populated with indexes to the actual
    # string data.
    data: [int] = [-1]

    types: [GGUFValueType] = []


class ReaderTensor(NamedTuple):
    name: str
    tensor_type: GGMLQuantizationType
    shape: npt.NDArray[np.uint32]
    n_elements: int
    n_bytes: int
    data_offset: int
    data: npt.NDArray
    field: ReaderField


class GGUFReader:
    byte_order: str = 'I'
    fields: 'OrderedDict[str, ReaderField]' = {}
    tensors: [ReaderTensor] = []
    alignment: int = GGUF_DEFAULT_ALIGNMENT

    _simple_value_map = {
        GGUFValueType.UINT8:   np.uint8,
        GGUFValueType.INT8:    np.int8,
        GGUFValueType.UINT16:  np.uint16,
        GGUFValueType.INT16:   np.int16,
        GGUFValueType.UINT32:  np.uint32,
        GGUFValueType.INT32:   np.int32,
        GGUFValueType.FLOAT32: np.float32,
        GGUFValueType.UINT64:  np.uint64,
        GGUFValueType.INT64:   np.int64,
        GGUFValueType.FLOAT64: np.float64,
        GGUFValueType.BOOL:    np.bool_,
    }

    _DT = TypeVar('T', bound = npt.DTypeLike)
    def _get(self, offset: int, dtype: _DT, count: int = 1, override_order: None | str = None) -> 'npt.NDArray[_DT]':
        end_offs = np.uint64(offset + np.uint64(dtype().nbytes * count))
        return (self.data[np.uint64(offset):end_offs]
            .view(dtype = dtype)[:count]
            .newbyteorder(override_order or self.byte_order))

    def _push_field(self, field: ReaderField, skip_sum: bool = False) -> int:
        if field.name in self.fields:
            raise KeyError(f'Duplicate {field.name} already in list at offset {field.offset}')
        self.fields[field.name] = field
        return 0 if skip_sum else sum(part.nbytes for part in field.parts)

    def _get_str(self, offset: int) -> (npt.NDArray[np.uint64], npt.NDArray[np.uint8]):
        slen = self._get(offset, np.uint64)
        return (slen, self._get(offset + 8, np.uint8, slen[0]))

    def _get_field_parts(self, orig_offs: int, raw_type: int) -> (int, [np.NDArray], [int], [GGUFValueType]):
        offs = orig_offs
        types = []
        gtype = GGUFValueType(raw_type)
        types.append(gtype)
        # Handle strings.
        if gtype == GGUFValueType.STRING:
            parts = list(self._get_str(offs))
            size = sum(part.nbytes for part in parts)
            return (size, parts, [1], types)
        # Check if it's a simple scalar type.
        nptype = self._simple_value_map.get(gtype)
        if nptype is not None:
            val = self._get(offs, nptype)
            return (val.nbytes, [val], [0], types)
        # Handle arrays.
        if gtype == GGUFValueType.ARRAY:
            raw_itype = self._get(offs, np.uint32)
            offs += raw_itype.nbytes
            alen = self._get(offs, np.uint64)
            offs += alen.nbytes
            parts = [raw_itype, alen]
            data_idxs = []
            for idx in range(alen[0]):
                curr_size, curr_parts, curr_idxs, curr_types = self._get_field_parts(offs, raw_itype[0])
                if idx == 0:
                    types += curr_types
                idxs_offs = len(parts)
                parts += curr_parts
                data_idxs += (idx + idxs_offs for idx in curr_idxs)
                offs += curr_size
            return (offs - orig_offs, parts, data_idxs, types)
        # We can't deal with this one.
        raise ValueError('Unknown/unhandled field type {gtype}')

    def _get_tensor(self, orig_offs: int) -> ReaderField:
        offs = np.uint64(orig_offs)
        name_len, name_data = self._get_str(offs)
        offs += name_len.nbytes + name_data.nbytes
        n_dims = self._get(offs, np.uint32)
        offs += n_dims.nbytes
        dims = self._get(offs, np.uint64, n_dims[0])
        offs += dims.nbytes
        raw_dtype = self._get(offs, np.uint32)
        offs += raw_dtype.nbytes
        offset_tensor = self._get(offs, np.uint64)
        offs += offset_tensor.nbytes
        return ReaderField(
            orig_offs,
            str(name_data, encoding = 'utf-8'),
            [name_len, name_data, n_dims, dims, raw_dtype, offset_tensor],
            [1, 3, 4, 5],
        )

    def _build_fields(self, offs, count) -> int:
        for _ in range(count):
            orig_offs = offs
            kv_klen, kv_kdata = self._get_str(offs)
            offs += kv_klen.nbytes + kv_kdata.nbytes
            raw_kv_type = self._get(offs, np.uint32)
            offs += raw_kv_type.nbytes
            parts = [kv_klen, kv_kdata, raw_kv_type]
            idxs_offs = len(parts)
            field_size, field_parts, field_idxs, field_types = self._get_field_parts(offs, raw_kv_type[0])
            parts += field_parts
            self._push_field(ReaderField(
                orig_offs,
                str(kv_kdata, encoding = 'utf-8'),
                parts,
                list(idx + idxs_offs for idx in field_idxs),
                field_types,
            ), skip_sum = True)
            offs += field_size
        return offs

    def _build_tensors_fields(self, offs, count) -> (int, [ReaderField]):
        tensor_fields = []
        for _ in range(count):
            field = self._get_tensor(offs)
            offs += sum(part.nbytes for part in field.parts)
            tensor_fields.append(field)
        return (offs, tensor_fields)

    def _build_tensors(self, start_offs: int, fields: [ReaderField]) -> None:
        tensors = []
        for field in fields:
            _name_len, name_data, _n_dims, dims, raw_dtype, offset_tensor = field.parts
            ggml_type = GGMLQuantizationType(raw_dtype[0])
            n_elems = np.prod(dims)
            block_size, type_size = GGML_QUANT_SIZES[ggml_type]
            n_bytes = np.uint64(np.uint64(n_elems) * np.uint64(type_size)) // np.uint64(block_size)
            data_offs = start_offs + offset_tensor[0]
            if ggml_type == GGMLQuantizationType.F32:
                item_count = n_elems
                item_type = np.float32
            elif ggml_type == GGMLQuantizationType.F16:
                item_count = n_elems
                item_type = np.float16
            else:
                item_count = n_bytes
                item_type = np.uint8
            tensors.append(ReaderTensor(
                name = str(name_data, encoding = 'utf-8'),
                tensor_type = ggml_type,
                shape = dims,
                n_elements = n_elems,
                n_bytes = n_bytes,
                data_offset = data_offs,
                data = self._get(data_offs, item_type, item_count),
                field = field,
            ))
        self.tensors = tensors


    def __init__(self, path: os.PathLike[str] | str, mode: str = 'r') -> None:
        self.data = np.memmap(path, mode = mode)
        offs = 0
        if self._get(offs, np.uint32, override_order = '<')[0] != GGUF_MAGIC:
            raise ValueError('GGUF magic invalid')
        offs += 4
        temp = self._get(offs, np.uint32)
        if temp[0] > 2000:
            self.byte_order = 'S'
            temp = temp.newbyteorder(self.byte_order)
        version = temp[0]
        if version not in READER_SUPPORTED_VERSIONS:
            raise ValueError(f'Sorry, file appears to be version {version} which we cannot handle')
        offs += self._push_field(ReaderField(offs, 'GGUF.version', [temp], [0], [GGUFValueType.UINT32]))
        temp = self._get(offs, np.uint64, 2)
        offs += self._push_field(ReaderField(offs, 'GGUF.tensor_count', [temp[:1]], [0], [GGUFValueType.UINT64]))
        offs += self._push_field(ReaderField(offs, 'GGUF.kv_count', [temp[1:]], [0], [GGUFValueType.UINT64]))
        tensor_count, kv_count = temp
        offs = self._build_fields(offs, kv_count)
        offs, tensors_fields = self._build_tensors_fields(offs, tensor_count)
        new_align = self.fields.get('general.alignment')
        if new_align is not None:
            if new_align.types != [GGUFValueType.UINT64]:
                raise ValueError('Bad type for general.alignment field')
            self.alignment = new_align.parts[-1]
        padding = offs % self.alignment
        if padding != 0:
            offs += self.alignment - padding
        self._build_tensors(offs, tensors_fields)


# Example usage:
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('gguf_reader: Error: Specify an input file', file = sys.stderr)
        sys.exit(1)
    print(f'* Loading: {sys.argv[1]}')
    reader = GGUFReader(sys.argv[1], 'r')
    print(f'\n* Dumping {len(reader.fields)} key/value pair(s)')
    for n, field in enumerate(reader.fields.values(), 1):
        if len(field.types) == 0:
            pretty_type = 'N/A'
        elif field.types[0] == GGUFValueType.ARRAY:
            nest_count = len(field.types) - 1
            pretty_type = '[' * nest_count + str(field.types[-1].name) + ']' * nest_count
        else:
            pretty_type = str(field.types[-1].name)
        print(f'  {n:5}: {pretty_type:10} | {len(field.data):8} | {field.name}', end = '')
        if len(field.types) == 1:
            curr_type = field.types[0]
            if curr_type == GGUFValueType.STRING:
                print(' = {0}'.format(repr(str(field.parts[-1], encoding='utf8')[:60])), end = '')
            elif field.types[0] in reader._simple_value_map:
                print(' = {0}'.format(field.parts[-1][0]), end = '')
        print()

    print(f'\n* Dumping {len(reader.tensors)} tensor(s)')
    for n, tensor in enumerate(reader.tensors, 1):

        prettydims = ', '.join('{0:5}'.format(d) for d in list(tensor.shape) + [1] * (4 - len(tensor.shape)))
        print(f'  {n:5}: {tensor.n_elements:10} | {prettydims} | {tensor.tensor_type.name:7} | {tensor.name}')
