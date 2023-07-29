"""TODOs
1. Implement writers for known architectures, LLaMA in particular.
2. Add docstrings from the format specs.
3. After development is done, Convert it to a proper pip-installable Python package, and possibly move it to its own repo under ggml-org.
"""

import struct
import constants
from enum import IntEnum
from typing import Any, IO, List

import numpy as np


class GGMLQuantizationType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    # Q4_2 = 4 # support has been removed
    # Q4_3 = 5 # support has been removed
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


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9

    @staticmethod
    def get_type(val):
        if isinstance(val, str) or isinstance(val, bytes):
            return GGUFValueType.STRING
        elif isinstance(val, list):
            return GGUFValueType.ARRAY
        elif isinstance(val, float):
            return GGUFValueType.FLOAT32
        elif isinstance(val, bool):
            return GGUFValueType.BOOL
        else:
            return GGUFValueType.INT32


class GGUFWriter:
    def __init__(self, fout: IO):
        self.fout = fout
        self.offset_tensor = 0

    def write_header(self, tensor_count: int, metadata_kv_count: int):
        self.fout.write(struct.pack("<I", constants.GGUF_MAGIC))
        self.fout.write(struct.pack("<I", constants.GGUF_VERSION))
        self.fout.write(struct.pack("<I", tensor_count))
        self.fout.write(struct.pack("<I", metadata_kv_count))

    @classmethod
    def open(cls, path: str) -> "GGUFWriter":
        f = open(path, "wb")
        return cls(f)

    def write_key(self, key: str):
        self.write_val(key, GGUFValueType.STRING, write_vtype=False)

    def write_uint8(self, key: str, val: int):
        self.write_key(key)
        self.write_val(val, GGUFValueType.UINT8)

    def write_int8(self, key: str, val: int):
        self.write_key(key)
        self.write_val(val, GGUFValueType.INT8)

    def write_uint16(self, key: str, val: int):
        self.write_key(key)
        self.write_val(val, GGUFValueType.UINT16)

    def write_int16(self, key: str, val: int):
        self.write_key(key)
        self.write_val(val, GGUFValueType.INT16)

    def write_uint32(self, key: str, val: int):
        self.write_key(key)
        self.write_val(val, GGUFValueType.UINT32)

    def write_int32(self, key: str, val: int):
        self.write_key(key)
        self.write_val(val, GGUFValueType.INT32)

    def write_float32(self, key: str, val: float):
        self.write_key(key)
        self.write_val(val, GGUFValueType.FLOAT32)

    def write_bool(self, key: str, val: bool):
        self.write_key(key)
        self.write_val(val, GGUFValueType.BOOL)

    def write_string(self, key: str, val: str):
        self.write_key(key)
        self.write_val(val, GGUFValueType.STRING)

    def write_array(self, key: str, val: list):
        if not isinstance(val, list):
            raise ValueError("Value must be a list for array type")

        self.write_key(key)
        self.write_val(val, GGUFValueType.ARRAY)

    def write_val(self: str, val: Any, vtype: GGUFValueType = None, write_vtype: bool = True):
        if vtype is None:
            vtype = GGUFValueType.get_type(val)

        if write_vtype:
            self.fout.write(struct.pack("<I", vtype))

        if vtype == GGUFValueType.UINT8:
            self.fout.write(struct.pack("<B", val))
        elif vtype == GGUFValueType.INT8:
            self.fout.write(struct.pack("<b", val))
        elif vtype == GGUFValueType.UINT16:
            self.fout.write(struct.pack("<H", val))
        elif vtype == GGUFValueType.INT16:
            self.fout.write(struct.pack("<h", val))
        elif vtype == GGUFValueType.UINT32:
            self.fout.write(struct.pack("<I", val))
        elif vtype == GGUFValueType.INT32:
            self.fout.write(struct.pack("<i", val))
        elif vtype == GGUFValueType.FLOAT32:
            self.fout.write(struct.pack("<f", val))
        elif vtype == GGUFValueType.BOOL:
            self.fout.write(struct.pack("?", val))
        elif vtype == GGUFValueType.STRING:
            encoded_val = val.encode("utf8") if isinstance(val, str) else val
            self.fout.write(struct.pack("<I", len(encoded_val)))
            self.fout.write(encoded_val)
        elif vtype == GGUFValueType.ARRAY:
            ltype = set([GGUFValueType.get_type(item) for item in val])
            assert len(ltype) == 1, "All items in a GGUF array should be of the same type"
            self.fout.write(struct.pack("<I", ltype[0]))
            self.fout.write(struct.pack("<I", len(val)))
            for item in val:
                self.write_val(item, write_vtype=False)
        else:
            raise ValueError("Invalid GGUF metadata value type")

    @staticmethod
    def ggml_pad(x: int, n: int) -> int:
        return ((x + n - 1) // n) * n

    def write_tensor_info(self, name: str, tensor: np.ndarray):
        self.write_key(name)
        n_dims = len(tensor.shape)
        self.fout.write(struct.pack("<i", n_dims))
        for i in range(n_dims):
            self.fout.write(struct.pack("<i", tensor.shape[n_dims - 1 - i]))

        assert tensor.dtype in (np.float32, np.float16), "Only F32 and F16 tensors are supported for now"
        dtype = GGMLQuantizationType.F32 if tensor.dtype == np.float32 else GGMLQuantizationType.F16
        self.fout.write(struct.pack("<i", dtype))
        self.fout.write(struct.pack("<Q", self.offset_tensor))
        self.offset_tensor += GGUFWriter.ggml_pad(tensor.nbytes, constants.GGUF_DEFAULT_ALIGNMENT)

        self.flush()

    def write_tensor(self, tensor: np.ndarray):
        pad = GGUFWriter.ggml_pad(self.fout.tell(), constants.GGUF_DEFAULT_ALIGNMENT) - self.fout.tell()
        if pad != 0:
            self.fout.write(bytes([0] * pad))

        tensor.tofile(self.fout)
        pad = GGUFWriter.ggml_pad(tensor.nbytes, constants.GGUF_DEFAULT_ALIGNMENT) - tensor.nbytes
        if pad != 0:
            self.fout.write(bytes([0] * pad))

    def flush(self):
        self.fout.flush()

    def close(self):
        self.fout.close()

    def write_architecture(self, architecture: str):
        self.write_string(constants.KEY_GENERAL_ARCHITECTURE,
                          architecture)

    def write_author(self, author: str):
        self.write_string(constants.KEY_GENERAL_AUTHOR, author)

    def write_url(self, url: str):
        self.write_string(constants.KEY_GENERAL_URL, url)

    def write_description(self, description: str):
        self.write_string(constants.KEY_GENERAL_DESCRIPTION, description)

    def write_file_type(self, file_type: str):
        self.write_string(constants.KEY_GENERAL_FILE_TYPE, file_type)

    def write_source_url(self, url: str):
        self.write_string(constants.KEY_GENERAL_SOURCE_URL, url)

    def write_source_hf_repo(self, repo: str):
        self.write_string(constants.KEY_GENERAL_SOURCE_HF_REPO, repo)

    def write_name(self, name: str):
        self.write_string(constants.KEY_GENERAL_NAME, name)

    def write_quantization_version(self, quantization_version: GGMLQuantizationType):
        self.write_uint32(
            constants.KEY_GENERAL_QUANTIZATION_VERSION, quantization_version)

    def write_context_length(self, llm: str, length: int):
        self.write_uint32(
            constants.KEY_LLM_CONTEXT_LENGTH.format(llm=llm), length)

    def write_embedding_length(self, llm: str, length: int):
        self.write_uint32(
            constants.KEY_LLM_EMBEDDING_LENGTH.format(llm=llm), length)

    def write_layer_count(self, llm: str, length: int):
        self.write_uint32(
            constants.KEY_LLM_LAYER_COUNT.format(llm=llm), length)

    def write_feed_forward_length(self, llm: str, length: int):
        self.write_uint32(
            constants.KEY_LLM_FEED_FORWARD_LENGTH.format(llm=llm), length)

    def write_parallel_residual(self, llm: str, use: bool):
        self.write_bool(
            constants.KEY_LLM_USE_PARALLEL_RESIDUAL.format(llm=llm), use)

    def write_tensor_data_layout(self, llm: str, layout: str):
        self.write_string(
            constants.KEY_LLM_TENSOR_DATA_LAYOUT.format(llm=llm), layout)

    def write_head_count(self, llm: str, count: int):
        self.write_uint32(
            constants.KEY_ATTENTION_HEAD_COUNT.format(llm=llm), count)

    def write_head_count_kv(self, llm: str, count: int):
        self.write_uint32(
            constants.KEY_ATTENTION_HEAD_COUNT_KV.format(llm=llm), count)

    def write_max_alibi_bias(self, llm: str, bias: float):
        self.write_float32(
            constants.KEY_ATTENTION_MAX_ALIBI_BIAS.format(llm=llm), bias)

    def write_clamp_kqv(self, llm: str, value: float):
        self.write_float32(
            constants.KEY_ATTENTION_CLAMP_KQV.format(llm=llm), value)

    def write_rope_dimension_count(self, llm: str, count: int):
        self.write_uint32(
            constants.KEY_ROPE_DIMENSION_COUNT.format(llm=llm), count)

    def write_rope_scale(self, llm: str, value:  float):
        self.write_float32(constants.KEY_ROPE_SCALE.format(llm=llm), value)

    def write_tokenizer_model(self, model: str):
        self.write_string(constants.KEY_TOKENIZER_MODEL, model)

    def write_token_list(self, tokens: List):
        self.write_array(constants.KEY_TOKENIZER_LIST, tokens)

    def write_token_scores(self, scores: List:
        self.write_array(constants.KEY_TOKENIZER_SCORES, scores)


# Example usage:
if __name__ == "__main__":
    # Example usage with a file
    gguf_writer = GGUFWriter.open("example.gguf")
    gguf_writer.write_header(2, 3)

    gguf_writer.write_architecture("llama")
    gguf_writer.write_uint32("answer", 42)  # Write a 32-bit integer
    gguf_writer.write_float32("answer_in_float", 42.0)  # Write a 32-bit float
    tensor1 = np.ones((32,), dtype=np.float32) * 100.0
    tensor2 = np.ones((32,), dtype=np.float32) * 101.0
    gguf_writer.write_tensor_info("tensor0", tensor1)
    gguf_writer.write_tensor_info("tensor1", tensor2)
    gguf_writer.write_tensor(tensor1)
    gguf_writer.write_tensor(tensor2)

    gguf_writer.close()
