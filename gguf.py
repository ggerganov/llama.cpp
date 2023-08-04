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
import sys

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
        else:
            print("Unknown type: "+str(type(val)))
            sys.exit()


class GGUFWriter:
    def __init__(self, fout: IO):
        self.fout = fout
        self.offset_tensor = 0
        self.kv_data = b""
        self.kv_data_count = 0
        self.ti_data = b""
        self.ti_data_count = 0

    def write_header_to_file(self):
        self.fout.write(struct.pack("<I", constants.GGUF_MAGIC))
        self.fout.write(struct.pack("<I", constants.GGUF_VERSION))
        self.fout.write(struct.pack("<I", self.ti_data_count))
        self.fout.write(struct.pack("<I", self.kv_data_count))
        self.flush()
#        print("tensors " + str(self.ti_data_count) + " kv " + str(self.kv_data_count))

    def write_kv_data_to_file(self):
        self.fout.write(self.kv_data)
        self.flush()

    def write_ti_data_to_file(self):
        self.fout.write(self.ti_data)
        self.flush()

    @classmethod
    def open(cls, path: str) -> "GGUFWriter":
        f = open(path, "wb")
        return cls(f)

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

    def add_bool(self, key: str, val: bool):
        self.add_key(key)
        self.add_val(val, GGUFValueType.BOOL)

    def add_string(self, key: str, val: str):
        self.add_key(key)
        self.add_val(val, GGUFValueType.STRING)

    def add_array(self, key: str, val: list):
        if not isinstance(val, list):
            raise ValueError("Value must be a list for array type")

        self.add_key(key)
        self.add_val(val, GGUFValueType.ARRAY)

    def add_val(self: str, val: Any, vtype: GGUFValueType = None, add_vtype: bool = True):
        if vtype is None:
            vtype = GGUFValueType.get_type(val)

        if add_vtype:
            self.kv_data += struct.pack("<I", vtype)
            self.kv_data_count += 1;

        if vtype == GGUFValueType.UINT8:
            self.kv_data += struct.pack("<B", val)
        elif vtype == GGUFValueType.INT8:
            self.kv_data += struct.pack("<b", val)
        elif vtype == GGUFValueType.UINT16:
            self.kv_data += struct.pack("<H", val)
        elif vtype == GGUFValueType.INT16:
            self.kv_data += struct.pack("<h", val)
        elif vtype == GGUFValueType.UINT32:
            self.kv_data += struct.pack("<I", val)
        elif vtype == GGUFValueType.INT32:
            self.kv_data += struct.pack("<i", val)
        elif vtype == GGUFValueType.FLOAT32:
            self.kv_data += struct.pack("<f", val)
        elif vtype == GGUFValueType.BOOL:
            self.kv_data += struct.pack("?", val)
        elif vtype == GGUFValueType.STRING:
            encoded_val = val.encode("utf8") if isinstance(val, str) else val
            self.kv_data += struct.pack("<I", len(encoded_val))
            self.kv_data += encoded_val
        elif vtype == GGUFValueType.ARRAY:
            ltype = set([GGUFValueType.get_type(item) for item in val])
            assert len(ltype) == 1, "All items in a GGUF array should be of the same type"
            self.kv_data += struct.pack("<I", list(ltype)[0])
            self.kv_data += struct.pack("<I", len(val))
            for item in val:
                self.add_val(item, add_vtype=False)
        else:
            raise ValueError("Invalid GGUF metadata value type")

    @staticmethod
    def ggml_pad(x: int, n: int) -> int:
        return ((x + n - 1) // n) * n

    def add_tensor_info(self, name: str, tensor: np.ndarray):
        encoded_name = name.encode("utf8")
        self.ti_data += struct.pack("<I", len(encoded_name))
        self.ti_data += encoded_name
        n_dims = len(tensor.shape)
        self.ti_data += struct.pack("<I", n_dims)
        for i in range(n_dims):
            self.ti_data += struct.pack("<I", tensor.shape[n_dims - 1 - i])

        assert tensor.dtype in (np.float32, np.float16), "Only F32 and F16 tensors are supported for now"
        dtype = GGMLQuantizationType.F32 if tensor.dtype == np.float32 else GGMLQuantizationType.F16
        self.ti_data += struct.pack("<I", dtype)
        self.ti_data += struct.pack("<Q", self.offset_tensor)
        self.offset_tensor += GGUFWriter.ggml_pad(tensor.nbytes, constants.GGUF_DEFAULT_ALIGNMENT)
        self.ti_data_count += 1

    def write_tensor_to_file(self, tensor: np.ndarray):
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

    def add_architecture(self, architecture: str):
        self.add_string(constants.KEY_GENERAL_ARCHITECTURE,
                          architecture)

    def add_author(self, author: str):
        self.add_string(constants.KEY_GENERAL_AUTHOR, author)

    def add_url(self, url: str):
        self.add_string(constants.KEY_GENERAL_URL, url)

    def add_description(self, description: str):
        self.add_string(constants.KEY_GENERAL_DESCRIPTION, description)

    def add_file_type(self, file_type: str):
        self.add_string(constants.KEY_GENERAL_FILE_TYPE, file_type)

    def add_source_url(self, url: str):
        self.add_string(constants.KEY_GENERAL_SOURCE_URL, url)

    def add_source_hf_repo(self, repo: str):
        self.add_string(constants.KEY_GENERAL_SOURCE_HF_REPO, repo)

    def add_name(self, name: str):
        self.add_string(constants.KEY_GENERAL_NAME, name)

    def add_quantization_version(self, quantization_version: GGMLQuantizationType):
        self.add_uint32(
            constants.KEY_GENERAL_QUANTIZATION_VERSION, quantization_version)

    def add_custom_alignment(self, alignment: int):
        self.add_uint32(constants.KEY_GENERAL_ALIGNMENT, alignment)

    def add_context_length(self, llm: str, length: int):
        self.add_uint32(
            constants.KEY_LLM_CONTEXT_LENGTH.format(llm=llm), length)

    def add_embedding_length(self, llm: str, length: int):
        self.add_uint32(
            constants.KEY_LLM_EMBEDDING_LENGTH.format(llm=llm), length)

    def add_layer_count(self, llm: str, length: int):
        self.add_uint32(
            constants.KEY_LLM_LAYER_COUNT.format(llm=llm), length)

    def add_feed_forward_length(self, llm: str, length: int):
        self.add_uint32(
            constants.KEY_LLM_FEED_FORWARD_LENGTH.format(llm=llm), length)

    def add_parallel_residual(self, llm: str, use: bool):
        self.add_bool(
            constants.KEY_LLM_USE_PARALLEL_RESIDUAL.format(llm=llm), use)

    def add_tensor_data_layout(self, llm: str, layout: str):
        self.add_string(
            constants.KEY_LLM_TENSOR_DATA_LAYOUT.format(llm=llm), layout)

    def add_head_count(self, llm: str, count: int):
        self.add_uint32(
            constants.KEY_ATTENTION_HEAD_COUNT.format(llm=llm), count)

    def add_head_count_kv(self, llm: str, count: int):
        self.add_uint32(
            constants.KEY_ATTENTION_HEAD_COUNT_KV.format(llm=llm), count)

    def add_max_alibi_bias(self, llm: str, bias: float):
        self.add_float32(
            constants.KEY_ATTENTION_MAX_ALIBI_BIAS.format(llm=llm), bias)

    def add_clamp_kqv(self, llm: str, value: float):
        self.add_float32(
            constants.KEY_ATTENTION_CLAMP_KQV.format(llm=llm), value)

    def add_layer_norm_eps(self, llm: str, value: float):
        self.add_float32(
            constants.KEY_ATTENTION_LAYERNORM_EPS.format(llm=llm), value)

    def add_layer_norm_rms_eps(self, llm: str, value: float):
        self.add_float32(
            constants.KEY_ATTENTION_LAYERNORM_RMS_EPS.format(llm=llm), value)

    def add_rope_dimension_count(self, llm: str, count: int):
        self.add_uint32(
            constants.KEY_ROPE_DIMENSION_COUNT.format(llm=llm), count)

    def add_rope_scale(self, llm: str, value:  float):
        self.add_float32(constants.KEY_ROPE_SCALE.format(llm=llm), value)

    def add_tokenizer_model(self, model: str):
        self.add_string(constants.KEY_TOKENIZER_MODEL, model)

    def add_token_list(self, tokens: List):
        self.add_array(constants.KEY_TOKENIZER_LIST, tokens)

    def add_token_merges(self, merges: List):
        self.add_array(constants.KEY_TOKENIZER_MERGES, merges)

    def add_token_scores(self, scores: List[float]):
        self.add_array(constants.KEY_TOKENIZER_SCORES, scores)
    
    def add_bos_token_id(self, id: int):
        self.add_uint32(constants.KEY_TOKENIZER_BOS_ID, id)

    def add_eos_token_id(self, id: int):
        self.add_uint32(constants.KEY_TOKENIZER_EOS_ID, id)

    def add_unk_token_id(self, id: int):
        self.add_uint32(constants.KEY_TOKENIZER_UNK_ID, id)

    def add_sep_token_id(self, id: int):
        self.add_uint32(constants.KEY_TOKENIZER_SEP_ID, id)

    def add_pad_token_id(self, id: int):
        self.add_uint32(constants.KEY_TOKENIZER_PAD_ID, id)


# Example usage:
if __name__ == "__main__":
    # Example usage with a file
    gguf_writer = GGUFWriter.open("example.gguf")

    gguf_writer.add_architecture("llama")
    gguf_writer.add_uint32("answer", 42)  # Write a 32-bit integer
    gguf_writer.add_float32("answer_in_float", 42.0)  # Write a 32-bit float
    gguf_writer.add_custom_alignment(64)
    tensor1 = np.ones((32,), dtype=np.float32) * 100.0
    tensor2 = np.ones((32,), dtype=np.float32) * 101.0
    gguf_writer.add_tensor_info("tensor0", tensor1)
    gguf_writer.add_tensor_info("tensor1", tensor2)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_ti_data_to_file()
    gguf_writer.write_tensor_to_file(tensor1)
    gguf_writer.write_tensor_to_file(tensor2)

    gguf_writer.close()
