"""TODOs
1. Implement writing tensor data with alignment.
2. Implement writers for known architectures, LLaMA in particular.
3. Add docstrings from the format specs.
4. After development is done, Convert it to a proper pip-installable Python package, and possibly move it to its own repo under ggml-org.
"""

import struct
from enum import IntEnum
from typing import List, Any
import constants


class GGMLQuantizationType(IntEnum):
    F32 = 0
    F16 = 1
    QR_0 = 2
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
    def get_type(value):
        if isinstance(value, str):
            return GGUFValueType.STRING
        elif isinstance(value, list):
            return GGUFValueType.ARRAY
        elif isinstance(value, float):
            return GGUFValueType.FLOAT32
        elif isinstance(value, bool):
            return GGUFValueType.BOOL
        else:
            return GGUFValueType.INT32


class GGUFWriter:
    def __init__(self, buffered_writer):
        self.buffered_writer = buffered_writer

    def write_header(self, tensor_count: int, metadata_kv_count: int):
        self.buffered_writer.write(struct.pack("<I", constants.GGUF_MAGIC))
        self.buffered_writer.write(struct.pack("<I", constants.GGUF_VERSION))
        self.buffered_writer.write(struct.pack("<I", tensor_count))
        self.buffered_writer.write(struct.pack("<I", metadata_kv_count))

    @classmethod
    def open(cls, path: str) -> "GGUFWriter":
        f = open(path, "wb")
        return cls(f)

    def write_key(self, key: str):
        self.write_value(key, GGUFValueType.STRING)

    def write_uint8(self, key: str, value: int):
        self.write_key(key)
        self.write_value(value, GGUFValueType.UINT8)

    def write_int8(self, key: str, value: int):
        self.write_key(key)
        self.write_value(value, GGUFValueType.INT8)

    def write_uint16(self, key: str, value: int):
        self.write_key(key)
        self.write_value(value, GGUFValueType.UINT16)

    def write_int16(self, key: str, value: int):
        self.write_key(key)
        self.write_value(value, GGUFValueType.INT16)

    def write_uint32(self, key: str, value: int):
        self.write_key(key)
        self.write(value, GGUFValueType.UINT32)

    def write_int32(self, key: str, value: int):
        self.write_key(key)
        self.write_value(value, GGUFValueType.INT32)

    def write_float32(self, key: str, value: float):
        self.write_key(key)
        self.write_value(value, GGUFValueType.FLOAT32)

    def write_bool(self, key: str, value: bool):
        self.write_key(key)
        self.write_value(value, GGUFValueType.BOOL)

    def write_string(self, key: str, value: str):
        self.write_key(key)
        self.write_value(value, GGUFValueType.STRING)

    def write_array(self, key: str, value: list):
        if not isinstance(value, list):
            raise ValueError("Value must be a list for array type")

        self.write_key(key)
        self.write_value(value, GGUFValueType.ARRAY)

    def write_value(self: str, value: Any, value_type: GGUFValueType = None):
        if value_type is None:
            value_type = GGUFValueType.get_type(value)

        self.buffered_writer.write(struct.pack("<I", value_type))

        if value_type == GGUFValueType.UINT8:
            self.buffered_writer.write(struct.pack("<B", value))
        elif value_type == GGUFValueType.INT8:
            self.buffered_writer.write(struct.pack("<b", value))
        elif value_type == GGUFValueType.UINT16:
            self.buffered_writer.write(struct.pack("<H", value))
        elif value_type == GGUFValueType.INT16:
            self.buffered_writer.write(struct.pack("<h", value))
        elif value_type == GGUFValueType.UINT32:
            self.buffered_writer.write(struct.pack("<I", value))
        elif value_type == GGUFValueType.INT32:
            self.buffered_writer.write(struct.pack("<i", value))
        elif value_type == GGUFValueType.FLOAT32:
            self.buffered_writer.write(struct.pack("<f", value))
        elif value_type == GGUFValueType.BOOL:
            self.buffered_writer.write(struct.pack("?", value))
        elif value_type == GGUFValueType.STRING:
            encoded_value = value.encode("utf8")
            self.buffered_writer.write(struct.pack("<I", len(encoded_value)))
            self.buffered_writer.write(encoded_value)
        elif value_type == GGUFValueType.ARRAY:
            self.buffered_writer.write(struct.pack("<I", len(value)))
            for item in value:
                self.write_value(item)
        else:
            raise ValueError("Invalid GGUF metadata value type")

    def flush(self):
        self.buffered_writer.flush()

    def close(self):
        self.buffered_writer.close()

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


# Example usage:
if __name__ == "__main__":
    # Example usage with a file
    gguf_writer = GGUFWriter.open("example.gguf")
    gguf_writer.write_header(0, 3)

gguf_writer.write_architecture("llama")
gguf_writer.write_uint32("answer", 42)  # Write a 32-bit integer
gguf_writer.write_float32("answer_in_float", 42.0)  # Write a 32-bit float
# Write an array of integers
#gguf_writer.write_array("simple_array", [1, 2, 3, 4])
# Write a nested array
#gguf_writer.write_array("nested", [1, "nested", [2, 3]])

gguf_writer.close()
