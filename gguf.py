#!/usr/bin/env python3
import shutil
import sys
import struct
import tempfile
import numpy as np

from enum import IntEnum, auto
from typing import Any, IO, List, Optional

#
# constants
#

GGUF_MAGIC             = 0x46554747
GGUF_VERSION           = 1
GGUF_DEFAULT_ALIGNMENT = 32

# general
KEY_GENERAL_ARCHITECTURE         = "general.architecture"
KEY_GENERAL_QUANTIZATION_VERSION = "general.quantization_version"
KEY_GENERAL_ALIGNMENT            = "general.alignment"
KEY_GENERAL_NAME                 = "general.name"
KEY_GENERAL_AUTHOR               = "general.author"
KEY_GENERAL_URL                  = "general.url"
KEY_GENERAL_DESCRIPTION          = "general.description"
KEY_GENERAL_LICENSE              = "general.license"
KEY_GENERAL_SOURCE_URL           = "general.source.url"
KEY_GENERAL_SOURCE_HF_REPO       = "general.source.hugginface.repository"
KEY_GENERAL_FILE_TYPE            = "general.file_type"

# LLM
KEY_CONTEXT_LENGTH        = "{arch}.context_length"
KEY_EMBEDDING_LENGTH      = "{arch}.embedding_length"
KEY_BLOCK_COUNT           = "{arch}.block_count"
KEY_FEED_FORWARD_LENGTH   = "{arch}.feed_forward_length"
KEY_USE_PARALLEL_RESIDUAL = "{arch}.use_parallel_residual"
KEY_TENSOR_DATA_LAYOUT    = "{arch}.tensor_data_layout"

# attention
KEY_ATTENTION_HEAD_COUNT        = "{arch}.attention.head_count"
KEY_ATTENTION_HEAD_COUNT_KV     = "{arch}.attention.head_count_kv"
KEY_ATTENTION_MAX_ALIBI_BIAS    = "{arch}.attention.max_alibi_bias"
KEY_ATTENTION_CLAMP_KQV         = "{arch}.attention.clamp_kqv"
KEY_ATTENTION_LAYERNORM_EPS     = "{arch}.attention.layer_norm_epsilon"
KEY_ATTENTION_LAYERNORM_RMS_EPS = "{arch}.attention.layer_norm_rms_epsilon"

# RoPE
KEY_ROPE_DIMENSION_COUNT = "{arch}.rope.dimension_count"
KEY_ROPE_FREQ_BASE       = "{arch}.rope.freq_base"
KEY_ROPE_SCALE_LINEAR    = "{arch}.rope.scale_linear"

# tokenization
KEY_TOKENIZER_MODEL      = "tokenizer.ggml.model"
KEY_TOKENIZER_LIST       = "tokenizer.ggml.tokens"
KEY_TOKENIZER_TOKEN_TYPE = "tokenizer.ggml.token_type"
KEY_TOKENIZER_SCORES     = "tokenizer.ggml.scores"
KEY_TOKENIZER_MERGES     = "tokenizer.ggml.merges"
KEY_TOKENIZER_BOS_ID     = "tokenizer.ggml.bos_token_id"
KEY_TOKENIZER_EOS_ID     = "tokenizer.ggml.eos_token_id"
KEY_TOKENIZER_UNK_ID     = "tokenizer.ggml.unknown_token_id"
KEY_TOKENIZER_SEP_ID     = "tokenizer.ggml.seperator_token_id"
KEY_TOKENIZER_PAD_ID     = "tokenizer.ggml.padding_token_id"
KEY_TOKENIZER_HF_JSON    = "tokenizer.huggingface.json"
KEY_TOKENIZER_RWKV       = "tokenizer.rwkv.world"


#
# recommended mapping of model tensor names for storage in gguf
#


class MODEL_ARCH(IntEnum):
    LLAMA   = auto()
    FALCON  = auto()
    GPT2    = auto()
    GPTJ    = auto()
    GPTNEOX = auto()
    MPT     = auto()


class MODEL_TENSOR(IntEnum):
    TOKEN_EMBD    = auto()
    POS_EMBD      = auto()
    OUTPUT        = auto()
    OUTPUT_NORM   = auto()
    ROPE_FREQS    = auto()
    ATTN_Q        = auto()
    ATTN_K        = auto()
    ATTN_V        = auto()
    ATTN_QKV      = auto()
    ATTN_OUT      = auto()
    ATTN_NORM     = auto()
    ATTN_NORM_2   = auto()
    ATTN_ROT_EMBD = auto()
    FFN_GATE      = auto()
    FFN_DOWN      = auto()
    FFN_UP        = auto()
    FFN_NORM      = auto()


MODEL_ARCH_NAMES = {
    MODEL_ARCH.LLAMA:   "llama",
    MODEL_ARCH.FALCON:  "falcon",
    MODEL_ARCH.GPT2:    "gpt2",
    MODEL_ARCH.GPTJ:    "gptj",
    MODEL_ARCH.GPTNEOX: "gptneox",
    MODEL_ARCH.MPT:     "mpt",
}

MODEL_TENSOR_NAMES = {
    MODEL_ARCH.LLAMA: {
        MODEL_TENSOR.TOKEN_EMBD:    "token_embd",
        MODEL_TENSOR.OUTPUT_NORM:   "output_norm",
        MODEL_TENSOR.OUTPUT:        "output",
        MODEL_TENSOR.ROPE_FREQS:    "rope_freqs",
        MODEL_TENSOR.ATTN_NORM:     "blk.{bid}.attn_norm",
        MODEL_TENSOR.ATTN_Q:        "blk.{bid}.attn_q",
        MODEL_TENSOR.ATTN_K:        "blk.{bid}.attn_k",
        MODEL_TENSOR.ATTN_V:        "blk.{bid}.attn_v",
        MODEL_TENSOR.ATTN_OUT:      "blk.{bid}.attn_output",
        MODEL_TENSOR.ATTN_ROT_EMBD: "blk.{bid}.attn_rot_embd",
        MODEL_TENSOR.FFN_NORM:      "blk.{bid}.ffn_norm",
        MODEL_TENSOR.FFN_GATE:      "blk.{bid}.ffn_gate",
        MODEL_TENSOR.FFN_DOWN:      "blk.{bid}.ffn_down",
        MODEL_TENSOR.FFN_UP:        "blk.{bid}.ffn_up",
    },
    MODEL_ARCH.GPTNEOX: {
        MODEL_TENSOR.TOKEN_EMBD:    "token_embd",
        MODEL_TENSOR.OUTPUT_NORM:   "output_norm",
        MODEL_TENSOR.OUTPUT:        "output",
        MODEL_TENSOR.ATTN_NORM:     "blk.{bid}.attn_norm",
        MODEL_TENSOR.ATTN_QKV:      "blk.{bid}.attn_qkv",
        MODEL_TENSOR.ATTN_OUT:      "blk.{bid}.attn_output",
        MODEL_TENSOR.FFN_NORM:      "blk.{bid}.ffn_norm",
        MODEL_TENSOR.FFN_DOWN:      "blk.{bid}.ffn_down",
        MODEL_TENSOR.FFN_UP:        "blk.{bid}.ffn_up",
    },
    MODEL_ARCH.FALCON: {
        MODEL_TENSOR.TOKEN_EMBD:  "token_embd",
        MODEL_TENSOR.OUTPUT_NORM: "output_norm",
        MODEL_TENSOR.OUTPUT:      "output",
        MODEL_TENSOR.ATTN_NORM:   "blk.{bid}.attn_norm",
        MODEL_TENSOR.ATTN_NORM_2: "blk.{bid}.attn_norm_2",
        MODEL_TENSOR.ATTN_QKV:    "blk.{bid}.attn_qkv",
        MODEL_TENSOR.ATTN_OUT:    "blk.{bid}.attn_output",
        MODEL_TENSOR.FFN_DOWN:    "blk.{bid}.ffn_down",
        MODEL_TENSOR.FFN_UP:      "blk.{bid}.ffn_up",
    },
    MODEL_ARCH.GPT2: {
        # TODO
    },
    # TODO
}

# tensors that will not be serialized
MODEL_TENSOR_SKIP = {
    MODEL_ARCH.LLAMA: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
}


# TODO: the following helper functions should be removed
#       instead, get_tensor_name_map should return tuples of (name, MODEL_TENSOR)
#       however, my Python is very bad, and I couldn't figure out how to do this, hence these functions
# REMOVE
def should_skip_tensor_TMP(arch: MODEL_ARCH, n_blocks: int, name: str) -> bool:
    for skip in MODEL_TENSOR_SKIP.get(arch, []):
        for i in range(n_blocks):
            if name == MODEL_TENSOR_NAMES[arch][skip].format(bid=i):
                return True

    return False


def get_tensor_name_map(arch: MODEL_ARCH, n_blocks: int) -> dict:
    tensor_map = {}

    # Token embeddings
    mapped_to = MODEL_TENSOR_NAMES[arch].get(MODEL_TENSOR.TOKEN_EMBD, None)

    tensor_map["gpt_neox.embed_in"]           = mapped_to  # gptneox
    tensor_map["transformer.wte"]             = mapped_to  # gpt2 mpt
    tensor_map["transformer.word_embeddings"] = mapped_to  # falcon
    tensor_map["model.embed_tokens"]          = mapped_to  # llama-hf
    tensor_map["tok_embeddings"]              = mapped_to  # llama-pth

    # Position embeddings
    mapped_to = MODEL_TENSOR_NAMES[arch].get(MODEL_TENSOR.POS_EMBD, None)

    tensor_map["transformer.wpe"] = mapped_to  # gpt2

    # Output
    mapped_to = MODEL_TENSOR_NAMES[arch].get(MODEL_TENSOR.OUTPUT, None)

    tensor_map["embed_out"] = mapped_to  # gptneox
    tensor_map["lm_head"]   = mapped_to  # gpt2 mpt falcon llama-hf
    tensor_map["output"]    = mapped_to  # llama-pth

    # Output norm
    mapped_to = MODEL_TENSOR_NAMES[arch].get(MODEL_TENSOR.OUTPUT_NORM, None)

    tensor_map["gpt_neox.final_layer_norm"] = mapped_to  # gptneox
    tensor_map["transformer.ln_f"]          = mapped_to  # gpt2 falcon
    tensor_map["transformer.norm_f"]        = mapped_to  # mpt
    tensor_map["model.norm"]                = mapped_to  # llama-hf
    tensor_map["norm"]                      = mapped_to  # llama-pth

    # Rope frequencies
    mapped_to = MODEL_TENSOR_NAMES[arch].get(MODEL_TENSOR.ROPE_FREQS, None)

    tensor_map["rope.freqs"] = mapped_to  # llama-pth

    # Attention and feed-forward blocks
    for i in range(0, n_blocks):
        # Attention norm
        # TODO: is there are simpler way to write these 2 lines in Python?
        mapped_to = MODEL_TENSOR_NAMES[arch].get(MODEL_TENSOR.ATTN_NORM, None)
        mapped_to = mapped_to.format(bid=i) if mapped_to else None

        tensor_map["gpt_neox.layers."+str(i)+".input_layernorm"] = mapped_to  # gptneox
        tensor_map["transformer.h."+str(i)+".ln_1"]              = mapped_to  # gpt2
        tensor_map["transformer.blocks."+str(i)+".norm_1"]       = mapped_to  # mpt
        tensor_map["transformer.h."+str(i)+".input_layernorm"]   = mapped_to  # falcon7b
        tensor_map["transformer.h."+str(i)+".ln_mlp"]            = mapped_to  # falcon40b
        tensor_map["model.layers."+str(i)+".input_layernorm"]    = mapped_to  # llama-hf
        tensor_map["layers."+str(i)+".attention_norm"]           = mapped_to  # llama-pth

        # Attention norm 2
        mapped_to = MODEL_TENSOR_NAMES[arch].get(MODEL_TENSOR.ATTN_NORM_2, None)
        mapped_to = mapped_to.format(bid=i) if mapped_to is not None else None

        tensor_map["transformer.h."+str(i)+".ln_attn"] = mapped_to  # falcon40b

        # Attention query-key-value
        mapped_to = MODEL_TENSOR_NAMES[arch].get(MODEL_TENSOR.ATTN_QKV, None)
        mapped_to = mapped_to.format(bid=i) if mapped_to is not None else None

        tensor_map["gpt_neox.layers."+str(i)+".attention.query_key_value"]    = mapped_to  # gptneox
        tensor_map["transformer.h."+str(i)+".attn.c_attn"]                    = mapped_to  # gpt2
        tensor_map["transformer.blocks."+str(i)+".attn.Wqkv"]                 = mapped_to  # mpt
        tensor_map["transformer.h."+str(i)+".self_attention.query_key_value"] = mapped_to  # falcon

        # Attention query
        mapped_to = MODEL_TENSOR_NAMES[arch].get(MODEL_TENSOR.ATTN_Q, None)
        mapped_to = mapped_to.format(bid=i) if mapped_to is not None else None

        tensor_map["model.layers."+str(i)+".self_attn.q_proj"] = mapped_to  # llama-hf
        tensor_map["layers."+str(i)+".attention.wq"]           = mapped_to  # llama-pth

        # Attention key
        mapped_to = MODEL_TENSOR_NAMES[arch].get(MODEL_TENSOR.ATTN_K, None)
        mapped_to = mapped_to.format(bid=i) if mapped_to is not None else None

        tensor_map["model.layers."+str(i)+".self_attn.k_proj"] = mapped_to  # llama-hf
        tensor_map["layers."+str(i)+".attention.wk"]           = mapped_to  # llama-pth

        # Attention value
        mapped_to = MODEL_TENSOR_NAMES[arch].get(MODEL_TENSOR.ATTN_V, None)
        mapped_to = mapped_to.format(bid=i) if mapped_to is not None else None

        tensor_map["model.layers."+str(i)+".self_attn.v_proj"] = mapped_to  # llama-hf
        tensor_map["layers."+str(i)+".attention.wv"]           = mapped_to  # llama-pth

        # Attention output
        mapped_to = MODEL_TENSOR_NAMES[arch].get(MODEL_TENSOR.ATTN_OUT, None)
        mapped_to = mapped_to.format(bid=i) if mapped_to is not None else None

        tensor_map["gpt_neox.layers."+str(i)+".attention.dense"]    = mapped_to  # gptneox
        tensor_map["transformer.h."+str(i)+".attn.c_proj"]          = mapped_to  # gpt2
        tensor_map["transformer.blocks."+str(i)+".attn.out_proj"]   = mapped_to  # mpt
        tensor_map["transformer.h."+str(i)+".self_attention.dense"] = mapped_to  # falcon
        tensor_map["model.layers."+str(i)+".self_attn.o_proj"]      = mapped_to  # llama-hf
        tensor_map["layers."+str(i)+".attention.wo"]                = mapped_to  # llama-pth

        # Rotary embeddings
        mapped_to = MODEL_TENSOR_NAMES[arch].get(MODEL_TENSOR.ATTN_ROT_EMBD, None)
        mapped_to = mapped_to.format(bid=i) if mapped_to is not None else None

        tensor_map["model.layers."+str(i)+".self_attn.rotary_emb.inv_freq"]  = mapped_to  # llama-hf
        tensor_map["layers."+str(i)+".attention.inner_attention.rope.freqs"] = mapped_to  # llama-pth

        # Feed-forward norm
        mapped_to = MODEL_TENSOR_NAMES[arch].get(MODEL_TENSOR.FFN_NORM, None)
        mapped_to = mapped_to.format(bid=i) if mapped_to is not None else None

        tensor_map["gpt_neox.layers."+str(i)+".post_attention_layernorm"] = mapped_to  # gptneox
        tensor_map["transformer.h."+str(i)+".ln_2"]                       = mapped_to  # gpt2
        tensor_map["transformer.blocks."+str(i)+".norm_2"]                = mapped_to  # mpt
        tensor_map["model.layers."+str(i)+".post_attention_layernorm"]    = mapped_to  # llama-hf
        tensor_map["layers."+str(i)+".ffn_norm"]                          = mapped_to  # llama-pth

        # Feed-forward up
        mapped_to = MODEL_TENSOR_NAMES[arch].get(MODEL_TENSOR.FFN_UP, None)
        mapped_to = mapped_to.format(bid=i) if mapped_to is not None else None

        tensor_map["gpt_neox.layers."+str(i)+".mlp.dense_h_to_4h"] = mapped_to  # gptneox
        tensor_map["transformer.h."+str(i)+".mlp.c_fc"]            = mapped_to  # gpt2
        tensor_map["transformer.blocks."+str(i)+".ffn.up_proj"]    = mapped_to  # mpt
        tensor_map["transformer.h."+str(i)+".mlp.dense_h_to_4h"]   = mapped_to  # falcon
        tensor_map["model.layers."+str(i)+".mlp.up_proj"]          = mapped_to  # llama-hf
        tensor_map["layers."+str(i)+".feed_forward.w3"]            = mapped_to  # llama-pth

        # Feed-forward gate
        mapped_to = MODEL_TENSOR_NAMES[arch].get(MODEL_TENSOR.FFN_GATE, None)
        mapped_to = mapped_to.format(bid=i) if mapped_to is not None else None

        tensor_map["model.layers."+str(i)+".mlp.gate_proj"] = mapped_to  # llama-hf
        tensor_map["layers."+str(i)+".feed_forward.w1"]     = mapped_to  # llama-pth

        # Feed-forward down
        mapped_to = MODEL_TENSOR_NAMES[arch].get(MODEL_TENSOR.FFN_DOWN, None)
        mapped_to = mapped_to.format(bid=i) if mapped_to is not None else None

        tensor_map["gpt_neox.layers."+str(i)+".mlp.dense_4h_to_h"] = mapped_to  # gptneox
        tensor_map["transformer.h."+str(i)+".mlp.c_proj"]          = mapped_to  # gpt2
        tensor_map["transformer.blocks."+str(i)+".ffn.down_proj"]  = mapped_to  # mpt
        tensor_map["transformer.h."+str(i)+".mlp.dense_4h_to_h"]   = mapped_to  # falcon
        tensor_map["model.layers."+str(i)+".mlp.down_proj"]        = mapped_to  # llama-hf
        tensor_map["layers."+str(i)+".feed_forward.w2"]            = mapped_to  # llama-pth

    return tensor_map


class TokenType(IntEnum):
    NORMAL       = 1
    UNKNOWN      = 2
    CONTROL      = 3
    USER_DEFINED = 4
    UNUSED       = 5
    BYTE         = 6

#
# implementation
#


class GGMLQuantizationType(IntEnum):
    F32  = 0
    F16  = 1
    Q4_0 = 2
    Q4_1 = 3
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
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9

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
    def __init__(self, path: str, arch: str, use_temp_file = True):
        self.fout = open(path, "wb")
        self.arch = arch
        self.offset_tensor = 0
        self.data_alignment = GGUF_DEFAULT_ALIGNMENT
        self.kv_data = b""
        self.kv_data_count = 0
        self.ti_data = b""
        self.ti_data_count = 0
        self.add_architecture()
        self.use_temp_file = use_temp_file
        self.tensors = []

    def write_header_to_file(self):
        self.fout.write(struct.pack("<I", GGUF_MAGIC))
        self.fout.write(struct.pack("<I", GGUF_VERSION))
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
        if len(val) == 0:
            return
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
            self.kv_data_count += 1

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

    def add_tensor_info(self, name: str, tensor_shape: np.ndarray, tensor_dtype: np.dtype, tensor_nbytes: int, raw_dtype: Optional[GGMLQuantizationType] = None):
        assert raw_dtype is not None or tensor_dtype in (np.float32, np.float16), "Only F32 and F16 tensors are supported for now"

        encoded_name = name.encode("utf8")
        self.ti_data += struct.pack("<I", len(encoded_name))
        self.ti_data += encoded_name
        n_dims = len(tensor_shape)
        self.ti_data += struct.pack("<I", n_dims)
        for i in range(n_dims):
            self.ti_data += struct.pack("<I", tensor_shape[n_dims - 1 - i])
        if raw_dtype is None:
            dtype = GGMLQuantizationType.F32 if tensor_dtype == np.float32 else GGMLQuantizationType.F16
        else:
            dtype = raw_dtype
        self.ti_data += struct.pack("<I", dtype)
        self.ti_data += struct.pack("<Q", self.offset_tensor)
        self.offset_tensor += GGUFWriter.ggml_pad(tensor_nbytes, self.data_alignment)
        self.ti_data_count += 1

    def add_tensor(self, name: str, tensor: np.ndarray, raw_shape: Optional[np.ndarray] = None, raw_dtype: Optional[GGMLQuantizationType] = None):
        if self.use_temp_file and not hasattr(self, "temp_file"):
            self.temp_file = tempfile.SpooledTemporaryFile(mode="w+b", max_size=256*1024*1024)
            self.temp_file.seek(0)

        self.add_tensor_info(name, raw_shape if raw_shape is not None else tensor.shape, tensor.dtype, tensor.nbytes, raw_dtype = raw_dtype)

        pad = GGUFWriter.ggml_pad(tensor.nbytes, self.data_alignment) - tensor.nbytes

        if not self.use_temp_file:
            self.tensors.append((tensor, pad))
            return

        tensor.tofile(self.temp_file)

        if pad != 0:
            self.temp_file.write(bytes([0] * pad))

    def write_tensor_data(self, tensor: np.ndarray):
        pad = GGUFWriter.ggml_pad(self.fout.tell(), self.data_alignment) - self.fout.tell()
        if pad != 0:
            self.fout.write(bytes([0] * pad))

        tensor.tofile(self.fout)

        pad = GGUFWriter.ggml_pad(tensor.nbytes, self.data_alignment) - tensor.nbytes
        if pad != 0:
            self.fout.write(bytes([0] * pad))

    def write_tensors_to_file(self):
        self.write_ti_data_to_file()

        pad = GGUFWriter.ggml_pad(self.fout.tell(), self.data_alignment) - self.fout.tell()
        if pad != 0:
            self.fout.write(bytes([0] * pad))

        if not self.use_temp_file:
            for (currtensor, currpad) in self.tensors:
                currtensor.tofile(self.fout)
                if currpad != 0:
                    self.fout.write(bytes([0] * currpad))
            return

        self.temp_file.seek(0)

        shutil.copyfileobj(self.temp_file, self.fout)
        self.flush()
        self.temp_file.close()

    def flush(self):
        self.fout.flush()

    def close(self):
        self.fout.close()

    def add_architecture(self):
        self.add_string(KEY_GENERAL_ARCHITECTURE, self.arch)

    def add_author(self, author: str):
        self.add_string(KEY_GENERAL_AUTHOR, author)

    def add_tensor_data_layout(self, layout: str):
        self.add_string(KEY_TENSOR_DATA_LAYOUT.format(arch=self.arch), layout)

    def add_url(self, url: str):
        self.add_string(KEY_GENERAL_URL, url)

    def add_description(self, description: str):
        self.add_string(KEY_GENERAL_DESCRIPTION, description)

    def add_source_url(self, url: str):
        self.add_string(KEY_GENERAL_SOURCE_URL, url)

    def add_source_hf_repo(self, repo: str):
        self.add_string(KEY_GENERAL_SOURCE_HF_REPO, repo)

    def add_file_type(self, ftype: int):
        self.add_uint32(KEY_GENERAL_FILE_TYPE, ftype)

    def add_name(self, name: str):
        self.add_string(KEY_GENERAL_NAME, name)

    def add_quantization_version(self, quantization_version: GGMLQuantizationType):
        self.add_uint32(
            KEY_GENERAL_QUANTIZATION_VERSION, quantization_version)

    def add_custom_alignment(self, alignment: int):
        self.data_alignment = alignment
        self.add_uint32(KEY_GENERAL_ALIGNMENT, alignment)

    def add_context_length(self, length: int):
        self.add_uint32(
            KEY_CONTEXT_LENGTH.format(arch=self.arch), length)

    def add_embedding_length(self, length: int):
        self.add_uint32(
            KEY_EMBEDDING_LENGTH.format(arch=self.arch), length)

    def add_block_count(self, length: int):
        self.add_uint32(
            KEY_BLOCK_COUNT.format(arch=self.arch), length)

    def add_feed_forward_length(self, length: int):
        self.add_uint32(
            KEY_FEED_FORWARD_LENGTH.format(arch=self.arch), length)

    def add_parallel_residual(self, use: bool):
        self.add_bool(
            KEY_USE_PARALLEL_RESIDUAL.format(arch=self.arch), use)

    def add_tensor_data_layout(self, layout: str):
        self.add_string(
            KEY_TENSOR_DATA_LAYOUT.format(arch=self.arch), layout)

    def add_head_count(self, count: int):
        self.add_uint32(
            KEY_ATTENTION_HEAD_COUNT.format(arch=self.arch), count)

    def add_head_count_kv(self, count: int):
        self.add_uint32(
            KEY_ATTENTION_HEAD_COUNT_KV.format(arch=self.arch), count)

    def add_max_alibi_bias(self, bias: float):
        self.add_float32(
            KEY_ATTENTION_MAX_ALIBI_BIAS.format(arch=self.arch), bias)

    def add_clamp_kqv(self, value: float):
        self.add_float32(
            KEY_ATTENTION_CLAMP_KQV.format(arch=self.arch), value)

    def add_layer_norm_eps(self, value: float):
        self.add_float32(
            KEY_ATTENTION_LAYERNORM_EPS.format(arch=self.arch), value)

    def add_layer_norm_rms_eps(self, value: float):
        self.add_float32(
            KEY_ATTENTION_LAYERNORM_RMS_EPS.format(arch=self.arch), value)

    def add_rope_dimension_count(self, count: int):
        self.add_uint32(
            KEY_ROPE_DIMENSION_COUNT.format(arch=self.arch), count)

    def add_rope_freq_base(self, value: float):
        self.add_float32(KEY_ROPE_FREQ_BASE.format(arch=self.arch), value)

    def add_rope_scale_linear(self, value: float):
        self.add_float32(KEY_ROPE_SCALE_LINEAR.format(arch=self.arch), value)

    def add_tokenizer_model(self, model: str):
        self.add_string(KEY_TOKENIZER_MODEL, model)

    def add_token_list(self, tokens: List):
        self.add_array(KEY_TOKENIZER_LIST, tokens)

    def add_token_merges(self, merges: List):
        self.add_array(KEY_TOKENIZER_MERGES, merges)

    def add_token_types(self, types: List[int]):
        self.add_array(KEY_TOKENIZER_TOKEN_TYPE, types)

    def add_token_scores(self, scores: List[float]):
        self.add_array(KEY_TOKENIZER_SCORES, scores)

    def add_bos_token_id(self, id: int):
        self.add_uint32(KEY_TOKENIZER_BOS_ID, id)

    def add_eos_token_id(self, id: int):
        self.add_uint32(KEY_TOKENIZER_EOS_ID, id)

    def add_unk_token_id(self, id: int):
        self.add_uint32(KEY_TOKENIZER_UNK_ID, id)

    def add_sep_token_id(self, id: int):
        self.add_uint32(KEY_TOKENIZER_SEP_ID, id)

    def add_pad_token_id(self, id: int):
        self.add_uint32(KEY_TOKENIZER_PAD_ID, id)


# Example usage:
if __name__ == "__main__":
    # Example usage with a file
    gguf_writer = GGUFWriter("example.gguf", "llama")

    gguf_writer.add_architecture()
    gguf_writer.add_block_count(12)
    gguf_writer.add_uint32("answer", 42)  # Write a 32-bit integer
    gguf_writer.add_float32("answer_in_float", 42.0)  # Write a 32-bit float
    gguf_writer.add_custom_alignment(64)

    tensor1 = np.ones((32,), dtype=np.float32) * 100.0
    tensor2 = np.ones((64,), dtype=np.float32) * 101.0
    tensor3 = np.ones((96,), dtype=np.float32) * 102.0

    gguf_writer.add_tensor("tensor1", tensor1)
    gguf_writer.add_tensor("tensor2", tensor2)
    gguf_writer.add_tensor("tensor3", tensor3)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()
