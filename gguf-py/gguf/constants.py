from __future__ import annotations

from enum import Enum, IntEnum, auto
from typing import Any

#
# constants
#
GGUF_MAGIC             = 0x46554747  # "GGUF"
GGUF_VERSION           = 3
GGUF_DEFAULT_ALIGNMENT = 32
GGML_QUANT_VERSION     = 2  # GGML_QNT_VERSION from ggml.h


#
# model metadata keys
#
class GGUFMetadataKeys:
    class General:
        ARCHITECTURE         = "general.architecture"
        QUANTIZATION_VERSION = "general.quantization_version"
        ALIGNMENT            = "general.alignment"
        NAME                 = "general.name"
        AUTHOR               = "general.author"
        VERSION              = "general.version"
        URL                  = "general.url"
        DESCRIPTION          = "general.description"
        LICENSE              = "general.license"
        SOURCE_URL           = "general.source.url"
        SOURCE_REPO          = "general.source.repository"
        FILE_TYPE            = "general.file_type"
        ENDIANESS            = "general.endianess"

    class LLM:
        VOCAB_SIZE                 = "{arch}.vocab_size"
        CONTEXT_LENGTH             = "{arch}.context_length"
        EMBEDDING_LENGTH           = "{arch}.embedding_length"
        BLOCK_COUNT                = "{arch}.block_count"
        LEADING_DENSE_BLOCK_COUNT  = "{arch}.leading_dense_block_count"
        FEED_FORWARD_LENGTH        = "{arch}.feed_forward_length"
        EXPERT_FEED_FORWARD_LENGTH = "{arch}.expert_feed_forward_length"
        USE_PARALLEL_RESIDUAL      = "{arch}.use_parallel_residual"
        TENSOR_DATA_LAYOUT         = "{arch}.tensor_data_layout"
        EXPERT_COUNT               = "{arch}.expert_count"
        EXPERT_USED_COUNT          = "{arch}.expert_used_count"
        EXPERT_SHARED_COUNT        = "{arch}.expert_shared_count"
        EXPERT_WEIGHTS_SCALE       = "{arch}.expert_weights_scale"
        POOLING_TYPE               = "{arch}.pooling_type"
        LOGIT_SCALE                = "{arch}.logit_scale"

    class Attention:
        HEAD_COUNT        = "{arch}.attention.head_count"
        HEAD_COUNT_KV     = "{arch}.attention.head_count_kv"
        MAX_ALIBI_BIAS    = "{arch}.attention.max_alibi_bias"
        CLAMP_KQV         = "{arch}.attention.clamp_kqv"
        KEY_LENGTH        = "{arch}.attention.key_length"
        VALUE_LENGTH      = "{arch}.attention.value_length"
        LAYERNORM_EPS     = "{arch}.attention.layer_norm_epsilon"
        LAYERNORM_RMS_EPS = "{arch}.attention.layer_norm_rms_epsilon"
        CAUSAL            = "{arch}.attention.causal"
        Q_LORA_RANK       = "{arch}.attention.q_lora_rank"
        KV_LORA_RANK      = "{arch}.attention.kv_lora_rank"

    class Rope:
        DIMENSION_COUNT         = "{arch}.rope.dimension_count"
        FREQ_BASE               = "{arch}.rope.freq_base"
        SCALING_TYPE            = "{arch}.rope.scaling.type"
        SCALING_FACTOR          = "{arch}.rope.scaling.factor"
        SCALING_ATTN_FACTOR     = "{arch}.rope.scaling.attn_factor"
        SCALING_ORIG_CTX_LEN    = "{arch}.rope.scaling.original_context_length"
        SCALING_FINETUNED       = "{arch}.rope.scaling.finetuned"
        SCALING_YARN_LOG_MUL    = "{arch}.rope.scaling.yarn_log_multiplier"

    class SSM:
        CONV_KERNEL    = "{arch}.ssm.conv_kernel"
        INNER_SIZE     = "{arch}.ssm.inner_size"
        STATE_SIZE     = "{arch}.ssm.state_size"
        TIME_STEP_RANK = "{arch}.ssm.time_step_rank"

    class Tokenizer:
        MODEL            = "tokenizer.model"               # STRING: e.g. llama, gpt2, etc...
        TYPE             = "tokenizer.type"                # STRING: BPE, SPM, WPM, etc.
        NORM             = "tokenizer.norm"                # OBJECT {"type": "ByteLevel", ...}
        PRE              = "tokenizer.pre"                 # OBJECT {"type": "ByteLevel", ...}
        ADDED            = "tokenizer.added"               # ARRAY of OBJECTs: [{"id": 1, ...}, ...]
        VOCAB            = "tokenizer.vocab"               # ARRAY of STRINGs: ["[BOS]", ...]
        MERGES           = "tokenizer.merges"              # ARRAY of STRINGs: ["▁ t", ...]
        TOKEN_TYPE       = "tokenizer.token_type"          # ARRAY of INT [2, ...]
        TOKEN_TYPE_COUNT = "tokenizer.token_type_count"    # BERT token types
        SCORES           = "tokenizer.scores"              # WPM only
        BOS_ID           = "tokenizer.bos_token_id"
        EOS_ID           = "tokenizer.eos_token_id"
        UNK_ID           = "tokenizer.unknown_token_id"
        SEP_ID           = "tokenizer.separator_token_id"  # Fixed typo
        PAD_ID           = "tokenizer.padding_token_id"
        CLS_ID           = "tokenizer.cls_token_id"
        MASK_ID          = "tokenizer.mask_token_id"
        ADD_BOS          = "tokenizer.add_bos_token"
        ADD_EOS          = "tokenizer.add_eos_token"
        ADD_PREFIX       = "tokenizer.add_space_prefix"
        RWKV             = "tokenizer.rwkv.world"
        CHAT_TEMPLATE    = "tokenizer.chat_template"
        CHAT_TEMPLATE_N  = "tokenizer.chat_template.{name}"
        CHAT_TEMPLATES   = "tokenizer.chat_templates"
        # FIM/Infill special tokens constants
        PREFIX_ID        = "tokenizer.prefix_token_id"
        SUFFIX_ID        = "tokenizer.suffix_token_id"
        MIDDLE_ID        = "tokenizer.middle_token_id"
        EOT_ID           = "tokenizer.eot_token_id"


#
# recommended mapping of model tensor names for storage in gguf
#


class MODEL_ARCH(IntEnum):
    LLAMA      = auto()
    FALCON     = auto()
    BAICHUAN   = auto()
    GROK       = auto()
    GPT2       = auto()
    GPTJ       = auto()
    GPTNEOX    = auto()
    MPT        = auto()
    STARCODER  = auto()
    REFACT     = auto()
    BERT       = auto()
    NOMIC_BERT = auto()
    JINA_BERT_V2 = auto()
    BLOOM      = auto()
    STABLELM   = auto()
    QWEN       = auto()
    QWEN2      = auto()
    QWEN2MOE   = auto()
    PHI2       = auto()
    PHI3       = auto()
    PLAMO      = auto()
    CODESHELL  = auto()
    ORION      = auto()
    INTERNLM2  = auto()
    MINICPM    = auto()
    GEMMA      = auto()
    STARCODER2 = auto()
    MAMBA      = auto()
    XVERSE     = auto()
    COMMAND_R  = auto()
    DBRX       = auto()
    OLMO       = auto()
    ARCTIC     = auto()
    DEEPSEEK2  = auto()


class MODEL_TENSOR(IntEnum):
    TOKEN_EMBD         = auto()
    TOKEN_EMBD_NORM    = auto()
    TOKEN_TYPES        = auto()
    POS_EMBD           = auto()
    OUTPUT             = auto()
    OUTPUT_NORM        = auto()
    ROPE_FREQS         = auto()
    ROPE_FACTORS_LONG  = auto()
    ROPE_FACTORS_SHORT = auto()
    ATTN_Q             = auto()
    ATTN_K             = auto()
    ATTN_V             = auto()
    ATTN_QKV           = auto()
    ATTN_OUT           = auto()
    ATTN_NORM          = auto()
    ATTN_NORM_2        = auto()
    ATTN_OUT_NORM      = auto()
    ATTN_ROT_EMBD      = auto()
    FFN_GATE_INP       = auto()
    FFN_GATE_INP_SHEXP = auto()
    FFN_NORM           = auto()
    FFN_GATE           = auto()
    FFN_DOWN           = auto()
    FFN_UP             = auto()
    FFN_ACT            = auto()
    FFN_NORM_EXP       = auto()
    FFN_GATE_EXP       = auto()
    FFN_DOWN_EXP       = auto()
    FFN_UP_EXP         = auto()
    FFN_GATE_SHEXP     = auto()
    FFN_DOWN_SHEXP     = auto()
    FFN_UP_SHEXP       = auto()
    ATTN_Q_NORM        = auto()
    ATTN_K_NORM        = auto()
    LAYER_OUT_NORM     = auto()
    SSM_IN             = auto()
    SSM_CONV1D         = auto()
    SSM_X              = auto()
    SSM_DT             = auto()
    SSM_A              = auto()
    SSM_D              = auto()
    SSM_OUT            = auto()
    ATTN_Q_A           = auto()
    ATTN_Q_B           = auto()
    ATTN_KV_A_MQA      = auto()
    ATTN_KV_B          = auto()
    ATTN_Q_A_NORM      = auto()
    ATTN_KV_A_NORM     = auto()


MODEL_ARCH_NAMES: dict[MODEL_ARCH, str] = {
    MODEL_ARCH.LLAMA:          "llama",
    MODEL_ARCH.FALCON:         "falcon",
    MODEL_ARCH.BAICHUAN:       "baichuan",
    MODEL_ARCH.GROK:           "grok",
    MODEL_ARCH.GPT2:           "gpt2",
    MODEL_ARCH.GPTJ:           "gptj",
    MODEL_ARCH.GPTNEOX:        "gptneox",
    MODEL_ARCH.MPT:            "mpt",
    MODEL_ARCH.STARCODER:      "starcoder",
    MODEL_ARCH.REFACT:         "refact",
    MODEL_ARCH.BERT:           "bert",
    MODEL_ARCH.NOMIC_BERT:     "nomic-bert",
    MODEL_ARCH.JINA_BERT_V2:   "jina-bert-v2",
    MODEL_ARCH.BLOOM:          "bloom",
    MODEL_ARCH.STABLELM:       "stablelm",
    MODEL_ARCH.QWEN:           "qwen",
    MODEL_ARCH.QWEN2:          "qwen2",
    MODEL_ARCH.QWEN2MOE:       "qwen2moe",
    MODEL_ARCH.PHI2:           "phi2",
    MODEL_ARCH.PHI3:           "phi3",
    MODEL_ARCH.PLAMO:          "plamo",
    MODEL_ARCH.CODESHELL:      "codeshell",
    MODEL_ARCH.ORION:          "orion",
    MODEL_ARCH.INTERNLM2:      "internlm2",
    MODEL_ARCH.MINICPM:        "minicpm",
    MODEL_ARCH.GEMMA:          "gemma",
    MODEL_ARCH.STARCODER2:     "starcoder2",
    MODEL_ARCH.MAMBA:          "mamba",
    MODEL_ARCH.XVERSE:         "xverse",
    MODEL_ARCH.COMMAND_R:      "command-r",
    MODEL_ARCH.DBRX:           "dbrx",
    MODEL_ARCH.OLMO:           "olmo",
    MODEL_ARCH.ARCTIC:         "arctic",
    MODEL_ARCH.DEEPSEEK2:      "deepseek2",
}

TENSOR_NAMES: dict[MODEL_TENSOR, str] = {
    MODEL_TENSOR.TOKEN_EMBD:         "token_embd",
    MODEL_TENSOR.TOKEN_EMBD_NORM:    "token_embd_norm",
    MODEL_TENSOR.TOKEN_TYPES:        "token_types",
    MODEL_TENSOR.POS_EMBD:           "position_embd",
    MODEL_TENSOR.OUTPUT_NORM:        "output_norm",
    MODEL_TENSOR.OUTPUT:             "output",
    MODEL_TENSOR.ROPE_FREQS:         "rope_freqs",
    MODEL_TENSOR.ROPE_FACTORS_LONG:  "rope_factors_long",
    MODEL_TENSOR.ROPE_FACTORS_SHORT: "rope_factors_short",
    MODEL_TENSOR.ATTN_NORM:          "blk.{bid}.attn_norm",
    MODEL_TENSOR.ATTN_NORM_2:        "blk.{bid}.attn_norm_2",
    MODEL_TENSOR.ATTN_QKV:           "blk.{bid}.attn_qkv",
    MODEL_TENSOR.ATTN_Q:             "blk.{bid}.attn_q",
    MODEL_TENSOR.ATTN_K:             "blk.{bid}.attn_k",
    MODEL_TENSOR.ATTN_V:             "blk.{bid}.attn_v",
    MODEL_TENSOR.ATTN_OUT:           "blk.{bid}.attn_output",
    MODEL_TENSOR.ATTN_ROT_EMBD:      "blk.{bid}.attn_rot_embd",
    MODEL_TENSOR.ATTN_Q_NORM:        "blk.{bid}.attn_q_norm",
    MODEL_TENSOR.ATTN_K_NORM:        "blk.{bid}.attn_k_norm",
    MODEL_TENSOR.ATTN_OUT_NORM:      "blk.{bid}.attn_output_norm",
    MODEL_TENSOR.FFN_GATE_INP:       "blk.{bid}.ffn_gate_inp",
    MODEL_TENSOR.FFN_GATE_INP_SHEXP: "blk.{bid}.ffn_gate_inp_shexp",
    MODEL_TENSOR.FFN_NORM:           "blk.{bid}.ffn_norm",
    MODEL_TENSOR.FFN_GATE:           "blk.{bid}.ffn_gate",
    MODEL_TENSOR.FFN_DOWN:           "blk.{bid}.ffn_down",
    MODEL_TENSOR.FFN_UP:             "blk.{bid}.ffn_up",
    MODEL_TENSOR.FFN_GATE_SHEXP:     "blk.{bid}.ffn_gate_shexp",
    MODEL_TENSOR.FFN_DOWN_SHEXP:     "blk.{bid}.ffn_down_shexp",
    MODEL_TENSOR.FFN_UP_SHEXP:       "blk.{bid}.ffn_up_shexp",
    MODEL_TENSOR.FFN_ACT:            "blk.{bid}.ffn",
    MODEL_TENSOR.FFN_NORM_EXP:       "blk.{bid}.ffn_norm_exps",
    MODEL_TENSOR.FFN_GATE_EXP:       "blk.{bid}.ffn_gate_exps",
    MODEL_TENSOR.FFN_DOWN_EXP:       "blk.{bid}.ffn_down_exps",
    MODEL_TENSOR.FFN_UP_EXP:         "blk.{bid}.ffn_up_exps",
    MODEL_TENSOR.LAYER_OUT_NORM:     "blk.{bid}.layer_output_norm",
    MODEL_TENSOR.SSM_IN:             "blk.{bid}.ssm_in",
    MODEL_TENSOR.SSM_CONV1D:         "blk.{bid}.ssm_conv1d",
    MODEL_TENSOR.SSM_X:              "blk.{bid}.ssm_x",
    MODEL_TENSOR.SSM_DT:             "blk.{bid}.ssm_dt",
    MODEL_TENSOR.SSM_A:              "blk.{bid}.ssm_a",
    MODEL_TENSOR.SSM_D:              "blk.{bid}.ssm_d",
    MODEL_TENSOR.SSM_OUT:            "blk.{bid}.ssm_out",
    MODEL_TENSOR.ATTN_Q_A:           "blk.{bid}.attn_q_a",
    MODEL_TENSOR.ATTN_Q_B:           "blk.{bid}.attn_q_b",
    MODEL_TENSOR.ATTN_KV_A_MQA:      "blk.{bid}.attn_kv_a_mqa",
    MODEL_TENSOR.ATTN_KV_B:          "blk.{bid}.attn_kv_b",
    MODEL_TENSOR.ATTN_Q_A_NORM:      "blk.{bid}.attn_q_a_norm",
    MODEL_TENSOR.ATTN_KV_A_NORM:     "blk.{bid}.attn_kv_a_norm",
}

MODEL_TENSORS: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    MODEL_ARCH.LLAMA: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.GROK: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.ATTN_OUT_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.LAYER_OUT_NORM,
    ],
    MODEL_ARCH.GPTNEOX: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.FALCON: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_NORM_2,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BAICHUAN: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.STARCODER: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BERT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.TOKEN_TYPES,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_OUT_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.LAYER_OUT_NORM,
    ],
    MODEL_ARCH.NOMIC_BERT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.TOKEN_TYPES,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_OUT_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.LAYER_OUT_NORM,
    ],
    MODEL_ARCH.JINA_BERT_V2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.TOKEN_TYPES,
        MODEL_TENSOR.ATTN_OUT_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.LAYER_OUT_NORM,
    ],
    MODEL_ARCH.MPT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_ACT,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.POS_EMBD,
    ],
    MODEL_ARCH.GPTJ: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.REFACT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BLOOM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.STABLELM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
    ],
    MODEL_ARCH.QWEN: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.QWEN2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.QWEN2MOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_INP_SHEXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
    ],
    MODEL_ARCH.PLAMO: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.GPT2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.PHI2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.PHI3: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.CODESHELL: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.ORION: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.INTERNLM2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.MINICPM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.GEMMA: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_NORM,
    ],
    MODEL_ARCH.STARCODER2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.MAMBA: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.SSM_IN,
        MODEL_TENSOR.SSM_CONV1D,
        MODEL_TENSOR.SSM_X,
        MODEL_TENSOR.SSM_DT,
        MODEL_TENSOR.SSM_A,
        MODEL_TENSOR.SSM_D,
        MODEL_TENSOR.SSM_OUT,
    ],
    MODEL_ARCH.XVERSE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.COMMAND_R: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_Q_NORM,
    ],
    MODEL_ARCH.DBRX: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_OUT_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.OLMO: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.ARCTIC: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_NORM_EXP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.DEEPSEEK2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_A,
        MODEL_TENSOR.ATTN_Q_B,
        MODEL_TENSOR.ATTN_KV_A_MQA,
        MODEL_TENSOR.ATTN_KV_B,
        MODEL_TENSOR.ATTN_Q_A_NORM,
        MODEL_TENSOR.ATTN_KV_A_NORM,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
    ],
    # TODO
}

# tensors that will not be serialized
MODEL_TENSOR_SKIP: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    MODEL_ARCH.LLAMA: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.BAICHUAN: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.QWEN: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.CODESHELL: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.ORION: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.STARCODER2: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.XVERSE: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.DEEPSEEK2: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
}


#
# types
#
class GGMLRopeScalingType(Enum):
    NONE   = 'none'
    LINEAR = 'linear'
    YARN   = 'yarn'


class GGMLPoolingType(IntEnum):
    NONE = 0
    MEAN = 1
    CLS  = 2


class GGMLQuantizationType(IntEnum):
    F32     = 0
    F16     = 1
    Q4_0    = 2
    Q4_1    = 3
    Q5_0    = 6
    Q5_1    = 7
    Q8_0    = 8
    Q8_1    = 9
    Q2_K    = 10
    Q3_K    = 11
    Q4_K    = 12
    Q5_K    = 13
    Q6_K    = 14
    Q8_K    = 15
    IQ2_XXS = 16
    IQ2_XS  = 17
    IQ3_XXS = 18
    IQ1_S   = 19
    IQ4_NL  = 20
    IQ3_S   = 21
    IQ2_S   = 22
    IQ4_XS  = 23
    I8      = 24
    I16     = 25
    I32     = 26
    I64     = 27
    F64     = 28
    IQ1_M   = 29
    BF16    = 30


# TODO: add GGMLFileType from ggml_ftype in ggml.h


# from llama_ftype in llama.h
# ALL VALUES SHOULD BE THE SAME HERE AS THEY ARE OVER THERE.
class GGUFFileType(IntEnum):
    ALL_F32              = 0
    MOSTLY_F16           = 1   # except 1d tensors
    MOSTLY_Q4_0          = 2   # except 1d tensors
    MOSTLY_Q4_1          = 3   # except 1d tensors
    MOSTLY_Q4_1_SOME_F16 = 4   # tok_embeddings.weight and output.weight are F16
    # MOSTLY_Q4_2        = 5   # support has been removed
    # MOSTLY_Q4_3        = 6   # support has been removed
    MOSTLY_Q8_0          = 7   # except 1d tensors
    MOSTLY_Q5_0          = 8   # except 1d tensors
    MOSTLY_Q5_1          = 9   # except 1d tensors
    MOSTLY_Q2_K          = 10  # except 1d tensors
    MOSTLY_Q3_K_S        = 11  # except 1d tensors
    MOSTLY_Q3_K_M        = 12  # except 1d tensors
    MOSTLY_Q3_K_L        = 13  # except 1d tensors
    MOSTLY_Q4_K_S        = 14  # except 1d tensors
    MOSTLY_Q4_K_M        = 15  # except 1d tensors
    MOSTLY_Q5_K_S        = 16  # except 1d tensors
    MOSTLY_Q5_K_M        = 17  # except 1d tensors
    MOSTLY_Q6_K          = 18  # except 1d tensors
    MOSTLY_IQ2_XXS       = 19  # except 1d tensors
    MOSTLY_IQ2_XS        = 20  # except 1d tensors
    MOSTLY_Q2_K_S        = 21  # except 1d tensors
    MOSTLY_IQ3_XS        = 22  # except 1d tensors
    MOSTLY_IQ3_XXS       = 23  # except 1d tensors
    MOSTLY_IQ1_S         = 24  # except 1d tensors
    MOSTLY_IQ4_NL        = 25  # except 1d tensors
    MOSTLY_IQ3_S         = 26  # except 1d tensors
    MOSTLY_IQ3_M         = 27  # except 1d tensors
    MOSTLY_IQ2_S         = 28  # except 1d tensors
    MOSTLY_IQ2_M         = 29  # except 1d tensors
    MOSTLY_IQ4_XS        = 30  # except 1d tensors
    MOSTLY_IQ1_M         = 31  # except 1d tensors
    MOSTLY_BF16          = 32  # except 1d tensors

    GUESSED              = 1024  # not specified in the model file


GGUF_FILE_TYPE_MAP: dict[str, GGUFFileType] = {
    "F32"  : GGUFFileType.ALL_F32,
    "F16"  : GGUFFileType.MOSTLY_F16,
    "BF16" : GGUFFileType.MOSTLY_BF16,
    "Q8_0" : GGUFFileType.MOSTLY_Q8_0,
}


GGUF_FILE_TYPE_NAMES: dict[GGUFFileType, str] = {
    GGUFFileType.ALL_F32     : "F32",
    GGUFFileType.MOSTLY_F16  : "F16",
    GGUFFileType.MOSTLY_BF16 : "BF16",
    GGUFFileType.MOSTLY_Q8_0 : "Q8_0",
}


class GGUFEndian(IntEnum):
    LITTLE = 0
    BIG    = 1


class GGUFValueType(IntEnum):
    UINT8   = auto()
    INT8    = auto()
    UINT16  = auto()
    INT16   = auto()
    UINT32  = auto()
    INT32   = auto()
    UINT64  = auto()
    INT64   = auto()
    FLOAT32 = auto()
    FLOAT64 = auto()
    BOOL    = auto()
    STRING  = auto()
    ARRAY   = auto()
    OBJECT  = auto()

    @staticmethod
    def get_type(val: Any) -> GGUFValueType:
        if isinstance(val, (str, bytes, bytearray)):
            return GGUFValueType.STRING

        elif isinstance(val, bool):
            return GGUFValueType.BOOL

        # TODO: Need help with 64-bit types in Python.
        # NOTE: Maybe use numpy, e.g. np.dtypes to determine data type?
        # Using base types is unreliable in python as all numbers in python are 64-bits.

        # If it's an integer (either signed or unsigned)
        if isinstance(val, int):
            return GGUFValueType.INT32

        elif isinstance(val, float):
            # NOTE: This is unreliable in python as all numbers in python are 64-bits
            return GGUFValueType.FLOAT32

        elif isinstance(val, list):
            return GGUFValueType.ARRAY

        elif isinstance(val, dict):
            # NOTE: JSON Object, Dict, or Mapping are valid types
            return GGUFValueType.OBJECT

        else:
            raise ValueError(f"Unknown type: {type(val)}")


# Items here are (block size, type size)
QK_K = 256
GGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {
    GGMLQuantizationType.F32:     (1, 4),
    GGMLQuantizationType.F16:     (1, 2),
    GGMLQuantizationType.Q4_0:    (32, 2 + 16),
    GGMLQuantizationType.Q4_1:    (32, 2 + 2 + 16),
    GGMLQuantizationType.Q5_0:    (32, 2 + 4 + 16),
    GGMLQuantizationType.Q5_1:    (32, 2 + 2 + 4 + 16),
    GGMLQuantizationType.Q8_0:    (32, 2 + 32),
    GGMLQuantizationType.Q8_1:    (32, 4 + 4 + 32),
    GGMLQuantizationType.Q2_K:    (256, 2 + 2 + QK_K // 16 + QK_K // 4),
    GGMLQuantizationType.Q3_K:    (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    GGMLQuantizationType.Q4_K:    (256, 2 + 2 + QK_K // 2 + 12),
    GGMLQuantizationType.Q5_K:    (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    GGMLQuantizationType.Q6_K:    (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.Q8_K:    (256, 4 + QK_K + QK_K // 8),
    GGMLQuantizationType.IQ2_XXS: (256, 2 + QK_K // 4),
    GGMLQuantizationType.IQ2_XS:  (256, 2 + QK_K // 4 + QK_K // 32),
    GGMLQuantizationType.IQ3_XXS: (256, 2 + QK_K // 4 + QK_K // 8),
    GGMLQuantizationType.IQ1_S:   (256, 2 + QK_K // 8 + QK_K // 16),
    GGMLQuantizationType.IQ4_NL:  (32, 2 + 16),
    GGMLQuantizationType.IQ3_S:   (256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    GGMLQuantizationType.IQ2_S:   (256, 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.IQ4_XS:  (256, 2 + 2 + QK_K // 2 + QK_K // 64),
    GGMLQuantizationType.I8:      (1, 1),
    GGMLQuantizationType.I16:     (1, 2),
    GGMLQuantizationType.I32:     (1, 4),
    GGMLQuantizationType.I64:     (1, 8),
    GGMLQuantizationType.F64:     (1, 8),
    GGMLQuantizationType.IQ1_M:   (256, QK_K // 8 + QK_K // 16  + QK_K // 32),
    GGMLQuantizationType.BF16:    (1, 2),
}


#
# Model File Types
#
class ModelFileExtension(Enum):
    PT          = ".pt"           # torch
    PTH         = ".pth"          # torch
    BIN         = ".bin"          # torch
    SAFETENSORS = ".safetensors"  # safetensors
    JSON        = ".json"         # transformers/tokenizers
    MODEL       = ".model"        # sentencepiece
    GGUF        = ".gguf"         # ggml/llama.cpp


#
# Tokenizer Types
#
class GGUFTokenType(IntEnum):
    NORMAL       = 1
    UNKNOWN      = 2
    CONTROL      = 3
    USER_DEFINED = 4
    UNUSED       = 5
    BYTE         = 6


class HFTokenizerType(Enum):
    SPM = "SPM"  # SentencePiece LLaMa tokenizer
    BPE = "BPE"  # BytePair GPT-2 tokenizer
    WPM = "WPM"  # WordPiece BERT tokenizer


#
# Normalizer Types
#
class HFNormalizerType(Enum):
    SEQUENCE = "Sequence"
    NFC      = "NFC"
    NFD      = "NFD"
    NFKC     = "NFKC"
    NFKD     = "NFKD"


#
# Pre-tokenizer Types
#
class HFPreTokenizerType(Enum):
    WHITESPACE         = "Whitespace"
    METASPACE          = "Metaspace"
    BYTE_LEVEL         = "ByteLevel"
    BERT_PRE_TOKENIZER = "BertPreTokenizer"
    SEQUENCE           = "Sequence"


#
# HF Vocab Files
#
HF_TOKENIZER_BPE_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
)

HF_TOKENIZER_SPM_FILES: tuple[str, ...] = HF_TOKENIZER_BPE_FILES + ("tokenizer.model",)

#
# Pre-tokenization Regular Expressions
#

# NOTE: `tokenizers` defaults to OpenAI GPT-2 `ByteLevel` RegEx.
# The pattern uses perl regex and formatting is arbitrary.
# https://github.com/openai/gpt-2/blob/master/src/encoder.py#L53
# https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs#L40-L42

# These are fallback values if the pre-tokenizer cannot be dynamically discovered at runtime.
GPT_PRE_TOKENIZER_DEFAULT = ("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",)

# Aliases for backward compatibility.

# general
KEY_GENERAL_ARCHITECTURE         = GGUFMetadataKeys.General.ARCHITECTURE
KEY_GENERAL_QUANTIZATION_VERSION = GGUFMetadataKeys.General.QUANTIZATION_VERSION
KEY_GENERAL_ALIGNMENT            = GGUFMetadataKeys.General.ALIGNMENT
KEY_GENERAL_NAME                 = GGUFMetadataKeys.General.NAME
KEY_GENERAL_AUTHOR               = GGUFMetadataKeys.General.AUTHOR
KEY_GENERAL_URL                  = GGUFMetadataKeys.General.URL
KEY_GENERAL_DESCRIPTION          = GGUFMetadataKeys.General.DESCRIPTION
KEY_GENERAL_LICENSE              = GGUFMetadataKeys.General.LICENSE
KEY_GENERAL_SOURCE_URL           = GGUFMetadataKeys.General.SOURCE_URL
KEY_GENERAL_SOURCE_REPO          = GGUFMetadataKeys.General.SOURCE_REPO
KEY_GENERAL_FILE_TYPE            = GGUFMetadataKeys.General.FILE_TYPE
KEY_GENERAL_ENDIANESS            = GGUFMetadataKeys.General.ENDIANESS

# LLM
KEY_VOCAB_SIZE                   = GGUFMetadataKeys.LLM.VOCAB_SIZE
KEY_CONTEXT_LENGTH               = GGUFMetadataKeys.LLM.CONTEXT_LENGTH
KEY_EMBEDDING_LENGTH             = GGUFMetadataKeys.LLM.EMBEDDING_LENGTH
KEY_BLOCK_COUNT                  = GGUFMetadataKeys.LLM.BLOCK_COUNT
KEY_FEED_FORWARD_LENGTH          = GGUFMetadataKeys.LLM.FEED_FORWARD_LENGTH
KEY_USE_PARALLEL_RESIDUAL        = GGUFMetadataKeys.LLM.USE_PARALLEL_RESIDUAL
KEY_TENSOR_DATA_LAYOUT           = GGUFMetadataKeys.LLM.TENSOR_DATA_LAYOUT

# attention
KEY_ATTENTION_HEAD_COUNT         = GGUFMetadataKeys.Attention.HEAD_COUNT
KEY_ATTENTION_HEAD_COUNT_KV      = GGUFMetadataKeys.Attention.HEAD_COUNT_KV
KEY_ATTENTION_MAX_ALIBI_BIAS     = GGUFMetadataKeys.Attention.MAX_ALIBI_BIAS
KEY_ATTENTION_CLAMP_KQV          = GGUFMetadataKeys.Attention.CLAMP_KQV
KEY_ATTENTION_LAYERNORM_EPS      = GGUFMetadataKeys.Attention.LAYERNORM_EPS
KEY_ATTENTION_LAYERNORM_RMS_EPS  = GGUFMetadataKeys.Attention.LAYERNORM_RMS_EPS

# RoPE
KEY_ROPE_DIMENSION_COUNT         = GGUFMetadataKeys.Rope.DIMENSION_COUNT
KEY_ROPE_FREQ_BASE               = GGUFMetadataKeys.Rope.FREQ_BASE
KEY_ROPE_SCALING_TYPE            = GGUFMetadataKeys.Rope.SCALING_TYPE
KEY_ROPE_SCALING_FACTOR          = GGUFMetadataKeys.Rope.SCALING_FACTOR
KEY_ROPE_SCALING_ORIG_CTX_LEN    = GGUFMetadataKeys.Rope.SCALING_ORIG_CTX_LEN
KEY_ROPE_SCALING_FINETUNED       = GGUFMetadataKeys.Rope.SCALING_FINETUNED

# SSM
KEY_SSM_CONV_KERNEL              = GGUFMetadataKeys.SSM.CONV_KERNEL
KEY_SSM_INNER_SIZE               = GGUFMetadataKeys.SSM.INNER_SIZE
KEY_SSM_STATE_SIZE               = GGUFMetadataKeys.SSM.STATE_SIZE
KEY_SSM_TIME_STEP_RANK           = GGUFMetadataKeys.SSM.TIME_STEP_RANK

# tokenization
KEY_TOKENIZER_MODEL              = GGUFMetadataKeys.Tokenizer.MODEL
KEY_TOKENIZER_TYPE               = GGUFMetadataKeys.Tokenizer.TYPE
KEY_TOKENIZER_NORM               = GGUFMetadataKeys.Tokenizer.NORM
KEY_TOKENIZER_PRE                = GGUFMetadataKeys.Tokenizer.PRE
KEY_TOKENIZER_ADDED              = GGUFMetadataKeys.Tokenizer.ADDED
KEY_TOKENIZER_VOCAB              = GGUFMetadataKeys.Tokenizer.VOCAB
KEY_TOKENIZER_MERGES             = GGUFMetadataKeys.Tokenizer.MERGES
KEY_TOKENIZER_TOKEN_TYPE         = GGUFMetadataKeys.Tokenizer.TOKEN_TYPE
KEY_TOKENIZER_TOKEN_TYPE_COUNT   = GGUFMetadataKeys.Tokenizer.TOKEN_TYPE_COUNT
KEY_TOKENIZER_SCORES             = GGUFMetadataKeys.Tokenizer.SCORES
KEY_TOKENIZER_BOS_ID             = GGUFMetadataKeys.Tokenizer.BOS_ID
KEY_TOKENIZER_EOS_ID             = GGUFMetadataKeys.Tokenizer.EOS_ID
KEY_TOKENIZER_UNK_ID             = GGUFMetadataKeys.Tokenizer.UNK_ID
KEY_TOKENIZER_SEP_ID             = GGUFMetadataKeys.Tokenizer.SEP_ID
KEY_TOKENIZER_PAD_ID             = GGUFMetadataKeys.Tokenizer.PAD_ID
KEY_TOKENIZER_CLS_ID             = GGUFMetadataKeys.Tokenizer.CLS_ID
KEY_TOKENIZER_MASK_ID            = GGUFMetadataKeys.Tokenizer.MASK_ID
KEY_TOKENIZER_ADD_BOS            = GGUFMetadataKeys.Tokenizer.ADD_BOS
KEY_TOKENIZER_ADD_EOS            = GGUFMetadataKeys.Tokenizer.ADD_EOS
KEY_TOKENIZER_ADD_PREFIX         = GGUFMetadataKeys.Tokenizer.ADD_PREFIX
KEY_TOKENIZER_RWKV               = GGUFMetadataKeys.Tokenizer.RWKV
KEY_TOKENIZER_CHAT_TEMPLATE      = GGUFMetadataKeys.Tokenizer.CHAT_TEMPLATE
KEY_TOKENIZER_CHAT_TEMPLATE_N    = GGUFMetadataKeys.Tokenizer.CHAT_TEMPLATE_N
KEY_TOKENIZER_CHAT_TEMPLATES     = GGUFMetadataKeys.Tokenizer.CHAT_TEMPLATES
KEY_TOKENIZER_PRIFIX_ID          = GGUFMetadataKeys.Tokenizer.PREFIX_ID
KEY_TOKENIZER_SUFFIX_ID          = GGUFMetadataKeys.Tokenizer.SUFFIX_ID
KEY_TOKENIZER_MIDDLE_ID          = GGUFMetadataKeys.Tokenizer.MIDDLE_ID
KEY_TOKENIZER_EOT_ID             = GGUFMetadataKeys.Tokenizer.EOT_ID
