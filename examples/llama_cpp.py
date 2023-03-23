import ctypes

from ctypes import (
    c_int,
    c_float,
    c_double,
    c_char_p,
    c_void_p,
    c_bool,
    POINTER,
    Structure,
)

import pathlib

# Load the library
libfile = pathlib.Path(__file__).parent / "libllama.so"
_lib = ctypes.CDLL(str(libfile))


# C types
llama_token = c_int
llama_token_p = POINTER(llama_token)


class llama_token_data(Structure):
    _fields_ = [
        ("id", llama_token),  # token id
        ("p", c_float),  # probability of the token
        ("plog", c_float),  # log probability of the token
    ]


llama_token_data_p = POINTER(llama_token_data)


class llama_context_params(Structure):
    _fields_ = [
        ("n_ctx", c_int),  # text context
        ("n_parts", c_int),  # -1 for default
        ("seed", c_int),  # RNG seed, 0 for random
        ("f16_kv", c_bool),  # use fp16 for KV cache
        (
            "logits_all",
            c_bool,
        ),  # the llama_eval() call computes all logits, not just the last one
        ("vocab_only", c_bool),  # only load the vocabulary, no weights
    ]


llama_context_params_p = POINTER(llama_context_params)

llama_context_p = c_void_p

# C functions
lib.llama_context_default_params.argtypes = []
lib.llama_context_default_params.restype = llama_context_params

lib.llama_init_from_file.argtypes = [c_char_p, llama_context_params]
lib.llama_init_from_file.restype = llama_context_p

lib.llama_free.argtypes = [llama_context_p]
lib.llama_free.restype = None

lib.llama_model_quantize.argtypes = [c_char_p, c_char_p, c_int, c_int]
lib.llama_model_quantize.restype = c_int

lib.llama_eval.argtypes = [llama_context_p, llama_token_p, c_int, c_int, c_int]
lib.llama_eval.restype = c_int

lib.llama_tokenize.argtypes = [llama_context_p, c_char_p, llama_token_p, c_int, c_bool]
lib.llama_tokenize.restype = c_int

lib.llama_n_vocab.argtypes = [llama_context_p]
lib.llama_n_vocab.restype = c_int

lib.llama_n_ctx.argtypes = [llama_context_p]
lib.llama_n_ctx.restype = c_int

lib.llama_get_logits.argtypes = [llama_context_p]
lib.llama_get_logits.restype = POINTER(c_float)

lib.llama_token_to_str.argtypes = [llama_context_p, llama_token]
lib.llama_token_to_str.restype = c_char_p

lib.llama_token_bos.argtypes = []
lib.llama_token_bos.restype = llama_token

lib.llama_token_eos.argtypes = []
lib.llama_token_eos.restype = llama_token

lib.llama_sample_top_p_top_k.argtypes = [
    llama_context_p,
    llama_token_p,
    c_int,
    c_int,
    c_double,
    c_double,
    c_double,
]
lib.llama_sample_top_p_top_k.restype = llama_token

lib.llama_print_timings.argtypes = [llama_context_p]
lib.llama_print_timings.restype = None

lib.llama_reset_timings.argtypes = [llama_context_p]
lib.llama_reset_timings.restype = None

lib.llama_print_system_info.argtypes = []
lib.llama_print_system_info.restype = c_char_p


# Python functions
def llama_context_default_params() -> llama_context_params:
    return _lib.llama_context_default_params()


def llama_init_from_file(
    path_model: bytes, params: llama_context_params
) -> llama_context_p:
    """Various functions for loading a ggml llama model.
    Allocate (almost) all memory needed for the model.
    Return NULL on failure"""
    return _lib.llama_init_from_file(path_model, params)


def llama_free(ctx: llama_context_p):
    """Free all allocated memory"""
    return _lib.llama_free(ctx)


def llama_model_quantize(
    fname_inp: bytes, fname_out: bytes, itype: c_int, qk: c_int
) -> c_int:
    """Returns 0 on success"""
    return _lib.llama_model_quantize(fname_inp, fname_out, itype, qk)


def llama_eval(
    ctx: llama_context_p,
    tokens: llama_token_p,
    n_tokens: c_int,
    n_past: c_int,
    n_threads: c_int,
) -> c_int:
    """Run the llama inference to obtain the logits and probabilities for the next token.
    tokens + n_tokens is the provided batch of new tokens to process
    n_past is the number of tokens to use from previous eval calls
    Returns 0 on success"""
    return _lib.llama_eval(ctx, tokens, n_tokens, n_past, n_threads)


def llama_tokenize(
    ctx: llama_context_p,
    text: bytes,
    tokens: llama_token_p,
    n_max_tokens: c_int,
    add_bos: c_bool,
) -> c_int:
    return _lib.llama_tokenize(ctx, text, tokens, n_max_tokens, add_bos)


def llama_n_vocab(ctx: llama_context_p) -> c_int:
    return _lib.llama_n_vocab(ctx)


def llama_n_ctx(ctx: llama_context_p) -> c_int:
    return _lib.llama_n_ctx(ctx)


def llama_get_logits(ctx: llama_context_p):
    """Token logits obtained from the last call to llama_eval()
    The logits for the last token are stored in the last row
    Can be mutated in order to change the probabilities of the next token
    Rows: n_tokens
    Cols: n_vocab"""
    return _lib.llama_get_logits(ctx)


def llama_token_to_str(ctx: llama_context_p, token: int) -> bytes:
    """Token Id -> String. Uses the vocabulary in the provided context"""
    return _lib.llama_token_to_str(ctx, token)


def llama_token_bos() -> llama_token:
    return _lib.llama_token_bos()


def llama_token_eos() -> llama_token:
    return _lib.llama_token_eos()


def llama_sample_top_p_top_k(
    ctx: llama_context_p,
    last_n_tokens_data: llama_token_p,
    last_n_tokens_size: c_int,
    top_k: c_int,
    top_p: c_double,
    temp: c_double,
    repeat_penalty: c_double,
) -> llama_token:
    return _lib.llama_sample_top_p_top_k(
        ctx, last_n_tokens_data, last_n_tokens_size, top_k, top_p, temp, repeat_penalty
    )


def llama_print_timings(ctx: llama_context_p):
    _lib.llama_print_timings(ctx)


def llama_reset_timings(ctx: llama_context_p):
    _lib.llama_reset_timings(ctx)


def llama_print_system_info() -> bytes:
    return _lib.llama_print_system_info()
