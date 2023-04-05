import sys
import os
import ctypes
from ctypes import (
    c_int,
    c_float,
    c_char_p,
    c_void_p,
    c_bool,
    POINTER,
    Structure,
    Array,
    c_uint8,
    c_size_t
)
import pathlib

# Load the library
def _load_shared_library(lib_base_name):
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        lib_ext = ".so"
    elif sys.platform == "darwin":
        lib_ext = ".dylib"
    elif sys.platform == "win32":
        lib_ext = ".dll"
    else:
        raise RuntimeError("Unsupported platform")

    # Construct the paths to the possible shared library names
    _base_path = pathlib.Path(__file__).parent.resolve()
    _local_path = pathlib.Path.cwd()
    # Searching for the library in the current directory under the name "libllama" (default name
    # for llamacpp) and "llama" (default name for this repo)
    _lib_paths = [
        _local_path / f"./lib{lib_base_name}{lib_ext}",
        _local_path / f"./{lib_base_name}{lib_ext}",
        _base_path / f"lib{lib_base_name}{lib_ext}",
        _base_path / f"{lib_base_name}{lib_ext}"
    ]

    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        os.add_dll_directory(str(_base_path))

    # Try to load the shared library, handling potential errors
    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path))
            except Exception as e:
                raise RuntimeError(f"Failed to load shared library '{_lib_path}': {e}")

    raise FileNotFoundError(f"Shared library with base name '{lib_base_name}' not found")

# Specify the base name of the shared library to load
_lib_base_name = "llama"

# Load the library
_lib = _load_shared_library(_lib_base_name)

# C types
llama_context_p = c_void_p


llama_token = c_int
llama_token_p = POINTER(llama_token)


class llama_token_data(Structure):
    _fields_ = [
        ("id", llama_token),  # token id
        ("p", c_float),  # probability of the token
        ("plog", c_float),  # log probability of the token
    ]


llama_token_data_p = POINTER(llama_token_data)

llama_progress_callback = ctypes.CFUNCTYPE(None, c_float, c_void_p)


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
        ("use_mlock", c_bool),  # force system to keep model in RAM
        ("embedding", c_bool),  # embedding mode only
        # called with a progress value between 0 and 1, pass NULL to disable
        ("progress_callback", llama_progress_callback),
        # context pointer passed to the progress callback
        ("progress_callback_user_data", c_void_p),
    ]


llama_context_params_p = POINTER(llama_context_params)


# Functions


def llama_context_default_params() -> llama_context_params:
    return _lib.llama_context_default_params()


_lib.llama_context_default_params.argtypes = []
_lib.llama_context_default_params.restype = llama_context_params


# Various functions for loading a ggml llama model.
# Allocate (almost) all memory needed for the model.
# Return NULL on failure
def llama_init_from_file(
    path_model: bytes, params: llama_context_params
) -> llama_context_p:
    return _lib.llama_init_from_file(path_model, params)


_lib.llama_init_from_file.argtypes = [c_char_p, llama_context_params]
_lib.llama_init_from_file.restype = llama_context_p


# Frees all allocated memory
def llama_free(ctx: llama_context_p):
    return _lib.llama_free(ctx)


_lib.llama_free.argtypes = [llama_context_p]
_lib.llama_free.restype = None


# TODO: not great API - very likely to change
# Returns 0 on success
def llama_model_quantize(
    fname_inp: bytes, fname_out: bytes, itype: c_int
) -> c_int:
    return _lib.llama_model_quantize(fname_inp, fname_out, itype)


_lib.llama_model_quantize.argtypes = [c_char_p, c_char_p, c_int]
_lib.llama_model_quantize.restype = c_int

# Returns the KV cache that will contain the context for the
# ongoing prediction with the model.
def llama_get_kv_cache(ctx: llama_context_p):
    return _lib.llama_get_kv_cache(ctx)

_lib.llama_get_kv_cache.argtypes = [llama_context_p]
_lib.llama_get_kv_cache.restype = POINTER(c_uint8)

# Returns the size of the KV cache
def llama_get_kv_cache_size(ctx: llama_context_p) -> c_size_t:
    return _lib.llama_get_kv_cache_size(ctx)

_lib.llama_get_kv_cache_size.argtypes = [llama_context_p]
_lib.llama_get_kv_cache_size.restype = c_size_t

# Returns the number of tokens in the KV cache
def llama_get_kv_cache_token_count(ctx: llama_context_p) -> c_int:
    return _lib.llama_get_kv_cache_token_count(ctx)

_lib.llama_get_kv_cache_token_count.argtypes = [llama_context_p]
_lib.llama_get_kv_cache_token_count.restype = c_int


# Sets the KV cache containing the current context for the model
def llama_set_kv_cache(ctx: llama_context_p, kv_cache, n_size: c_size_t, n_token_count: c_int):
    return _lib.llama_set_kv_cache(ctx, kv_cache, n_size, n_token_count)

_lib.llama_set_kv_cache.argtypes = [llama_context_p, POINTER(c_uint8), c_size_t, c_int]
_lib.llama_set_kv_cache.restype = None


# Run the llama inference to obtain the logits and probabilities for the next token.
# tokens + n_tokens is the provided batch of new tokens to process
# n_past is the number of tokens to use from previous eval calls
# Returns 0 on success
def llama_eval(
    ctx: llama_context_p,
    tokens,  # type: Array[llama_token]
    n_tokens: c_int,
    n_past: c_int,
    n_threads: c_int,
) -> c_int:
    return _lib.llama_eval(ctx, tokens, n_tokens, n_past, n_threads)


_lib.llama_eval.argtypes = [llama_context_p, llama_token_p, c_int, c_int, c_int]
_lib.llama_eval.restype = c_int


# Convert the provided text into tokens.
# The tokens pointer must be large enough to hold the resulting tokens.
# Returns the number of tokens on success, no more than n_max_tokens
# Returns a negative number on failure - the number of tokens that would have been returned
# TODO: not sure if correct
def llama_tokenize(
    ctx: llama_context_p,
    text: bytes,
    tokens,  # type: Array[llama_token]
    n_max_tokens: c_int,
    add_bos: c_bool,
) -> c_int:
    return _lib.llama_tokenize(ctx, text, tokens, n_max_tokens, add_bos)


_lib.llama_tokenize.argtypes = [llama_context_p, c_char_p, llama_token_p, c_int, c_bool]
_lib.llama_tokenize.restype = c_int


def llama_n_vocab(ctx: llama_context_p) -> c_int:
    return _lib.llama_n_vocab(ctx)


_lib.llama_n_vocab.argtypes = [llama_context_p]
_lib.llama_n_vocab.restype = c_int


def llama_n_ctx(ctx: llama_context_p) -> c_int:
    return _lib.llama_n_ctx(ctx)


_lib.llama_n_ctx.argtypes = [llama_context_p]
_lib.llama_n_ctx.restype = c_int


def llama_n_embd(ctx: llama_context_p) -> c_int:
    return _lib.llama_n_embd(ctx)


_lib.llama_n_embd.argtypes = [llama_context_p]
_lib.llama_n_embd.restype = c_int


# Token logits obtained from the last call to llama_eval()
# The logits for the last token are stored in the last row
# Can be mutated in order to change the probabilities of the next token
# Rows: n_tokens
# Cols: n_vocab
def llama_get_logits(ctx: llama_context_p):
    return _lib.llama_get_logits(ctx)


_lib.llama_get_logits.argtypes = [llama_context_p]
_lib.llama_get_logits.restype = POINTER(c_float)


# Get the embeddings for the input
# shape: [n_embd] (1-dimensional)
def llama_get_embeddings(ctx: llama_context_p):
    return _lib.llama_get_embeddings(ctx)


_lib.llama_get_embeddings.argtypes = [llama_context_p]
_lib.llama_get_embeddings.restype = POINTER(c_float)


# Token Id -> String. Uses the vocabulary in the provided context
def llama_token_to_str(ctx: llama_context_p, token: llama_token) -> bytes:
    return _lib.llama_token_to_str(ctx, token)


_lib.llama_token_to_str.argtypes = [llama_context_p, llama_token]
_lib.llama_token_to_str.restype = c_char_p

# Special tokens


def llama_token_bos() -> llama_token:
    return _lib.llama_token_bos()


_lib.llama_token_bos.argtypes = []
_lib.llama_token_bos.restype = llama_token


def llama_token_eos() -> llama_token:
    return _lib.llama_token_eos()


_lib.llama_token_eos.argtypes = []
_lib.llama_token_eos.restype = llama_token


# TODO: improve the last_n_tokens interface ?
def llama_sample_top_p_top_k(
    ctx: llama_context_p,
    last_n_tokens_data,  # type: Array[llama_token]
    last_n_tokens_size: c_int,
    top_k: c_int,
    top_p: c_float,
    temp: c_float,
    repeat_penalty: c_float,
) -> llama_token:
    return _lib.llama_sample_top_p_top_k(
        ctx, last_n_tokens_data, last_n_tokens_size, top_k, top_p, temp, repeat_penalty
    )


_lib.llama_sample_top_p_top_k.argtypes = [
    llama_context_p,
    llama_token_p,
    c_int,
    c_int,
    c_float,
    c_float,
    c_float,
]
_lib.llama_sample_top_p_top_k.restype = llama_token


# Performance information


def llama_print_timings(ctx: llama_context_p):
    _lib.llama_print_timings(ctx)


_lib.llama_print_timings.argtypes = [llama_context_p]
_lib.llama_print_timings.restype = None


def llama_reset_timings(ctx: llama_context_p):
    _lib.llama_reset_timings(ctx)


_lib.llama_reset_timings.argtypes = [llama_context_p]
_lib.llama_reset_timings.restype = None


# Print system information
def llama_print_system_info() -> bytes:
    return _lib.llama_print_system_info()


_lib.llama_print_system_info.argtypes = []
_lib.llama_print_system_info.restype = c_char_p
