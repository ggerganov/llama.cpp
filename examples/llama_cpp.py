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
    c_size_t,
)
import pathlib


# Load the library
def _load_shared_library(lib_base_name: str):
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
    # Searching for the library in the current directory under the name "libllama" (default name
    # for llamacpp) and "llama" (default name for this repo)
    _lib_paths = [
        _base_path / f"lib{lib_base_name}{lib_ext}",
        _base_path / f"{lib_base_name}{lib_ext}",
    ]

    if "LLAMA_CPP_LIB" in os.environ:
        lib_base_name = os.environ["LLAMA_CPP_LIB"]
        _lib = pathlib.Path(lib_base_name)
        _base_path = _lib.parent.resolve()
        _lib_paths = [_lib.resolve()]

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

    raise FileNotFoundError(
        f"Shared library with base name '{lib_base_name}' not found"
    )


# Specify the base name of the shared library to load
_lib_base_name = "llama"

# Load the library
_lib = _load_shared_library(_lib_base_name)

# C types
LLAMA_FILE_VERSION = c_int(1)
LLAMA_FILE_MAGIC = b"ggjt"
LLAMA_FILE_MAGIC_UNVERSIONED = b"ggml"
LLAMA_SESSION_MAGIC = b"ggsn"
LLAMA_SESSION_VERSION = c_int(1)

llama_context_p = c_void_p


llama_token = c_int
llama_token_p = POINTER(llama_token)


class llama_token_data(Structure):
    _fields_ = [
        ("id", llama_token),  # token id
        ("logit", c_float),  # log-odds of the token
        ("p", c_float),  # probability of the token
    ]


llama_token_data_p = POINTER(llama_token_data)


class llama_token_data_array(Structure):
    _fields_ = [
        ("data", llama_token_data_p),
        ("size", c_size_t),
        ("sorted", c_bool),
    ]


llama_token_data_array_p = POINTER(llama_token_data_array)

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
        ("use_mmap", c_bool),  # use mmap if possible
        ("use_mlock", c_bool),  # force system to keep model in RAM
        ("embedding", c_bool),  # embedding mode only
        # called with a progress value between 0 and 1, pass NULL to disable
        ("progress_callback", llama_progress_callback),
        # context pointer passed to the progress callback
        ("progress_callback_user_data", c_void_p),
    ]


llama_context_params_p = POINTER(llama_context_params)

LLAMA_FTYPE_ALL_F32 = c_int(0)
LLAMA_FTYPE_MOSTLY_F16 = c_int(1)  # except 1d tensors
LLAMA_FTYPE_MOSTLY_Q4_0 = c_int(2)  # except 1d tensors
LLAMA_FTYPE_MOSTLY_Q4_1 = c_int(3)  # except 1d tensors
LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = c_int(
    4
)  # tok_embeddings.weight and output.weight are F16
LLAMA_FTYPE_MOSTLY_Q4_2 = c_int(5)  # except 1d tensors
# LLAMA_FTYPE_MOSTLY_Q4_3 = c_int(6)  # except 1d tensors
LLAMA_FTYPE_MOSTLY_Q8_0 = c_int(7)  # except 1d tensors
LLAMA_FTYPE_MOSTLY_Q5_0 = c_int(8)  # except 1d tensors
LLAMA_FTYPE_MOSTLY_Q5_1 = c_int(9)  # except 1d tensors

# Functions


def llama_context_default_params() -> llama_context_params:
    return _lib.llama_context_default_params()


_lib.llama_context_default_params.argtypes = []
_lib.llama_context_default_params.restype = llama_context_params


def llama_mmap_supported() -> c_bool:
    return _lib.llama_mmap_supported()


_lib.llama_mmap_supported.argtypes = []
_lib.llama_mmap_supported.restype = c_bool


def llama_mlock_supported() -> c_bool:
    return _lib.llama_mlock_supported()


_lib.llama_mlock_supported.argtypes = []
_lib.llama_mlock_supported.restype = c_bool


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
# nthread - how many threads to use. If <=0, will use std::thread::hardware_concurrency(), else the number given
def llama_model_quantize(
    fname_inp: bytes, fname_out: bytes, ftype: c_int, nthread: c_int
) -> c_int:
    return _lib.llama_model_quantize(fname_inp, fname_out, ftype, nthread)


_lib.llama_model_quantize.argtypes = [c_char_p, c_char_p, c_int, c_int]
_lib.llama_model_quantize.restype = c_int


# Apply a LoRA adapter to a loaded model
# path_base_model is the path to a higher quality model to use as a base for
# the layers modified by the adapter. Can be NULL to use the current loaded model.
# The model needs to be reloaded before applying a new adapter, otherwise the adapter
# will be applied on top of the previous one
# Returns 0 on success
def llama_apply_lora_from_file(
    ctx: llama_context_p,
    path_lora: c_char_p,
    path_base_model: c_char_p,
    n_threads: c_int,
) -> c_int:
    return _lib.llama_apply_lora_from_file(ctx, path_lora, path_base_model, n_threads)


_lib.llama_apply_lora_from_file.argtypes = [llama_context_p, c_char_p, c_char_p, c_int]
_lib.llama_apply_lora_from_file.restype = c_int


# Returns the number of tokens in the KV cache
def llama_get_kv_cache_token_count(ctx: llama_context_p) -> c_int:
    return _lib.llama_get_kv_cache_token_count(ctx)


_lib.llama_get_kv_cache_token_count.argtypes = [llama_context_p]
_lib.llama_get_kv_cache_token_count.restype = c_int


# Sets the current rng seed.
def llama_set_rng_seed(ctx: llama_context_p, seed: c_int):
    return _lib.llama_set_rng_seed(ctx, seed)


_lib.llama_set_rng_seed.argtypes = [llama_context_p, c_int]
_lib.llama_set_rng_seed.restype = None


# Returns the maximum size in bytes of the state (rng, logits, embedding
# and kv_cache) - will often be smaller after compacting tokens
def llama_get_state_size(ctx: llama_context_p) -> c_size_t:
    return _lib.llama_get_state_size(ctx)


_lib.llama_get_state_size.argtypes = [llama_context_p]
_lib.llama_get_state_size.restype = c_size_t


# Copies the state to the specified destination address.
# Destination needs to have allocated enough memory.
# Returns the number of bytes copied
def llama_copy_state_data(
    ctx: llama_context_p, dest  # type: Array[c_uint8]
) -> c_size_t:
    return _lib.llama_copy_state_data(ctx, dest)


_lib.llama_copy_state_data.argtypes = [llama_context_p, POINTER(c_uint8)]
_lib.llama_copy_state_data.restype = c_size_t


# Set the state reading from the specified address
# Returns the number of bytes read
def llama_set_state_data(
    ctx: llama_context_p, src  # type: Array[c_uint8]
) -> c_size_t:
    return _lib.llama_set_state_data(ctx, src)


_lib.llama_set_state_data.argtypes = [llama_context_p, POINTER(c_uint8)]
_lib.llama_set_state_data.restype = c_size_t


# Save/load session file
def llama_load_session_file(
    ctx: llama_context_p,
    path_session: bytes,
    tokens_out,  # type: Array[llama_token]
    n_token_capacity: c_size_t,
    n_token_count_out,  # type: Array[c_size_t]
) -> c_size_t:
    return _lib.llama_load_session_file(
        ctx, path_session, tokens_out, n_token_capacity, n_token_count_out
    )


_lib.llama_load_session_file.argtypes = [
    llama_context_p,
    c_char_p,
    llama_token_p,
    c_size_t,
    POINTER(c_size_t),
]
_lib.llama_load_session_file.restype = c_size_t


def llama_save_session_file(
    ctx: llama_context_p,
    path_session: bytes,
    tokens,  # type: Array[llama_token]
    n_token_count: c_size_t,
) -> c_size_t:
    return _lib.llama_save_session_file(ctx, path_session, tokens, n_token_count)


_lib.llama_save_session_file.argtypes = [
    llama_context_p,
    c_char_p,
    llama_token_p,
    c_size_t,
]
_lib.llama_save_session_file.restype = c_size_t


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
) -> int:
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
def llama_get_logits(ctx: llama_context_p):  # type: (...) -> Array[float] # type: ignore
    return _lib.llama_get_logits(ctx)


_lib.llama_get_logits.argtypes = [llama_context_p]
_lib.llama_get_logits.restype = POINTER(c_float)


# Get the embeddings for the input
# shape: [n_embd] (1-dimensional)
def llama_get_embeddings(ctx: llama_context_p):  # type: (...) -> Array[float] # type: ignore
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


def llama_token_nl() -> llama_token:
    return _lib.llama_token_nl()


_lib.llama_token_nl.argtypes = []
_lib.llama_token_nl.restype = llama_token


# Sampling functions


# @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
def llama_sample_repetition_penalty(
    ctx: llama_context_p,
    candidates, # type: Array[llama_token_data]
    last_tokens_data, # type: Array[llama_token]
    last_tokens_size: c_int,
    penalty: c_float,
):
    return _lib.llama_sample_repetition_penalty(
        ctx, candidates, last_tokens_data, last_tokens_size, penalty
    )


_lib.llama_sample_repetition_penalty.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    llama_token_p,
    c_int,
    c_float,
]
_lib.llama_sample_repetition_penalty.restype = None


# @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
def llama_sample_frequency_and_presence_penalties(
    ctx: llama_context_p,
    candidates, # type: Array[llama_token_data]
    last_tokens_data, # type: Array[llama_token]
    last_tokens_size: c_int,
    alpha_frequency: c_float,
    alpha_presence: c_float,
):
    return _lib.llama_sample_frequency_and_presence_penalties(
        ctx,
        candidates,
        last_tokens_data,
        last_tokens_size,
        alpha_frequency,
        alpha_presence,
    )


_lib.llama_sample_frequency_and_presence_penalties.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    llama_token_p,
    c_int,
    c_float,
    c_float,
]
_lib.llama_sample_frequency_and_presence_penalties.restype = None


# @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
def llama_sample_softmax(
    ctx: llama_context_p,
    candidates # type: Array[llama_token_data]
):
    return _lib.llama_sample_softmax(ctx, candidates)


_lib.llama_sample_softmax.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
]
_lib.llama_sample_softmax.restype = None


# @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
def llama_sample_top_k(
    ctx: llama_context_p,
    candidates,  # type: Array[llama_token_data]
    k: c_int,
    min_keep: c_size_t = c_size_t(1)
):
    return _lib.llama_sample_top_k(ctx, candidates, k, min_keep)


_lib.llama_sample_top_k.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_int,
    c_size_t,
]
_lib.llama_sample_top_k.restype = None


# @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
def llama_sample_top_p(
    ctx: llama_context_p,
    candidates, # type: Array[llama_token_data]
    p: c_float,
    min_keep: c_size_t = c_size_t(1)
):
    return _lib.llama_sample_top_p(ctx, candidates, p, min_keep)


_lib.llama_sample_top_p.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_float,
    c_size_t,
]
_lib.llama_sample_top_p.restype = None


# @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
def llama_sample_tail_free(
    ctx: llama_context_p,
    candidates, # type: Array[llama_token_data]
    z: c_float,
    min_keep: c_size_t = c_size_t(1)
):
    return _lib.llama_sample_tail_free(ctx, candidates, z, min_keep)


_lib.llama_sample_tail_free.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_float,
    c_size_t,
]
_lib.llama_sample_tail_free.restype = None


# @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
def llama_sample_typical(
    ctx: llama_context_p,
    candidates, # type: Array[llama_token_data]
    p: c_float,
    min_keep: c_size_t = c_size_t(1)
):
    return _lib.llama_sample_typical(ctx, candidates, p, min_keep)


_lib.llama_sample_typical.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_float,
    c_size_t,
]
_lib.llama_sample_typical.restype = None


def llama_sample_temperature(
    ctx: llama_context_p,
    candidates, # type: Array[llama_token_data]
    temp: c_float
):
    return _lib.llama_sample_temperature(ctx, candidates, temp)


_lib.llama_sample_temperature.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_float,
]
_lib.llama_sample_temperature.restype = None


# @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
# @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
# @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
# @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
# @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
# @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
def llama_sample_token_mirostat(
    ctx: llama_context_p,
    candidates, # type: Array[llama_token_data]
    tau: c_float,
    eta: c_float,
    m: c_int,
    mu # type: Array[c_float]
) -> llama_token:
    return _lib.llama_sample_token_mirostat(ctx, candidates, tau, eta, m, mu)


_lib.llama_sample_token_mirostat.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_float,
    c_float,
    c_int,
    POINTER(c_float),
]
_lib.llama_sample_token_mirostat.restype = llama_token


# @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
# @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
# @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
# @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
# @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
def llama_sample_token_mirostat_v2(
    ctx: llama_context_p,
    candidates, # type: Array[llama_token_data]
    tau: c_float,
    eta: c_float,
    mu # type: Array[c_float]
) -> llama_token:
    return _lib.llama_sample_token_mirostat_v2(ctx, candidates, tau, eta, mu)


_lib.llama_sample_token_mirostat_v2.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_float,
    c_float,
    POINTER(c_float),
]
_lib.llama_sample_token_mirostat_v2.restype = llama_token


# @details Selects the token with the highest probability.
def llama_sample_token_greedy(
    ctx: llama_context_p,
    candidates # type: Array[llama_token_data]
) -> llama_token:
    return _lib.llama_sample_token_greedy(ctx, candidates)


_lib.llama_sample_token_greedy.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
]
_lib.llama_sample_token_greedy.restype = llama_token


# @details Randomly selects a token from the candidates based on their probabilities.
def llama_sample_token(
    ctx: llama_context_p,
    candidates # type: Array[llama_token_data]
) -> llama_token:
    return _lib.llama_sample_token(ctx, candidates)


_lib.llama_sample_token.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
]
_lib.llama_sample_token.restype = llama_token


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
