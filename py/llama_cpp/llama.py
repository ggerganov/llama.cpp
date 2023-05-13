import os
import sys
import glob
import ctypes

from ctypes import c_int, c_float, c_double, c_char_p, c_void_p, c_bool, c_size_t, c_ubyte, POINTER, Structure


# Load the library
if sys.platform == 'win32':
    lib = ctypes.cdll.LoadLibrary(next(iter(glob.glob(os.path.join(os.path.dirname(__file__), '..', '..', '**', 'llama.dll'), recursive=True))))
else:
    lib = ctypes.cdll.LoadLibrary(next(iter(glob.glob(os.path.join(os.path.dirname(__file__), '..', '..', '**', 'libllama.so'), recursive=True))))


# C types
llama_token = c_int
llama_token_p = POINTER(llama_token)

class llama_token_data(Structure):
    _fields_ = [
        ('id',   llama_token), # token id
        ('p',    c_float), # probability of the token
        ('plog', c_float), # log probability of the token
    ]

llama_token_data_p = POINTER(llama_token_data)

class llama_token_data_array(Structure):
    _fields_ = [
        ('data',   llama_token_data_p),
        ('size',   c_size_t),
        ('sorted', c_bool),
    ]

llama_token_data_array_p = POINTER(llama_token_data_array)

llama_progress_callback = ctypes.CFUNCTYPE(None, c_float, c_void_p)
class llama_context_params(Structure):
    _fields_ = [
        ('n_ctx',        c_int),  # text context
        ('n_parts',      c_int),  # -1 for default
        ('n_gpu_layers', c_int),  # number of layers to store in VRAM
        ('seed',         c_int),  # RNG seed, 0 for random
        ('f16_kv',       c_bool), # use fp16 for KV cache
        ('logits_all',   c_bool), # the llama_eval() call computes all logits, not just the last one
        ('vocab_only',   c_bool), # only load the vocabulary, no weights
        ('use_mmap',     c_bool), # use mmap if possible
        ('use_mlock',    c_bool), # force system to keep model in RAM
        ('embedding',    c_bool), # embedding mode only
        ('progress_callback',           llama_progress_callback), # called with a progress value between 0 and 1, pass NULL to disable
        ('progress_callback_user_data', c_void_p),                # context pointer passed to the progress callback
    ]


llama_context_params_p = POINTER(llama_context_params)

llama_context_p = c_void_p

c_size_p = POINTER(c_size_t)
c_ubyte_p = POINTER(c_ubyte)
c_float_p = POINTER(c_float)

# C functions
lib.llama_context_default_params.argtypes = []
lib.llama_context_default_params.restype = llama_context_params

lib.llama_mmap_supported.argtypes = []
lib.llama_mmap_supported.restype = c_bool

lib.llama_mlock_supported.argtypes = []
lib.llama_mlock_supported.restype = c_bool

lib.llama_init_from_file.argtypes = [c_char_p, llama_context_params]
lib.llama_init_from_file.restype = llama_context_p

lib.llama_free.argtypes = [llama_context_p]
lib.llama_free.restype = None

lib.llama_model_quantize.argtypes = [c_char_p, c_char_p, c_int, c_int]
lib.llama_model_quantize.restype = c_int

lib.llama_apply_lora_from_file.argtypes = [llama_context_p, c_char_p, c_char_p, c_int]
lib.llama_apply_lora_from_file.restype = c_int

lib.llama_get_kv_cache_token_count.argtypes = [llama_context_p]
lib.llama_get_kv_cache_token_count.restype = c_int

lib.llama_set_rng_seed.argtypes = [llama_context_p, c_int]
lib.llama_set_rng_seed.restype = None

lib.llama_get_state_size.argtypes = [llama_context_p]
lib.llama_get_state_size.restype = c_size_t

lib.llama_copy_state_data.argtypes = [llama_context_p, c_ubyte_p]
lib.llama_copy_state_data.restype = c_size_t

lib.llama_set_state_data.argtypes = [llama_context_p, c_ubyte_p]
lib.llama_set_state_data.restype = c_size_t

lib.llama_load_session_file.argtypes = [llama_context_p, c_char_p, llama_token_p, c_size_t, c_size_p]
lib.llama_load_session_file.restype = c_bool

lib.llama_save_session_file.argtypes = [llama_context_p, c_char_p, llama_token_p, c_size_t]
lib.llama_save_session_file.restype = c_bool

lib.llama_eval.argtypes = [llama_context_p, llama_token_p, c_int, c_int, c_int]
lib.llama_eval.restype = c_int

lib.llama_tokenize.argtypes = [llama_context_p, c_char_p, llama_token_p, c_int, c_bool]
lib.llama_tokenize.restype = c_int

lib.llama_n_vocab.argtypes = [llama_context_p]
lib.llama_n_vocab.restype = c_int

lib.llama_n_ctx.argtypes = [llama_context_p]
lib.llama_n_ctx.restype = c_int

lib.llama_n_embd.argtypes = [llama_context_p]
lib.llama_n_embd.restype = c_int

lib.llama_get_logits.argtypes = [llama_context_p]
lib.llama_get_logits.restype = c_float_p

lib.llama_get_embeddings.argtypes = [llama_context_p]
lib.llama_get_embeddings.restype = c_float_p

lib.llama_token_to_str.argtypes = [llama_context_p, llama_token]
lib.llama_token_to_str.restype = c_char_p

lib.llama_token_bos.argtypes = []
lib.llama_token_bos.restype = llama_token

lib.llama_token_eos.argtypes = []
lib.llama_token_eos.restype = llama_token

lib.llama_token_nl.argtypes = []
lib.llama_token_nl.restype = llama_token

lib.llama_sample_repetition_penalty.argtypes = [llama_context_p, llama_token_data_array_p, llama_token_p, c_size_t, c_float]
lib.llama_sample_repetition_penalty.restype = None

lib.llama_sample_frequency_and_presence_penalties.argtypes = [llama_context_p, llama_token_data_array_p, llama_token_p, c_size_t, c_float, c_float]
lib.llama_sample_frequency_and_presence_penalties.restype = None

lib.llama_sample_softmax.argtypes = [llama_context_p, llama_token_data_array_p]
lib.llama_sample_softmax.restype = None

lib.llama_sample_top_k.argtypes = [llama_context_p, llama_token_data_array_p, c_int, c_size_t]
lib.llama_sample_top_k.restype = None

lib.llama_sample_top_p.argtypes = [llama_context_p, llama_token_data_array_p, c_float, c_size_t]
lib.llama_sample_top_p.restype = None

lib.llama_sample_tail_free.argtypes = [llama_context_p, llama_token_data_array_p, c_float, c_size_t]
lib.llama_sample_tail_free.restype = None

lib.llama_sample_typical.argtypes = [llama_context_p, llama_token_data_array_p, c_float, c_size_t]
lib.llama_sample_typical.restype = None

lib.llama_sample_temperature.argtypes = [llama_context_p, llama_token_data_array_p, c_float]
lib.llama_sample_temperature.restype = None

lib.llama_sample_token_mirostat.argtypes = [llama_context_p, llama_token_data_array_p, c_float, c_float, c_int, c_float_p]
lib.llama_sample_token_mirostat.restype = llama_token

lib.llama_sample_token_mirostat_v2.argtypes = [llama_context_p, llama_token_data_array_p, c_float, c_float, c_float_p]
lib.llama_sample_token_mirostat_v2.restype = llama_token

lib.llama_sample_token_greedy.argtypes = [llama_context_p, llama_token_data_array_p]
lib.llama_sample_token_greedy.restype = llama_token

lib.llama_sample_token.argtypes = [llama_context_p, llama_token_data_array_p]
lib.llama_sample_token.restype = llama_token

lib.llama_print_timings.argtypes = [llama_context_p]
lib.llama_print_timings.restype = None

lib.llama_reset_timings.argtypes = [llama_context_p]
lib.llama_reset_timings.restype = None

lib.llama_print_system_info.argtypes = []
lib.llama_print_system_info.restype = c_char_p


# Python functions
def llama_context_default_params() -> llama_context_params:
    params = lib.llama_context_default_params()
    return params

def llama_mmap_supported() -> bool:
    return lib.llama_mmap_supported()

def llama_mlock_supported() -> bool:
    return lib.llama_mlock_supported()

def llama_init_from_file(path_model: str, params: llama_context_params) -> llama_context_p:
    """Various functions for loading a ggml llama model.
    Allocate (almost) all memory needed for the model.
    Return NULL on failure """
    return lib.llama_init_from_file(path_model.encode('utf-8'), params)

def llama_free(ctx: llama_context_p):
    """Free all allocated memory"""
    lib.llama_free(ctx)

def llama_model_quantize(fname_inp: str, fname_out: str, itype: c_int, qk: c_int) -> c_int:
    """Returns 0 on success"""
    return lib.llama_model_quantize(fname_inp.encode('utf-8'), fname_out.encode('utf-8'), itype, qk)

def llama_apply_lora_from_file(ctx: llama_context_p, path_lora: str, path_base_model: str, n_threads: c_int) -> c_int:
    return lib.llama_apply_lora_from_file(ctx, path_lora.encode('utf-8'), path_base_model.encode('utf-8'), n_threads)

def llama_get_kv_cache_token_count(ctx: llama_context_p) -> c_int:
    return lib.llama_get_kv_cache_token_count(ctx)

def llama_set_rng_seed(ctx: llama_context_p, seed: c_int):
    return lib.llama_set_rng_seed(ctx, seed)

def llama_get_state_size(ctx: llama_context_p) -> c_size_t:
    return lib.llama_get_state_size(ctx)

def llama_copy_state_data(ctx: llama_context_p, dst: c_ubyte_p) -> c_size_t:
    return lib.llama_copy_state_data(ctx, dst)

def llama_set_state_data(ctx: llama_context_p, src: c_ubyte_p) -> c_size_t:
    return lib.llama_set_state_data(ctx, src)

def llama_load_session_file(ctx: llama_context_p, path_session: str, tokens_out: llama_token_p, n_token_capacity: c_size_t, n_token_count_out: c_size_p) -> c_bool:
    return lib.llama_load_session_file(ctx, path_session.encode('utf-8'), tokens_out, n_token_capacity, n_token_count_out)

def llama_save_session_file(ctx: llama_context_p, path_session: str, tokens: llama_token_p, n_token_count: c_size_t) -> c_bool:
    return lib.llama_save_session_file(ctx, path_session.encode('utf-8'), tokens, n_token_count)

def llama_eval(ctx: llama_context_p, tokens: llama_token_p, n_tokens: c_int, n_past: c_int, n_threads: c_int) -> c_int:
    """Run the llama inference to obtain the logits and probabilities for the next token.
    tokens + n_tokens is the provided batch of new tokens to process
    n_past is the number of tokens to use from previous eval calls
    Returns 0 on success"""
    return lib.llama_eval(ctx, tokens, n_tokens, n_past, n_threads)

def llama_tokenize(ctx: llama_context_p, text: str, tokens: llama_token_p, n_max_tokens: c_int, add_bos: c_bool) -> c_int:
    """Convert the provided text into tokens.
    The tokens pointer must be large enough to hold the resulting tokens.
    Returns the number of tokens on success, no more than n_max_tokens
    Returns a negative number on failure - the number of tokens that would have been returned"""
    return lib.llama_tokenize(ctx, text.encode('utf-8'), tokens, n_max_tokens, add_bos)

def llama_n_vocab(ctx: llama_context_p) -> c_int:
    return lib.llama_n_vocab(ctx)

def llama_n_ctx(ctx: llama_context_p) -> c_int:
    return lib.llama_n_ctx(ctx)

def llama_n_embd(ctx: llama_context_p) -> c_int:
    return lib.llama_n_embd(ctx)

def llama_get_logits(ctx: llama_context_p) -> c_float_p:
    """Token logits obtained from the last call to llama_eval()
    The logits for the last token are stored in the last row
    Can be mutated in order to change the probabilities of the next token
    Rows: n_tokens
    Cols: n_vocab"""
    return lib.llama_get_logits(ctx)

def llama_get_embeddings(ctx: llama_context_p) -> c_float_p:
    """Get the embeddings for the input
    shape: [n_embd] (1-dimensional)"""
    return lib.llama_get_embeddings(ctx)

def llama_token_to_str(ctx: llama_context_p, token: int) -> str:
    """Token Id -> String. Uses the vocabulary in the provided context"""
    return lib.llama_token_to_str(ctx, token).decode('utf-8', errors='ignore')

def llama_token_bos() -> llama_token:
    return lib.llama_token_bos()

def llama_token_eos() -> llama_token:
    return lib.llama_token_eos()

def llama_token_nl() -> llama_token:
    return lib.llama_token_nl()

def llama_sample_repetition_penalty(ctx: llama_context_p, candidates: llama_token_data_array_p, last_tokens: llama_token_p, last_tokens_size: c_size_t, penalty: float):
    lib.llama_sample_repetition_penalty(ctx, candidates, last_tokens, last_tokens_size, penalty)

def llama_sample_frequency_and_presence_penalties(ctx: llama_context_p, candidates: llama_token_data_array_p, last_tokens: llama_token_p, last_tokens_size: c_size_t, alpha_frequency: float, alpha_presence: float):
    lib.llama_sample_frequency_and_presence_penalties(ctx, candidates, last_tokens, last_tokens_size, alpha_frequency, alpha_presence)

def llama_sample_softmax(ctx: llama_context_p, candidates: llama_token_data_array_p):
    lib.llama_sample_softmax(ctx, candidates)

def llama_sample_top_k(ctx: llama_context_p, candidates: llama_token_data_array_p, k: c_int, min_keep: c_size_t):
    lib.llama_sample_top_k(ctx, candidates, k, min_keep)

def llama_sample_top_p(ctx: llama_context_p, candidates: llama_token_data_array_p, p: float, min_keep: c_size_t):
    lib.llama_sample_top_p(ctx, candidates, c_float(p), c_size_t(min_keep))

def llama_sample_tail_free(ctx: llama_context_p, candidates: llama_token_data_array_p, z: float, min_keep: c_size_t):
    lib.llama_sample_tail_free(ctx, candidates, z, min_keep)

def llama_sample_typical(ctx: llama_context_p, candidates: llama_token_data_array_p, p: float, min_keep: c_size_t):
    lib.llama_sample_typical(ctx, candidates, p, min_keep)

def llama_sample_temperature(ctx: llama_context_p, candidates: llama_token_data_array_p, temp: float):
    lib.llama_sample_temperature(ctx, candidates, temp)

def llama_sample_token_mirostat(ctx: llama_context_p, candidates: llama_token_data_array_p, tau: float, eta: float, m: c_int, mu: c_float_p) -> llama_token:
    return lib.llama_sample_token_mirostat(ctx, candidates, tau, eta, m, mu)

def llama_sample_token_mirostat_v2(ctx: llama_context_p, candidates: llama_token_data_array_p, tau: float, eta: float, mu: c_float_p) -> llama_token:
    return lib.llama_sample_token_mirostat_v2(ctx, candidates, tau, eta, mu)

def llama_sample_token_greedy(ctx: llama_context_p, candidates: llama_token_data_array_p) -> llama_token:
    return lib.llama_sample_token_greedy(ctx, candidates)

def llama_sample_token(ctx: llama_context_p, candidates: llama_token_data_array_p) -> llama_token:
    return lib.llama_sample_token(ctx, candidates)

def llama_print_timings(ctx: llama_context_p):
    lib.llama_print_timings(ctx)

def llama_reset_timings(ctx: llama_context_p):
    lib.llama_reset_timings(ctx)

def llama_print_system_info() -> c_char_p:
    return lib.llama_print_system_info()