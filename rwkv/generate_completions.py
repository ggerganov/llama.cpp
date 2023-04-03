# Generates completions from RWKV model based on a prompt.

import argparse
import os
import pathlib
import time
import sampling
import tokenizers
import rwkv_cpp_model
import rwkv_cpp_shared_library


# ======================================== Script settings ========================================

prompt: str = """# rwkv.cpp

This is a port of [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) to [ggerganov/ggml](https://github.com/ggerganov/ggml).

Besides usual **FP32**, it supports **FP16** and **quantized INT4** inference on CPU. This project is **CPU only**."""

# How many completions to generate.
generation_count: int = 3
# Token count per single completion.
tokens_per_generation: int = 100

# Sampling settings.
temperature: float = 0.8
top_p: float = 0.5

# =================================================================================================

parser = argparse.ArgumentParser(description='Generate completions from RWKV model based on a prompt')
parser.add_argument('model_path', help='Path to RWKV model in ggml format')
args = parser.parse_args()

assert prompt != '', 'Prompt must not be empty'

print('Loading 20B tokenizer')
tokenizer_path = pathlib.Path(os.path.abspath(__file__)).parent / '20B_tokenizer.json'
tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))

library = rwkv_cpp_shared_library.load_rwkv_shared_library()
print(f'System info: {library.rwkv_get_system_info_string()}')

print('Loading RWKV model')
model = rwkv_cpp_model.RWKVModel(library, args.model_path)

prompt_tokens = tokenizer.encode(prompt).ids
prompt_token_count = len(prompt_tokens)
print(f'{prompt_token_count} tokens in prompt')

init_logits, init_state = None, None

for token in prompt_tokens:
    init_logits, init_state = model.eval(token, init_state, init_state, init_logits)

for GENERATION in range(generation_count):
    print(f'\n--- Generation {GENERATION} ---\n')
    print(prompt, end='[')
    start = time.time()

    logits, state = init_logits.clone(), init_state.clone()

    for i in range(tokens_per_generation):
        token = sampling.sample_logits(logits, temperature, top_p)

        print(tokenizer.decode([token]), end='')

        logits, state = model.eval(token, state, state, logits)

    delay = time.time() - start
    print(']\n\nTook %.3f sec, %d ms per token' % (delay, delay / tokens_per_generation * 1000))
