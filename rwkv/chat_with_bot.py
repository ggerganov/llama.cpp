# Provides terminal-based chat interface for RWKV model.

import os
import sys
import argparse
import pathlib
import sampling
import tokenizers
import rwkv_cpp_model
import rwkv_cpp_shared_library

# ======================================== Script settings ========================================

# Copied from https://github.com/ggerganov/llama.cpp/blob/6e7801d08d81c931a5427bae46f00763e993f54a/prompts/chat-with-bob.txt
prompt: str = """Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User: Hello, Bob.
Bob: Hello. How may I help you today?
User: Please tell me the largest city in Europe.
Bob: Sure. The largest city in Europe is Moscow, the capital of Russia."""

# No trailing space here!
bot_message_prefix: str = 'Bob:'
user_message_prefix: str = 'User:'

max_tokens_per_generation: int = 100

# Sampling settings.
temperature: float = 0.8
top_p: float = 0.5

# =================================================================================================

parser = argparse.ArgumentParser(description='Provide terminal-based chat interface for RWKV model')
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
print(f'Processing {prompt_token_count} prompt tokens, may take a while')

logits, state = None, None

for token in prompt_tokens:
    logits, state = model.eval(token, state, state, logits)

print('\nChat initialized! Write something and press Enter.')

while True:
    # Read user input
    print('> ', end='')
    user_input = sys.stdin.readline()

    # Process the input
    new_tokens = tokenizer.encode('\n' + user_message_prefix + ' ' + user_input + '\n' + bot_message_prefix).ids

    for token in new_tokens:
        logits, state = model.eval(token, state, state, logits)

    # Generate and print bot response
    print(bot_message_prefix, end='')

    decoded = ''

    for i in range(max_tokens_per_generation):
        token = sampling.sample_logits(logits, temperature, top_p)

        decoded = tokenizer.decode([token])

        print(decoded, end='', flush=True)

        if '\n' in decoded:
            break

        logits, state = model.eval(token, state, state, logits)

    if '\n' not in decoded:
        print()
