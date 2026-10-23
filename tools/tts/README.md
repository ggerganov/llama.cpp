# llama.cpp TTS

This is a tool to demonstrate audio generation capability in llama.cpp via `libmtmd`. It was added via PR [#26254](https://github.com/ggml-org/llama.cpp/pull/26254)

Note: this tool used to serve as a demo for OuteTTS, but it was converted to a more model-agnostic tool.

## Common usage

Simple usage:

```sh
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF -p "Hello world" --output out.wav
```

Note: it's highly recommended to always provide a speaker reference voice (via `--tts-speaker-file`); otherwise, the model's performance may be degraded.

Common params:
- Sampling params such as `--top-k`, `--top-p`, `--temp`, etc.
- `-n <number_of_frames>` limits the output length, e.g. `-n 500`. Note that how many milliseconds each frame represents varies by model
- Core inference params such as `-ngl`, `-b`, `-ub`, etc.

## Qwen3-TTS

Available params:
- `--tts-lang` can be `zh`, `en`, `de`, `it`, `pt`, `es`, `ja`, `ko`, `fr`, `ru` (default: `en`)
- `--tts-speaker-file` should point to a speaker reference audio file (wav, mp3)

Example usage:

```sh
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF \
    -p "Hello world" \
    --tts-lang english \
    --tts-speaker-file speaker.mp3 \
    --output out.wav
```

## Pocket TTS

Available params:
- `--tts-speaker-file` should point to a speaker reference audio file (wav, mp3). It is required, the model produces almost no audio without it
- Note: `lang` is not used, the language is a property of the weights

Example usage:

```sh
llama-tts -m pocket-tts.gguf \
    -mm mmproj-pocket-tts.gguf \
    -p "Hello world" \
    --tts-speaker-file speaker.mp3 \
    --output out.wav
```

**Note for GGUF conversion:**

The [upstream repository](https://huggingface.co/kyutai/pocket-tts) holds one complete model per language under `languages/`, next to a set of shared files at the root. Convert one of the `languages/<name>` directories, **not** the root directory:

```sh
python convert_hf_to_gguf.py path/to/pocket-tts/languages/english --outfile pocket-tts.gguf
python convert_hf_to_gguf.py path/to/pocket-tts/languages/english --mmproj --outfile mmproj-pocket-tts.gguf
```

## Chatterbox

Two variants are supported: [Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) (English) and [Chatterbox Multilingual V3](https://huggingface.co/ResembleAI/chatterbox) (23 languages).

Available params:
- `--tts-speaker-file` should point to a speaker reference audio file (wav, mp3). Optional, the default voice of the model is used without it
- `--tts-lang` sets the language of the multilingual variant, for example `fr` (default: none). It is not used by Turbo

Example usage:

```sh
llama-tts -m chatterbox-mtl.gguf \
    -mm mmproj-chatterbox-mtl.gguf \
    -p "Bonjour tout le monde" \
    --tts-lang fr \
    --tts-speaker-file speaker.mp3 \
    --output out.wav
```

Note: the multilingual variant runs classifier-free guidance with a second sequence. `llama-tts` sets this up by itself, but `llama-server` does not support it yet.

**Note for GGUF conversion:**

The upstream repositories ship raw checkpoints without `config.json`. Download the files below into a directory, then convert it as usual (the variant is detected from the talker file):
- Turbo: `t3_turbo_v1.safetensors`, `s3gen_meanflow.safetensors`, `ve.safetensors`, `conds.pt` and the GPT-2 tokenizer files (`vocab.json`, `merges.txt`, `added_tokens.json`, `special_tokens_map.json`, `tokenizer_config.json`)
- Multilingual V3: `t3_mtl23ls_v3.safetensors`, `s3gen_v3.safetensors`, `ve.safetensors`, `conds.pt`, `mtl_tokenizer.json`

```sh
python convert_hf_to_gguf.py path/to/chatterbox --outfile chatterbox.gguf
python convert_hf_to_gguf.py path/to/chatterbox --mmproj --outfile mmproj-chatterbox.gguf
```
