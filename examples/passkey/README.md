# jarvis.cpp/example/passkey

A passkey retrieval task is an evaluation method used to measure a language
models ability to recall information from long contexts.

See the following PRs for more info:

- https://github.com/ggerganov/jarvis.cpp/pull/3856
- https://github.com/ggerganov/jarvis.cpp/pull/4810

### Usage

```bash
make -j && ./jarvis-passkey -m ./models/jarvis-7b-v2/ggml-model-f16.gguf --junk 250
```
