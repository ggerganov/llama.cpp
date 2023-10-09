# LLaVA

Currently this implementation supports [llava-v1.5](https://huggingface.co/liuhaotian/llava-v1.5-7b) variants.

The pre-converted 7b model can be found [here](https://huggingface.co/mys/ggml_llava-v1.5-7b).

After API is confirmed, more models will be supported / uploaded.
## Usage
The `llava` target is cmake-only for now (TODO: add to `make`) and built as a part of examples.

After building, run: `./bin/llava` to see the usage. For example:

```sh
./bin/llava path/to/llava-v1.5-7b/ggml-model-q5_k.gguf path/to/llava-v1.5-7b/mmproj-model-f16.gguf path/to/an/image.jpg
```

## TODO

These will be include in this pr:

- [ ] Better command line interface.
- [ ] Document model conversion.

These will be another PR:

- [ ] Support server mode.
- [ ] Support non-CPU backend for the image encoding part.
- [ ] Support different sampling methods.
- [ ] Support more model variants.
