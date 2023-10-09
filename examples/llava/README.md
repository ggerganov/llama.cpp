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

## Model conversion

- Clone `llava-v15-7b`` and `clip-vit-large-patch14-336`` locally:

```sh
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b

git clone https://huggingface.co/openai/clip-vit-large-patch14-336
```

2. Use `llava_surgery.py` to split the LLaVA model to LLaMA and multimodel projector constituents:

```sh
python ./examples/llava/llava_surgery.py -m ../llava-v1.5-7b
```

3. Use `convert_image_encoder_to_gguf.py` to convert the LLaVA image encoder to GGUF:

```sh
python ./examples/llava/convert_image_encoder_to_gguf -m ../clip-vit-large-patch14-336 --llava-projector ../llava-v1.5-7b/llava.projector --output-dir ../llava-v1.5-7b
```

4. Use `convert.py` to convert the LLaMA part of LLaVA to GGUF:

```sh
python ./convert.py ../llava-v1.5-7b
```

Now both the LLaMA part and the image encoder is in the `llava-v1.5-7b` directory.

## TODO

These will be include in this pr:

- [ ] Better command line interface.

These will be another PR:

- [ ] Support server mode.
- [ ] Support non-CPU backend for the image encoding part.
- [ ] Support different sampling methods.
- [ ] Support more model variants.
