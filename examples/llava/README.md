# LLaVA

Currently this implementation supports [llava-v1.5](https://huggingface.co/liuhaotian/llava-v1.5-7b) variants.

The pre-converted [7b](https://huggingface.co/mys/ggml_llava-v1.5-7b)
and [13b](https://huggingface.co/mys/ggml_llava-v1.5-13b)
models are available.

After API is confirmed, more models will be supported / uploaded.

## Usage
Build with cmake or run `make llava-cli` to build it.

After building, run: `./llava-cli` to see the usage. For example:

```sh
./llava-cli -m ../llava-v1.5-7b/ggml-model-f16.gguf --mmproj ../llava-v1.5-7b/mmproj-model-f16.gguf --image path/to/an/image.jpg
```

**note**: A lower temperature like 0.1 is recommended for better quality. add `--temp 0.1` to the command to do so.

## Model conversion

- Clone `llava-v15-7b` and `clip-vit-large-patch14-336` locally:

```sh
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b

git clone https://huggingface.co/openai/clip-vit-large-patch14-336
```

2. Install the required Python packages:

```sh
pip install -r examples/llava/requirements.txt
```

3. Use `llava-surgery.py` to split the LLaVA model to LLaMA and multimodel projector constituents:

```sh
python ./examples/llava/llava-surgery.py -m ../llava-v1.5-7b
```

4. Use `convert-image-encoder-to-gguf.py` to convert the LLaVA image encoder to GGUF:

```sh
python ./examples/llava/convert-image-encoder-to-gguf.py -m ../clip-vit-large-patch14-336 --llava-projector ../llava-v1.5-7b/llava.projector --output-dir ../llava-v1.5-7b
```

5. Use `convert.py` to convert the LLaMA part of LLaVA to GGUF:

```sh
python ./convert.py ../llava-v1.5-7b
```

Now both the LLaMA part and the image encoder is in the `llava-v1.5-7b` directory.

## TODO

- [ ] Support non-CPU backend for the image encoding part.
- [ ] Support different sampling methods.
- [ ] Support more model variants.
