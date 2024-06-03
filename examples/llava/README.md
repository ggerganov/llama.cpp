# LLaVA

Currently this implementation supports [llava-v1.5](https://huggingface.co/liuhaotian/llava-v1.5-7b) variants,
as well as llava-1.6 [llava-v1.6](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2) variants.

The pre-converted [7b](https://huggingface.co/mys/ggml_llava-v1.5-7b)
and [13b](https://huggingface.co/mys/ggml_llava-v1.5-13b)
models are available.
For llava-1.6 a variety of prepared gguf models are available as well [7b-34b](https://huggingface.co/cmp-nct/llava-1.6-gguf)

After API is confirmed, more models will be supported / uploaded.

## Usage
Build with cmake or run `make llava-cli` to build it.

After building, run: `./llava-cli` to see the usage. For example:

```sh
./llava-cli -m ../llava-v1.5-7b/ggml-model-f16.gguf --mmproj ../llava-v1.5-7b/mmproj-model-f16.gguf --image path/to/an/image.jpg
```

**note**: A lower temperature like 0.1 is recommended for better quality. add `--temp 0.1` to the command to do so.
**note**: For GPU offloading ensure to use the `-ngl` flag just like usual

## LLaVA 1.5

1. Clone a LLaVA and a CLIP model ([available options](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)). For example:

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

5. Use `examples/convert-legacy-llama.py` to convert the LLaMA part of LLaVA to GGUF:

```sh
python ./examples/convert-legacy-llama.py ../llava-v1.5-7b --skip-unknown
```

Now both the LLaMA part and the image encoder are in the `llava-v1.5-7b` directory.

## LLaVA 1.6 gguf conversion
1) First clone a LLaVA 1.6 model:
```console
git clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b
```

2) Install the required Python packages:

```sh
pip install -r examples/llava/requirements.txt
```

3) Use `llava-surgery-v2.py` which also supports llava-1.5 variants pytorch as well as safetensor models:
```console
python examples/llava/llava-surgery-v2.py -C -m ../llava-v1.6-vicuna-7b/
```
- you will find a llava.projector and a llava.clip file in your model directory

4) Copy the llava.clip file into a subdirectory (like vit), rename it to pytorch_model.bin and add a fitting vit configuration to the directory:
```console
mkdir vit
cp ../llava-v1.6-vicuna-7b/llava.clip vit/pytorch_model.bin
cp ../llava-v1.6-vicuna-7b/llava.projector vit/
curl -s -q https://huggingface.co/cmp-nct/llava-1.6-gguf/raw/main/config_vit.json -o vit/config.json
```

5) Create the visual gguf model:
```console
python ./examples/llava/convert-image-encoder-to-gguf.py -m vit --llava-projector vit/llava.projector --output-dir vit --clip-model-is-vision
```
- This is similar to llava-1.5, the difference is that we tell the encoder that we are working with the pure vision model part of CLIP

6) Then convert the model to gguf format:
```console
python ./examples/convert-legacy-llama.py ../llava-v1.6-vicuna-7b/ --skip-unknown
```

7) And finally we can run the llava-cli using the 1.6 model version:
```console
./llava-cli -m ../llava-v1.6-vicuna-7b/ggml-model-f16.gguf --mmproj vit/mmproj-model-f16.gguf --image some-image.jpg -c 4096
```

**note** llava-1.6 needs more context than llava-1.5, at least 3000 is needed (just run it at -c 4096)
**note** llava-1.6 greatly benefits from batched prompt processing (defaults work)

## Phi-3-Vision-128K-Instruct gguf conversion
1) Set a working directory for PHI3V and PHI3 instruct. Clone both into this dir. (It's easiest to cd into your local hf cache and copy the models from there to here)

```console
mkdir phi3-fun
cd phi3-fun

mkdir phi3-base
git clone https://huggingface.co/microsoft/Phi-3-mini-128k-instruct

mkdir phi3-vision
git clone https://huggingface.co/microsoft/Phi-3-vision-128k-instruct

```

2) Use `llava-surgery-v2.py` to extract clip from PHI3V:
```console
python examples/llava/llava-surgery-v2.py -C -m phi3-fun/phi3-vision/
```
- you will find a llava.projector and a llava.clip file in your model directory

4) Copy the llava.clip file into a subdirectory (like vit), rename it to pytorch_model.bin and add a fitting vit configuration to the directory:
```console
// under phi3-fun/phi-vision dir
mkdir vit 
cp llava.clip vit/pytorch_model.bin
cp llava.projector vit/
curl -s -q https://huggingface.co/cmp-nct/llava-1.6-gguf/raw/main/config_vit.json -o vit/config.json
```

5) Create the visual gguf model:
```console
python examples/llava/convert-image-encoder-to-gguf.py -m phi3-fun/phi3-vision/vit --llava-projector phi3-fun/phi3-vision/vit/llava.projector --output-dir phi3-fun/phi3-vision/vit --clip-model-is-vision
```

6) Extract the language-modelling  (everything except CLIP) part of PHI3V and assign the weights to a normal PHI3 model

```console
python examples/llava/phi3-weight-transfer.py --phi3-instruct-base-path phi3-fun/phi3-base --phi3v-base-path phi3-fun/phi3-vision
```

7) Convert this to a normal gguf
(First delete the old safetensors from this directory)
```console
python convert-hf-to-gguf.py phi3-fun/phi3-base
```

8) Invoke
(recompile llama.cpp first)
```console
./llava-cli -m phi3-fun/phi3-base/ggml-model-f16.gguf --mmproj phi3-fun/phi3-vision/vit/mmproj-model-f16.gguf --image IMAGE -c 4096  --temp .1 -p "PROMPT"
```

## llava-cli templating and llava-1.6 prompting

llava-1.5 models all use the same vicuna prompt, here you can just add your image question like `-p "Provide a full description."`
For llava-1.5 models which are not vicuna (mistral and Yi) you need to adapt system prompt as well as user prompt, for this purpose llava-cli has a basic templating system:

**For Mistral and using llava-cli binary:**
Add this: `-p "<image>\nUSER:\nProvide a full description.\nASSISTANT:\n"`
The mistral template for llava-1.6 seems to be no system print and a USER/ASSISTANT role

**For the 34B this should work:**
Add this: `-e -p <|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nProvide a full description.<|im_end|><|im_start|>assistant\n`


## How to know if you are running in llava-1.5 or llava-1.6 mode

When running llava-cli you will see a visual information right before the prompt is being processed:

**Llava-1.5:**
`encode_image_with_clip: image embedding created: 576 tokens`

**Llava-1.6 (anything above 576):**
`encode_image_with_clip: image embedding created: 2880 tokens`


Alternatively just pay notice to how many "tokens" have been used for your prompt, it will also show 1000+ tokens for llava-1.6




## TODO

- [x] Support non-CPU backend for the image encoding part.
- [ ] Support different sampling methods.
- [ ] Support more model variants.

