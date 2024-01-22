# MobileVLM

Currently this implementation supports [MobileVLM-v1.7](https://huggingface.co/mtgv/MobileVLM-1.7B) variants.

for more information, please go to [Meituan-AutoML/MobileVLM](https://github.com/Meituan-AutoML/MobileVLM)

The implementation is based on llava, and is compatible with llava and mobileVLM. The usage is basically same as llava.

## Usage
Build with cmake or run `make llava-cli` to build it.

After building, run: `./llava-cli` to see the usage. For example:

```sh
./llava-cli -m MobileVLM-1.7B/ggml-model-q4_k.gguf \
    --mmproj MobileVLM-1.7B/mmproj-model-f16.gguf \
    --image path/to/an/image.jpg \
    -p "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nWho is the author of this book? Answer the question using a single word or phrase. ASSISTANT:"
```

## Model conversion

- Clone `mobileVLM-1.7B` and `clip-vit-large-patch14-336` locally:

```sh
git clone https://huggingface.co/mtgv/MobileVLM-1.7B

git clone https://huggingface.co/openai/clip-vit-large-patch14-336
```

2. Use `llava-surgery.py` to split the LLaVA model to LLaMA and multimodel projector constituents:

```sh
python ./examples/llava/llava-surgery.py -m path/to/MobileVLM-1.7B
```

3. Use `convert-image-encoder-to-gguf.py` with `--projector-type ldp` to convert the LLaVA image encoder to GGUF:

```sh
python ./examples/llava/convert-image-encoder-to-gguf \
    -m path/to/clip-vit-large-patch14-336 \
    --llava-projector path/to/MobileVLM-1.7B/llava.projector \
    --output-dir path/to/MobileVLM-1.7B \
    --projector-type ldp
```

4. Use `convert.py` to convert the LLaMA part of LLaVA to GGUF:

```sh
python ./convert.py path/to/MobileVLM-1.7B
```

5. Use `quantize` to convert LLaMA part's DataType from `fp16` to `q4_k`
```sh
./quantize path/to/MobileVLM-1.7B/ggml-model-f16.gguf path/to/MobileVLM-1.7B/ggml-model-q4_k.gguf q4_k_s
```

Now both the LLaMA part and the image encoder is in the `MobileVLM-1.7B` directory.

## Android compile and run
### compile
refer to `examples/llava/android/build_64.sh`
```sh
mkdir examples/llava/android/build_64
cd examples/llava/android/build_64
../build_64.sh
```
### run on Android
refer to `android/adb_run.sh`, modify resources' `name` and `path`

## some result on Android with `Snapdragon 888` chip
### case 1
**input**
```sh
/data/local/tmp/llava-cli \
    -m /data/local/tmp/ggml-model-q4_k.gguf \
    --mmproj /data/local/tmp/mmproj-model-f16.gguf \
    -t 4 \
    --image /data/local/tmp/demo.jpg \
    -p "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nWho is the author of this book? \nAnswer the question using a single word or phrase. ASSISTANT:"
```
**output**
```sh
encode_image_with_clip: image encoded in 21148.71 ms by CLIP (  146.87 ms per image patch)
 Susan Wise Bauer
llama_print_timings:        load time =   23574.72 ms
llama_print_timings:      sample time =       1.24 ms /     6 runs   (    0.21 ms per token,  4850.44 tokens per second)
llama_print_timings: prompt eval time =   12460.15 ms /   246 tokens (   50.65 ms per token,    19.74 tokens per second)
llama_print_timings:        eval time =     424.86 ms /     6 runs   (   70.81 ms per token,    14.12 tokens per second)
llama_print_timings:       total time =   34731.93 ms
```
### case 2
**input**
```sh
/data/local/tmp/llava-cli \
    -m /data/local/tmp/ggml-model-q4_k.gguf \
    --mmproj /data/local/tmp/mmproj-model-f16.gguf \
    -t 4 \
    --image /data/local/tmp/cat.jpeg \
    -p "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nWhat is in the image? ASSISTANT:"
```

**output**
```sh
encode_image_with_clip: image encoded in 21149.51 ms by CLIP (  146.87 ms per image patch)
 The image depicts a cat sitting in the grass near some tall green plants.
llama_print_timings:        load time =   23257.32 ms
llama_print_timings:      sample time =       5.25 ms /    18 runs   (    0.29 ms per token,  3430.53 tokens per second)
llama_print_timings: prompt eval time =   11900.73 ms /   232 tokens (   51.30 ms per token,    19.49 tokens per second)
llama_print_timings:        eval time =    1279.03 ms /    18 runs   (   71.06 ms per token,    14.07 tokens per second)
llama_print_timings:       total time =   34570.79 ms
```

## Minor shortcomings
The `n_patch` of output in `ldp` is 1/4 of the input. In order to implement quickly, we uniformly modified `clip_n_patches` function to a quarter. when counting the time consumption, the calculated time will be 4 times bigger than the real cost.

## TODO

- [ ] Support non-CPU backend for the new operators, such as `depthwise`, `hardswish`, `hardsigmoid`
- [ ] Optimize LDP projector performance

      - Optimize the structure definition to avoid unnecessary memory rearrangements, to reduce the use of `ggml_permute_cpy`;
      - Optimize operator implementation (ARM CPU/NVIDIA GPU): such as depthwise conv, hardswish, hardsigmoid, etc.
- [ ] run MobileVLM on `Jetson Orin`
- [ ] Support more model variants, such as `MobileVLM-3B`.


## contributor
```sh
zhangjidong05, yangyang260, huyiming03, chenxiaotao03
```
