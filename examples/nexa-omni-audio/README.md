# nexa-audio-omni

## Build this example

Build the whole project from repo root dir

```shell
cmake -B build-cpu
cmake --build build-cpu --config Release -j 24
```


```shell
cmake -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda --config Release -j 24
```

## Prepare GGUF files

### mmproj (mel-filter + whisper-encoder + audio-projector)

1. Extract whisper-medium encoder (`audio_tower`) from `nexa-collaboration/nano-omini-instruct`

    ```shell
    python convert-to-gguf-f16.py
    ```

2. Extract mel filters from `ggml-medium.bin` and add thems to previously extracted `nano-omni-audio-encoder-f16.gguf`

    ```shell
    python add-mel-filters.py
    ```

    > Run `bash download-ggml-model.sh` to donwload `ggml-medium.bin` and move it into the `models` folder
    > Don't forget to modify the input and output file paths in the Python script above before running it.

### gemma2

```shell
python prepare-gemma2-hf.py
python ../../convert_hf_to_gguf.py ./models/nano-omini-instruct-gemma2 --outfile ./models/nano-omini-instruct.gemma2.gguf [--outtype bf16]
```

## Run nexa-audio-omni

From the root directory of the repo, run commands below:

```shell
./build-cpu/bin/nexa-omni-cli \
    --model examples/nexa-omni-audio/models/nano-omni-instruct.gemma2.gguf \
    --mmproj examples/nexa-omni-audio/models/nano-omni-instruct.mel-filters-audio_tower-multi_modal_projector.gguf \
    --file examples/nexa-omni-audio/samples/jfk.wav \
    --prompt "this conversation talks about"
```

```shell
./build/bin/nexa-qwen2-cli \
    --model /home/azureuser/zack/ggml-project-apollo/llama.cpp.origin/examples/qwen2-audio/qwen2/Qwen2-7.8B-F16.gguf \
    --mmproj /home/azureuser/zack/ggml-project-apollo/llama.cpp/examples/audio-lm-python/qwen2-audio-instruct.mel-filters-audio_tower-multi_modal_projector.gguf \
    --file /home/azureuser/zack/ggml-project-apollo/examples/whisper/samples/jfk.wav \
    --prompt "this conversation talks about" \
    --n-gpu-layers 27  # offload all 27 layers of gemma2 model to GPU
```
