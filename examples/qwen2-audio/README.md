## Run Qwee2-audio

From the root directory of the repo, run commands below:

```shell
./build/bin/nexa-qwen2-cli \
    --model /home/azureuser/zack/ggml-project-apollo/llama.cpp.origin/examples/nano-omni-audio/gemma2-2b.gguf \
    --mmproj /home/azureuser/zack/ggml-project-apollo/llama.cpp.origin/examples/nano-omni-audio/nano-omni-instruct.mel-filters-audio_tower-multi_modal_projector.gguf \
    --file /home/azureuser/zack/ggml-project-apollo/examples/whisper/samples/jfk.wav \
    --prompt "this conversation talks about" \
    --n-gpu-layers 27  # offload all 27 layers of gemma2 model to GPU
```
