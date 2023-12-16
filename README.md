# PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU
---

## Demo 🔥

https://github.com/SJTU-IPADS/PowerInfer/assets/34213478/d26ae05b-d0cf-40b6-8788-bda3fe447e28

PowerInfer v.s. llama.cpp on a single RTX 4090(24G) running Falcon(ReLU)-40B-FP16 with a 11x speedup!

<sub>Both PowerInfer and llama.cpp were running on the same hardware and fully utilized VRAM on RTX 4090.</sub>

---
## Abstract

We introduce PowerInfer, a high-speed Large Language Model (LLM) inference engine on a personal computer (PC) 
equipped with a single consumer-grade GPU. The key underlying the design of PowerInfer is exploiting the high locality 
inherent in LLM inference, characterized by a power-law distribution in neuron activation. 
This distribution indicates that a small subset of neurons, termed hot neurons, are consistently activated 
across inputs, while the majority, cold neurons, vary based on specific inputs.
PowerInfer exploits such an insight to design a GPU-CPU hybrid inference engine:
hot-activated neurons are preloaded onto the GPU for fast access, while cold-activated neurons are computed 
on the CPU, thus significantly reducing GPU memory demands and CPU-GPU data transfers.
PowerInfer further integrates adaptive predictors and neuron-aware sparse operators,
optimizing the efficiency of neuron activation and computational sparsity.
Evaluation shows that PowerInfer attains an average token generation rate of 13.20 tokens/s, with a peak of 29.08 tokens/s, across various LLMs (including OPT-175B) on a single NVIDIA RTX 4090 GPU,
only 18\% lower than that achieved by a top-tier server-grade A100 GPU.
This significantly outperforms llama.cpp by up to 11.69x while retaining model accuracy.

## Feature
PowerInfer is a high-speed and easy-to-use inference engine for deploying LLM locally. Interestingly, we observe that in ReLU LLM, every neuron is an expert! And a small subset of neurons consistently contributes to the output.
PowerInfer is fast with:

- Exploiting the high locality in LLM inference
- Neuron-aware hybrid CPU/GPU sparse operator
- Neuron granularity offloading

PowerInfer is flexible and easy to use with:

- Integration with popular [ReLU-sparse models](https://huggingface.co/SparseLLM)
- Low-latency serving locally with one single consumer-grade GPU 

PowerInfer supports the following models:

- Falcon-40B model
- Llama family models

Now PowerInfer supports the following architectures:

- Intel CPU with AVX2 instructions
- Nvidia GPU
  



## Getting Started

- [Installation](##setup--installation)
- [Model Weights](##model-weights)

## Setup & Installation
### Get the Code

```bash
git clone https://github.com/SJTU-IPADS/PowerInfer
cd PowerInfer
```
### Build
In order to build PowerInfer you have two different options. These commands are supposed to be run from the root directory of the project.

Using `make` on Linux or MacOS:
```bash
make
```

Using `CMake`:
* If you have one GPU:
```bash
cmake -S . -B build -DLLAMA_CUBLAS=ON
cmake --build build --config Release
```
* If you just CPU:
```bash
cmake -S . -B build
cmake --build build --config Release
```

## Model Weights
As for now, we have not released the predictor training code, we suggest you download the sparse model from huggingface in the following link.
| Base Model | GGUF Format Link | Original Model |
|------------|------------------|----------------|
| LLaMA(ReLU)-2-7B   | [PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF](https://huggingface.co/PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF)    | [SparseLLM/ReluLLaMA-7B](https://huggingface.co/SparseLLM/ReluLLaMA-7B)     |
| LLaMA(ReLU)-2-13B    | [PowerInfer/ReluLLaMA-13B-PowerInfer-GGUF](https://huggingface.co/PowerInfer/ReluLLaMA-13B-PowerInfer-GGUF)   | [SparseLLM/ReluLLaMA-13B](https://huggingface.co/SparseLLM/ReluLLaMA-13B)  |
| Falcon(ReLU)-40B    | [PowerInfer/ReluFalcon-40B-PowerInfer-GGUF](https://huggingface.co/PowerInfer/ReluFalcon-40B-PowerInfer-GGUF)    | [SparseLLM/ReluFalcon-40B](https://huggingface.co/SparseLLM/ReluFalcon-40B)      |
| LLaMA(ReLU)-2-70B    | [PowerInfer/ReluLLaMA-70B-PowerInfer-GGUF](https://huggingface.co/PowerInfer/ReluLLaMA-70B-PowerInfer-GGUF)    | [SparseLLM/ReluLLaMA-70B](https://huggingface.co/SparseLLM/ReluLLaMA-70B)      |

## Inference
- If you just have CPU:
```bash
  ./build/bin/main -m /PATH/TO/MODEL -n $(output_token_count) -t $(thread_num) -p $(prompt)
```
- If you have CPU with one GPU:
```bash
./build/bin/main -m /PATH/TO/MODEL -n $(output_token_count) -t $(thread_num) -p $(prompt) --vram-budget $(GPU_VRAM_OFFLOADING)
```

As for now, it requires an offline-generated "GPU index" file to split FFNs on GPU. If you want to try it, please use the following instructions to generate the GPU index file:
```bash
python scripts/export-gpu-split.py $(activation_count_path) $(output_idx_path) solver
```
Then, you can use the following instructions to run PowerInfer with GPU index:
```bash
./build/bin/main -m /PATH/TO/MODEL -n $(output_token_count) -t $(thread_num) -p $(prompt) --gpu-index $(split_path)
```

## Evaluation

![github-eval-4090](https://github.com/SJTU-IPADS/PowerInfer/assets/34213478/d700fa6c-77ba-462f-a2fc-3fd21c898f33)

![github-eval-2080ti-q4](https://github.com/SJTU-IPADS/PowerInfer/assets/34213478/0fc1bfc4-aafc-4e82-a865-bec0143aff1a)

PowerInfer achieves up to 11.69x and 8.00x speedup for FP16 and INT4 models!

## TODOs
We will release the code and data in the following order, please stay tuned!

- [x] Release core code of PowerInfer, supporting Llama-2, Falcon-40B.
- [ ] Release perplexity evaluation code
- [ ] Support Metal for Mac
- [ ] Release code for OPT models
- [ ] Release predictor training code 
- [ ] Support online split for FFN network
- [ ] Support Multi-GPU


## Citation

If you find PowerInfer useful or relevant to your project and research, please kindly cite our paper:

```bibtex
Stay tuned!
```

## Acknowledgement
We are thankful for the easily modifiable operator library [ggml](https://github.com/ggerganov/ggml) and execution runtime provided by [llama.cpp](https://github.com/ggerganov/llama.cpp). We also extend our gratitude to [THUNLP](https://nlp.csai.tsinghua.edu.cn/) for their support of ReLU-based sparse models. We also appreciate the research of [DejaVu](https://proceedings.mlr.press/v202/liu23am.html), which inspires PowerInfer.
