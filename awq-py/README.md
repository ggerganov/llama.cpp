# AWQ: Activation-aware Weight Quantization for LLM - version apply to llamacpp
[[Paper](https://arxiv.org/abs/2306.00978)][[Original Repo](https://github.com/mit-han-lab/llm-awq)][[Easy-to-use Repo](https://github.com/casper-hansen/AutoAWQ)]

**Supported models:**

- [X] LLaMA
- [x] LLaMA 2
- [X] MPT
- [X] Mistral AI v0.1
- [ ] Bloom
- [ ] Mixtral MoE

**TODO:**
- [x] Update version work with both MPT and MPT-AWQ model
- [ ] Add OPT model
- [ ] Add Bloom model
- [ ] Add Mixtral MoE
- [ ] Support w3, w2


## Contents

- [Install](##Install)
- [Convert](##Convert)
- [Quantize](##Quantize)
- [Test](##Test)
- [Benchmark](##Benchmark)
- [Results](##Results)

## Install
Install requirements
```bash
pip install -r requirements.txt
```
Get the pre-computed AWQ search results for multiple model families, including LLaMA, LLaMA2, MPT, OPT
```bash
git clone https://huggingface.co/datasets/mit-han-lab/awq-model-zoo awq_cache
```

## Convert
Example for llama model
```bash
# For llama7b and llama2 models
python convert.py models/llama-7b/ --awq-path awq_cache/llama-7b-w4-g128.pt --outfile models/llama_7b_fp16.gguf
# For mistral and mpt models
python convert-hf-to-gguf.py models/mpt-7b/ --awq-path awq_cache/mpt-7b-w4-g128.pt --outfile models/mpt_7b_fp16.gguf
```

## Quantize
```bash
# We only benchmark and confirm the results on q4_0, q4_1, and q2_k types.
./quantize models/llama_7b_fp16.gguf models/llama_7b_q4_0.gguf q4_0
```

## Test
```bash
# For all models.
./build/bin/main -m models/llama_7b_q4_0.gguf -n 128 --prompt "Once upon a time"
```

## Benchmark
The perplexity measurements in table above are done against the `wikitext2` test dataset (https://paperswithcode.com/dataset/wikitext-2), with context length of 512.
```bash
# For llama and llama2, and mistral models.
./perplexity -m models/llama_7b_q4_0.gguf -f datasets/wikitext-2-raw/wiki.test.raw
```

## Results
Results are run on OpenBLAS (CPU) and CuBLAS (GPU) for fair comparison
We use three types of llamacpp quantization methods to work with our version, including q4_0, q4_1, and q2_k

### Llama 7B (Build with OpenBLAS)

| Model      | Measure      | F16    | Q4_0   | Q4_1   | Q2_K   |
|-----------:|--------------|-------:|-------:|-------:|-------:|
|Llama 7B    | perplexity   | 5.9066 | 6.1214 | 6.0643 | 6.5808 |
|Llama 7B    | file size    |  12.9G  |   3.5G |   3.9G |   2.7G |
|Llama 7B    | bits/weight  |   16.0 |    4.5 |    5.0 |    2.6 |
|AWQ-LLama 7B| perplexity   | 5.9175 | 6.0252 | 5.9987 | 6.3692 |
|AWQ-LLama 7B| file size    |  12.9G  |   3.5G |   3.9G |   2.7G |
|AWQ-LLama 7B| bits/weight  |   16.0 |    4.5 |    5.0 |    2.6 |


### Llama2 7B (Build with CuBLAS)

| Model       | Measure      | F16    | Q4_0   | Q4_1   | Q2_K   |
|------------:|--------------|-------:|-------:|-------:|-------:|
|Llama2 7B    | perplexity   | 5.8664 | 6.0260 | 6.0656 | 6.4496 |
|Llama2 7B    | file size    |  12.9G  |   3.5G |   3.9G |   2.7G |
|Llama2 7B    | bits/weight  |   16.0 |    4.5 |    5.0 |    2.6 |
|AWQ-LLama2 7B| perplexity   | 5.8801 | 6.0054 | 5.9849 | 6.3650 |
|AWQ-LLama2 7B| file size    |  12.9G  |   3.5G |   3.9G |   2.7G |
|AWQ-LLama2 7B| bits/weight  |   16.0 |    4.5 |    5.0 |    2.6 |


### Mistral 7B v0.1 (Build with CuBLAS)

| Model        | Measure      | F16    | Q4_0   | Q4_1   | Q2_K   |
|-------------:|--------------|-------:|-------:|-------:|-------:|
|Mistral 7B    | perplexity   | 5.6931 | 5.8202 | 5.8268 | 6.1645 |
|Mistral 7B    | file size     |  14.5G |   4.1G |   4.5G |   3.1G |
|Mistral 7B    | bits/weight  |   16.0 |    4.5 |    5.0 |    2.6 |
|AWQ-Mistral 7B| perplexity   | 5.6934 | 5.8020 | 5.7691 | 6.0426 |
|AWQ-Mistral 7B| file size     |  14.5G |   4.1G |   4.5G |   3.1G |
|AWQ-Mistral 7B| bits/weight  |   16.0 |    4.5 |    5.0 |    2.6 |

### MPT 7B (Build with OpenBLAS)

| Model    | Measure      | F16    | Q4_0   | Q4_1   | Q2_K    |
|---------:|--------------|-------:|-------:|-------:|--------:|
|MPT 7B    | perplexity   | 8.4369 | 8.7956 | 8.6265 | 11.4913 |
|MPT 7B    | file size    |  13.7G  |   3.9G |   4.3G |   2.8G  |
|MPT 7B    | bits/weight  |   16.0 |    4.5 |    5.0 |    2.6  |
|AWQ-MPT 7B| perplexity   | 8.4944 | 8.7053 |  8.6750 | 10.2873|
|AWQ-MPT 7B| file size    |  13.7G  |   3.9G |   4.3G |   2.8G  |
|AWQ-MPT 7B| bits/weight  |   16.0 |    4.5 |    5.0 |    2.6  |
