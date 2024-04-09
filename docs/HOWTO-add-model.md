# HOW TO

## Add a new model architecture to `llama.cpp`

LLaMA C++ is built on top of [ggml](https://github.com/ggerganov/ggml) Tensor library for machine learning.
Model are stored in [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md).

#### Quick start

Adding a model requires few steps:

1. Convert the model to GGUF
2. Define the model architecture in `llama.cpp`
3. Build the GGML graph implementation

After following this step, you can open PR.

Also, it is important to check that the examples and main ggml backends (CUDA, METAL, CPU) are working with the new architecture, especially:
- [main](../examples/main)
- [imatrix](../examples/imatrix)
- [quantize](../examples/quantize)
- [server](../examples/server)

### 1. Convert the model to GGUF

This step is done in python with a `convert` script using [gguf-writer](https://pypi.org/project/gguf/) library.
Depending on the model architecture, you can use either [convert.py](../convert.py) or [convert-hf-to-gguf.py](../convert-hf-to-gguf.py).

The convert script reads the model configuration, tokenizer, tensor names+data and convert them to GGUF Metadata and tensors.

The required steps to implement for an HF model are:

1. Define the model `Model.register` annotation in a new `Model` subclass, example:

```python
@Model.register("MyModelForCausalLM")
class MyModel(Model):
    model_arch = gguf.MODEL_ARCH.GROK
```

2. Define the layout of the GGUF tensors in [constants.py](../gguf-py/gguf/constants.py)

Add an enum entry in `MODEL_ARCH`, the model human friendly name in `MODEL_ARCH_NAMES` and the GGUF tensor names in `MODEL_TENSORS`.

Example for `falcon` model:
```python
    MODEL_ARCH.FALCON: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_NORM_2,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ]
```

3. Map the original tensor names to the standardize equivalent in GGUF

As a general rule, before adding a new tensor name to GGUF, be sure the equivalent naming does not already exist.

Once you have found the GGUF tensor name equivalent, add it to the [tensor_mapping.py](../gguf-py/gguf/tensor_mapping.py) file.

If the tensor name is part of a repetitive layer/block, the key word `bid` substitutes it.

Example for the normalization tensor in attention layers:

```python
block_mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {
        # Attention norm
        MODEL_TENSOR.ATTN_NORM: (
            "gpt_neox.layers.{bid}.input_layernorm",                # gptneox
            "transformer.h.{bid}.ln_1",                             # gpt2 gpt-j refact qwen
            "transformer.blocks.{bid}.norm_1",                      # mpt
            ...
        )
}
```

`transformer.blocks.{bid}.norm_1` will be mapped to `blk.{bid}.attn_norm` in GGUF.

Depending on the model configuration, tokenizer, code and tensors layout, you will have to override:
- `Model#set_gguf_parameters`
- `Model#set_vocab`
- `Model#write_tensors`

NOTE: Tensor names must end with `.weight` suffix, that is the convention and several tools like `quantize` expect this to proceed weights.

### 2. Define the model architecture in `llama.cpp`

The model params and tensors layout must be defined in `llama.cpp`:
1. Define a new `llm_arch`
2. Define the tensors layout in `LLM_TENSOR_NAMES`
3. Add any non standard metadata in `llm_load_hparams`
4. Create the tensors for inference in `llm_load_tensors`

NOTE: The dimensions in `ggml` are typically in the reverse order of the `pytorch` dimensions.

### 3. Build the GGML graph implementation

This is the funniest part, you have to provide the inference graph implementation of the new model architecture in `llama_build_graph`.

Have a look to existing implementation like `build_llama`, `build_dbrx` or `build_bert`.

When implementing a new graph, please note that the underlying `ggml` backends do not support them all, support of missing backend operations can be added in another PR.

## Terminology

| term    | description                                                                                                                                       | link                                                       |
|---------|---------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| GGML    | Georgi Gerganov Model Language a.k.a GPT-Generated Model Language                                                                                 | https://github.com/ggerganov/ggml                          |
| GGUF    | GGML Universal File a.k.a GPT-Generated Unified Format, successor to GGML format, GGUF’s creation aligns with the needs of large-scale AI models. | https://github.com/ggerganov/ggml/blob/master/docs/gguf.md |

## Resources

- [GGML - Large Language Models for Everyone](https://github.com/rustformers/llm/blob/main/crates/ggml/README.md): a
  description of the GGML format provided by the maintainers of the `llm` Rust crate, which provides Rust bindings for
  GGML
- YaRN RoPE scaling #2268
- support Baichuan serial models #3009
- support attention bias #4283
- Mixtral support #4406
- BERT embeddings #5423
- Grok-1 support #6204
- Command R Plus support #6491
- support arch DBRX #6515
- How to convert HuggingFace model to GGUF format #2948
