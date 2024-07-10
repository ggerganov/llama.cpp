# Server Multi Adaptations for Different Scenarios

## Goal
Service multiple scenarios on memory-constrained devices. The GGUF models are stored in the same folder.

## Usage
Use the `-mpa` parameter to pass the alias and model path.

### Flag to Switch Derived Model
```c
llama_ctx_switch_derived_model(ctx, "summarize");
```

### Pass Model Path and Alias for Derived Models
```sh
llama_multi-adaptation.exe -m models\Phi-3-mini-4k-instruct-adaptor-base.gguf \
 -mpa code_writer=models\Phi-3-mini-4k-instruct-adaptor-code_writer.gguf \
 -mpa summarize=models\Phi-3-mini-4k-instruct-adaptor-summarization.gguf
```

## Foundation Model
The **foundation** GGUF contains the weights shared across models.
The **adaptor** GGUF contains the task-specific weights.

Here are the combinations for hosting three models:
- `model-adaptor-base.gguf (0.77GB) + model-foundation.gguf (1.56GB)`
- `model-adaptor-taskA.gguf + model-foundation.gguf`
- `model-adaptor-taskB.gguf + model-foundation.gguf`

The benefit is that it supports hosting multiple scenarios while keeping only one copy of the shared weights in memory. With the benefit of `mmap`, the task-specific GGUF is only loaded when the corresponding task is called.

## Example
Use the GGUF splits in this model repository: [Phi-3-mini-4k-instruct_multi-adaptor_gguf](https://huggingface.co/zhhan/Phi-3-mini-4k-instruct_multi-adaptor_gguf)
