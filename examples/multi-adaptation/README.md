Server multi adaptations for different scenario.

## Goal
Service multi scenarios on memory constrained devices. The offline models are in the same folder. Use the -mpa parameter to pass the alias and model path. Split the gguf model as below:


## Foundation model

The foundation model contains all the weights parameters used by the runtime. It play as shared split and will be referenced by other gguf models.

model-adaptor-taskA.gguf + model-foundation.gguf
model-adaptor-taskB.gguf + model-foundation.gguf
model-adaptor-base.gguf + model-foundation.gguf

## Model adaptation

Contains partial collections of the model weights that are overlaid onto the foundation model. These adaptation weights can be load dynamically and swapped out based on the usage.

## Example
Use the gguf splits in this model repo: https://huggingface.co/zhhan/Phi-3-mini-4k-instruct_multi-adaptor_gguf
Configuration to run multi-adaptation in visual studio:

{
  "type": "default",
  "project": "CMakeLists.txt",
  "projectTarget": "llama_multi-adaptation.exe (bin\\llama_multi-adaptation.exe)",
  "name": "llama_multi-adaptation.exe (bin\\llama_multi-adaptation.exe)",
    "args": [
        "-ngl 32",
        "-m models\phi3_adaptors\\Phi-3-mini-4k-instruct-ft-q4_att-adaptor-base.gguf",
        "-mpa codewriter=models\\phi3_adaptors\\Phi-3-mini-4k-instruct-ft-q4_att-adaptor-code_writer.gguf",
        "-mpa summarize=models\\phi3_adaptors\\Phi-3-mini-4k-instruct-ft-q4_att-adaptor-summarization.gguf",
        "-p \u0022\u003C|user|\u003EHow to explain Internet for a medieval knight?\u003C|end|\u003E\u003C|assistant|\u003E\u0022",
        "--color",
        "-c 4096",
        "--temp 0.7",
        "--repeat_penalty 1.1",
        "-n 256"
    ]
}
