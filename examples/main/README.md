# llama.cpp/example/main

This example program allows you to use various LLaMA language models in an easy and efficient way. It is specifically designed to work with the [llama.cpp](https://github.com/ggerganov/llama.cpp) project, which provides a plain C/C++ implementation with optional 4-bit quantization support for faster, lower memory inference, and is optimized for desktop CPUs. This program can be used to perform various inference tasks with LLaMA models, including generating text based on user-provided prompts and chat-like interactions with reverse prompts.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Common Options](#common-options)
3. [Input Prompts](#input-prompts)
4. [Interaction](#interaction)
5. [Context Management](#context-management)
6. [Generation Flags](#generation-flags)
7. [Performance Tuning and Memory Options](#performance-tuning-and-memory-options)
8. [Additional Options](#additional-options)

## Quick Start

To get started right away, run the following command, making sure to use the correct path for the model you have:

```bash
./main -m models/7B/ggml-model.bin --prompt "Once upon a time"
```

The following command generates "infinite" text from a starting prompt (you can use `Ctrl-C` to stop it):

```bash
./main -m models/7B/ggml-model.bin --ignore-eos --n_predict -1 --keep -1 --prompt "Once upon a time"
```

For an interactive experience, try this command:

```bash
./main -m models/7B/ggml-model.bin -n -1 --color -r "User:" --in-prefix " " --prompt $'User: Hi\nAI: Hello. I am an AI chatbot. Would you like to talk?\nUser: Sure!\nAI: What would you like to talk about?\nUser:'
```

Note that the newline characters in the prompt string above only work on Linux. On Windows, you will have to use the ``--file`` option (see below) to load a multi-line prompt from file instead.

## Common Options

In this section, we cover the most commonly used options for running the `main` program with the LLaMA models:

-   `-m FNAME, --model FNAME`: Specify the path to the LLaMA model file (e.g., `models/7B/ggml-model.bin`).
-   `-i, --interactive`: Run the program in interactive mode, allowing you to provide input directly and receive real-time responses.
-   `-ins, --instruct`: Run the program in instruction mode, which is particularly useful when working with Alpaca models.
-   `-t N, --threads N`: Set the number of threads to use during computation. It is recommended to set this to the number of physical cores your CPU has.
-   `-n N, --n_predict N`: Set the number of tokens to predict when generating text. Adjusting this value can influence the length of the generated text.
-   `-c N, --ctx_size N`: Set the size of the prompt context. The default is 512, but LLaMA models were built with a context of 2048, which will provide better results for longer input/inference.

## Input Prompts

The `main` program provides several ways to interact with the LLaMA models using input prompts:

-   `--prompt PROMPT`: Provide a prompt directly as a command-line option.
-   `--file FNAME`: Provide a file containing a prompt or multiple prompts.
-   `--interactive-first`: Run the program in interactive mode and wait for input right away. (More on this below.)
-   `--random-prompt`: Start with a randomized prompt.

## Interaction

The `main` program offers a seamless way to interact with LLaMA models, allowing users to engage in real-time conversations or provide instructions for specific tasks. The interactive mode can be triggered using various options, including `--interactive`, `--interactive-first`, and `--instruct`.

In interactive mode, users can participate in text generation by injecting their input during the process. Users can press `Ctrl+C` at any time to interject and type their input, followed by pressing `Return` to submit it to the LLaMA model. To submit additional lines without finalizing input, users can end the current line with a backslash (`\`) and continue typing.

### Interaction Options

-   `-i, --interactive`: Run the program in interactive mode, allowing users to engage in real-time conversations or provide specific instructions to the model.
-   `--interactive-first`: Run the program in interactive mode and immediately wait for user input before starting the text generation.
-   `-ins, --instruct`: Run the program in instruction mode, which is specifically designed to work with Alpaca models that excel in completing tasks based on user instructions.
-   `--color`: Enable colorized output to differentiate visually distinguishing between prompts, user input, and generated text.

By understanding and utilizing these interaction options, you can create engaging and dynamic experiences with the LLaMA models, tailoring the text generation process to your specific needs.

### Reverse Prompts

Reverse prompts are a powerful way to create a chat-like experience with a LLaMA model by pausing the text generation when specific text strings are encountered:

-   `-r PROMPT, --reverse-prompt PROMPT`: Specify one or multiple reverse prompts to pause text generation and switch to interactive mode. For example, `-r "User:"` can be used to jump back into the conversation whenever it's the user's turn to speak. This helps create a more interactive and conversational experience. However, the reverse prompt doesn't work when it ends with a space.

To overcome this limitation, you can use the `--in-prefix` flag to add a space or any other characters after the reverse prompt.

### In-Prefix

The `--in-prefix` flag is used to add a prefix to your input, primarily, this is used to insert a space after the reverse prompt. Here's an example of how to use the `--in-prefix` flag in conjunction with the `--reverse-prompt` flag:

```sh
./main -r "User:" --in-prefix " "
```

### Instruction Mode

Instruction mode is particularly useful when working with Alpaca models, which are designed to follow user instructions for specific tasks:

-   `-ins, --instruct`: Enable instruction mode to leverage the capabilities of Alpaca models in completing tasks based on user-provided instructions.

Technical detail: the user's input is internally prefixed with the reverse prompt (or ``### Instruction:`` as the default), and followed by ``### Response:`` (except if you just press Return without any input, to keep generating a longer response).

By understanding and utilizing these interaction options, you can create engaging and dynamic experiences with the LLaMA models, tailoring the text generation process to your specific needs.

## Context Management

During text generation, LLaMA models have a limited context size, which means they can only consider a certain number of tokens from the input and generated text. When the context fills up, the model resets internally, potentially losing some information from the beginning of the conversation or instructions. Context management options help maintain continuity and coherence in these situations.

### Context Size

The `--ctx_size` option allows you to set the size of the prompt context used by the LLaMA models during text generation. A larger context size helps the model to better comprehend and generate responses for longer input or conversations.

-   `-c N, --ctx_size N`: Set the size of the prompt context (default: 512). The LLaMA models were built with a context of 2048, which will yield the best results on longer input/inference. However, increasing the context size beyond 2048 may lead to unpredictable results.

### Keep Prompt

The `--keep` option allows users to retain the original prompt when the model runs out of context, ensuring a connection to the initial instruction or conversation topic is maintained.

-   `--keep N`: Specify the number of tokens from the initial prompt to retain when the model resets its internal context. By default, this value is set to 0 (meaning no tokens are kept). Use `-1` to retain all tokens from the initial prompt.

By utilizing context management options like `--ctx_size` and `--keep`, you can maintain a more coherent and consistent interaction with the LLaMA models, ensuring that the generated text remains relevant to the original prompt or conversation.

## Generation Flags

The following options are related to controlling the text generation process, influencing the diversity, creativity, and quality of the generated text. Understanding these options will help you fine-tune the output according to your needs:

### Number of Tokens to Predict

-   `-n N, --n_predict N`: Set the number of tokens to predict when generating text (default: 128, -1 = infinity).

The `--n_predict` option controls the number of tokens the model generates in response to the input prompt. By adjusting this value, you can influence the length of the generated text. A higher value will result in longer text, while a lower value will produce shorter text. A value of -1 will cause text to be generated without limit.

It is important to note that the generated text may be shorter than the specified number of tokens if an End-of-Sequence (EOS) token or a reverse prompt is encountered. In interactive mode text generation will pause and control will be returned to the user. In non-interactive mode, the program will end. In both cases, the text generation may stop before reaching the specified `n_predict` value. If you want the model to keep going without ever producing End-of-Sequence on its own, you can use the ``--ignore-eos`` parameter.

### RNG Seed

-   `-s SEED, --seed SEED`: Set the random number generator (RNG) seed (default: -1).

The RNG seed is used to initialize the random number generator that influences the text generation process. By setting a specific seed value, you can obtain consistent and reproducible results across multiple runs with the same input and settings. This can be helpful for testing, debugging, or comparing the effects of different options on the generated text to see when they diverge. If the seed is set to a value less than or equal to 0, a random seed will be used, which will result in different outputs on each run.

### Temperature

-   `--temp N`: Adjust the randomness of the generated text (default: 0.8).

Temperature is a hyperparameter that controls the randomness of the generated text. It affects the probability distribution of the model's output tokens. A higher temperature (e.g., 1.5) makes the output more random and creative, while a lower temperature (e.g., 0.5) makes the output more focused, deterministic, and conservative. The default value is 0.8, which provides a balance between randomness and determinism. At the extreme, a temperature of 0 will always pick the most likely next token, leading to identical outputs in each run.

Example usage: `--temp 0.8`

### Repeat Penalty

-   `--repeat_penalty N`: Control the repetition of token sequences in the generated text (default: 1.1).

Repeat penalty is a hyperparameter used to penalize the repetition of token sequences during text generation. It helps prevent the model from generating repetitive or monotonous text. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. The default value is 1.1.

Example usage: `--repeat_penalty 1.1`

### Top-K Sampling

-   `--top_k N`: Limit the next token selection to the K most probable tokens (default: 40).

Top-k sampling is a text generation method that selects the next token only from the top k most likely tokens predicted by the model. It helps reduce the risk of generating low-probability or nonsensical tokens, but it may also limit the diversity of the output. A higher value for top_k (e.g., 100) will consider more tokens and lead to more diverse text, while a lower value (e.g., 10) will focus on the most probable tokens and generate more conservative text. The default value is 40.

Example usage: `--top_k 40`

### Top-P Sampling

-   `--top_p N`: Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P (default: 0.9).

Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. The default value is 0.9.

Example usage: `--top_p 0.9`

By adjusting these options, you can control the diversity, quality, and creativity of the generated text to better suit your needs. You can experiment with different combinations of values to find the best settings for your specific use case.

## Performance Tuning and Memory Options

These options help improve the performance and memory usage of the LLaMA models:

-   `-t N, --threads N`: Set the number of threads to use during computation. Using the correct number of threads can greatly improve performance. It is recommended to set this value to the number of CPU cores.
-   `--mlock`: Lock the model in memory, preventing it from being swapped out when mmaped. This can improve performance.
-   `--no-mmap`: Do not memory-map the model. This results in a slower load time but may reduce pageouts if you're not using `mlock`.
-   `--memory_f32`: Use 32 bit floats instead of 16 bit floats for memory key+value, allowing higher quality inference at the cost of memory.
-   `-b N, --batch_size N`: Set the batch size for prompt processing (default: 512). This large batch size benefits users who have BLAS installed and enabled it during the build. If you don't have BLAS enabled ("BLAS=0"), you can use a smaller number, such as 8, to see the prompt progress as it's evaluated in some situations.

For information about 4-bit quantization, which can significantly improve performance and reduce memory usage, please refer to llama.cpp's primary [README](../../README.md#prepare-data--run).

By understanding and using these performance tuning settings, you can optimize the LLaMA model's behavior to achieve the best performance for your specific needs.

## Additional Options

These options provide extra functionality and customization when running the LLaMA models:

-   `-h, --help`: Display a help message showing all available options and their default values. This is particularly useful for checking the latest options and default values, as they can change frequently, and the information in this document may become outdated.
-   `--verbose-prompt`: Print the prompt before generating text.
-   `--mtest`: Test the model's functionality by running a series of tests to ensure it's working properly.
-   `--lora FNAME`: Apply a LoRA (Low-Rank Adaptation) adapter to the model (implies --no-mmap). This allows you to adapt the pretrained model to specific tasks or domains.
-   `--lora-base FNAME`: Optional model to use as a base for the layers modified by the LoRA adapter. This flag is used in conjunction with the `--lora` flag, and specifies the base model for the adaptation.
