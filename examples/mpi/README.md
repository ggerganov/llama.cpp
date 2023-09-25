# llama.cpp/example/mpi

This example program allows you to use various LLaMA language models in an easy and efficient way across an MPI cluster.
It is specifically designed to work with the [llama.cpp](https://github.com/ggerganov/llama.cpp) project, which provides a plain C/C++ implementation with optional 4-bit quantization support for faster, lower memory inference, and is optimized for desktop CPUs. This program can be used to perform various inference tasks with LLaMA models, including generating text based on user-provided prompts and chat-like interactions with reverse prompts.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Common Options](#common-options)

## Quick Start

To get started right away, write the following to a file on each node, making sure to use the correct path for the model you have:
```bash
--mpi-layer-split 0.8,0.2 -t 4 -m ~/llm-local/codellama-7b.Q3_K_M.gguf --color -c 512 --temp 0.0 --repeat_penalty 1.0 -n 128 -p "double fast_inverse_square_root(double x"
```

Each node may have different options, currently they must have the same number of arguments to the mpi-layer-split option and the same
model path, but that will eventually be synchronized from the head node.

Next, write the hostsfile on the head node. Make sure there is only one slot on each node.

Finally, run the following command on the head node to start the program across the cluster:

#### Unix-based systems (Linux, macOS, etc.):

```bash
mpirun -hostfile hostsfile -mca orte_keep_fqdn_hostnames t --bind-to none ./mpi options.txt
```

Where `hostsfile` is the file containing the cluster hostname configuration and `options.txt` is the path
where each node can find its own options. Storing the model on a network filesystem has not yet been
tested and optimized for.

#### Windows:
Not supported currently.

For an interactive experience, try this command:

#### Unix-based systems (Linux, macOS, etc.):

```bash
./main -m models/7B/ggml-model.bin -n -1 --color -r "User:" --in-prefix " " \
'User: Hi
AI: Hello. I am an AI chatbot. Would you like to talk?
User: Sure!
AI: What would you like to talk about?
User:'
```

## Common Options

In this section, we cover the most commonly used options for running the `mpi` program with the LLaMA models:

-   `-m FNAME, --model FNAME`: Specify the path to the LLaMA model file (e.g., `models/7B/ggml-model.bin`).
-   `-i, --interactive`: Run the program in interactive mode, allowing you to provide input directly and receive real-time responses.
-   `-ins, --instruct`: Run the program in instruction mode, which is particularly useful when working with Alpaca models.
-   `-n N, --n-predict N`: Set the number of tokens to predict when generating text. Adjusting this value can influence the length of the generated text.
-   `-c N, --ctx-size N`: Set the size of the prompt context. The default is 512, but LLaMA models were built with a context of 2048, which will provide better results for longer input/inference.
-   `--mpi-layer-split`: Set the percentage of layers to distribute to each node. Must have the same number of arguments as the number of nodes in the cluster. Only the layer split percentages passed to the head node are used, they are scattered to all other nodes in the cluster.
