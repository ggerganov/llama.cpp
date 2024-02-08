#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys

import yaml

CLI_ARGS_MAIN_PERPLEXITY = [
    "batch-size", "cfg-negative-prompt", "cfg-scale", "chunks", "color", "ctx-size", "escape",
    "export", "file", "frequency-penalty", "grammar", "grammar-file", "hellaswag",
    "hellaswag-tasks", "ignore-eos", "in-prefix", "in-prefix-bos", "in-suffix", "instruct",
    "interactive", "interactive-first", "keep", "logdir", "logit-bias", "lora", "lora-base",
    "low-vram", "main-gpu", "memory-f32", "mirostat", "mirostat-ent", "mirostat-lr", "mlock",
    "model", "multiline-input", "n-gpu-layers", "n-predict", "no-mmap", "no-mul-mat-q",
    "np-penalize-nl", "numa", "ppl-output-type", "ppl-stride", "presence-penalty", "prompt",
    "prompt-cache", "prompt-cache-all", "prompt-cache-ro", "random-prompt", "repeat-last-n",
    "repeat-penalty", "reverse-prompt", "rope-freq-base", "rope-freq-scale", "rope-scale", "seed",
    "simple-io", "tensor-split", "threads", "temp", "tfs", "top-k", "top-p", "typical",
    "verbose-prompt"
]

CLI_ARGS_LLAMA_BENCH = [
    "batch-size", "memory-f32", "low-vram", "model", "mul-mat-q", "n-gen", "n-gpu-layers",
    "n-prompt", "output", "repetitions", "tensor-split", "threads", "verbose"
]

CLI_ARGS_SERVER = [
    "alias", "batch-size", "ctx-size", "embedding", "host", "memory-f32", "lora", "lora-base",
    "low-vram", "main-gpu", "mlock", "model", "n-gpu-layers", "n-probs", "no-mmap", "no-mul-mat-q",
    "numa", "path", "port", "rope-freq-base", "timeout", "rope-freq-scale", "tensor-split",
    "threads", "verbose"
]

description = """Run llama.cpp binaries with presets from YAML file(s).
To specify which binary should be run, specify the "binary" property (main, perplexity, llama-bench, and server are supported).
To get a preset file template, run a llama.cpp binary with the "--logdir" CLI argument.

Formatting considerations:
- The YAML property names are the same as the CLI argument names of the corresponding binary.
- Properties must use the long name of their corresponding llama.cpp CLI arguments.
- Like the llama.cpp binaries the property names do not differentiate between hyphens and underscores.
- Flags must be defined as "<PROPERTY_NAME>: true" to be effective.
- To define the logit_bias property, the expected format is "<TOKEN_ID>: <BIAS>" in the "logit_bias" namespace.
- To define multiple "reverse_prompt" properties simultaneously the expected format is a list of strings.
- To define a tensor split, pass a list of floats.
"""
usage = "run-with-preset.py [-h] [yaml_files ...] [--<ARG_NAME> <ARG_VALUE> ...]"
epilog = ("  --<ARG_NAME> specify additional CLI ars to be passed to the binary (override all preset files). "
          "Unknown args will be ignored.")

parser = argparse.ArgumentParser(
    description=description, usage=usage, epilog=epilog, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-bin", "--binary", help="The binary to run.")
parser.add_argument("yaml_files", nargs="*",
                    help="Arbitrary number of YAML files from which to read preset values. "
                    "If two files specify the same values the later one will be used.")

known_args, unknown_args = parser.parse_known_args()

if not known_args.yaml_files and not unknown_args:
    parser.print_help()
    sys.exit(0)

props = dict()

for yaml_file in known_args.yaml_files:
    with open(yaml_file, "r") as f:
        props.update(yaml.load(f, yaml.SafeLoader))

props = {prop.replace("_", "-"): val for prop, val in props.items()}

binary = props.pop("binary", "main")
if known_args.binary:
    binary = known_args.binary

if os.path.exists(f"./{binary}"):
    binary = f"./{binary}"

if binary.lower().endswith("main") or binary.lower().endswith("perplexity"):
    cli_args = CLI_ARGS_MAIN_PERPLEXITY
elif binary.lower().endswith("llama-bench"):
    cli_args = CLI_ARGS_LLAMA_BENCH
elif binary.lower().endswith("server"):
    cli_args = CLI_ARGS_SERVER
else:
    print(f"Unknown binary: {binary}")
    sys.exit(1)

command_list = [binary]

for cli_arg in cli_args:
    value = props.pop(cli_arg, None)

    if not value or value == -1:
        continue

    if cli_arg == "logit-bias":
        for token, bias in value.items():
            command_list.append("--logit-bias")
            command_list.append(f"{token}{bias:+}")
        continue

    if cli_arg == "reverse-prompt" and not isinstance(value, str):
        for rp in value:
            command_list.append("--reverse-prompt")
            command_list.append(str(rp))
        continue

    command_list.append(f"--{cli_arg}")

    if cli_arg == "tensor-split":
        command_list.append(",".join([str(v) for v in value]))
        continue

    value = str(value)

    if value != "True":
        command_list.append(str(value))

num_unused = len(props)
if num_unused > 10:
    print(f"The preset file contained a total of {num_unused} unused properties.")
elif num_unused > 0:
    print("The preset file contained the following unused properties:")
    for prop, value in props.items():
        print(f"  {prop}: {value}")

command_list += unknown_args

sp = subprocess.Popen(command_list)

while sp.returncode is None:
    try:
        sp.wait()
    except KeyboardInterrupt:
        pass

sys.exit(sp.returncode)
