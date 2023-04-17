import os
import argparse

from dataclasses import dataclass, field
from typing import List, Optional

# Based on https://github.com/ggerganov/llama.cpp/blob/master/examples/common.cpp


@dataclass
class GptParams:
    seed: int = -1
    n_threads: int = min(4, os.cpu_count() or 1)
    n_predict: int = 128
    repeat_last_n: int = 64
    n_parts: int = -1
    n_ctx: int = 512
    n_batch: int = 8
    n_keep: int = 0

    top_k: int = 40
    top_p: float = 0.95
    temp: float = 0.80
    repeat_penalty: float = 1.10

    model: str = "./models/llama-7B/ggml-model.bin"
    prompt: str = ""
    input_prefix: str = " "

    antiprompt: List[str] = field(default_factory=list)

    memory_f16: bool = True
    random_prompt: bool = False
    use_color: bool = False
    interactive: bool = False

    embedding: bool = False
    interactive_start: bool = False

    instruct: bool = False
    ignore_eos: bool = False
    perplexity: bool = False
    use_mmap: bool = True
    use_mlock: bool = False
    mem_test: bool = False
    verbose_prompt: bool = False

    file: str = None

    # If chat ended prematurely, append this to the conversation to fix it.
    # Set to "\nUser:" etc.
    # This is an alternative to input_prefix which always adds it, so it potentially duplicates "User:""
    fix_prefix: str = ""
    output_postfix: str = ""
    input_echo: bool = True,

    # Default instructions for Alpaca
    # switch to "Human" and "Assistant" for Vicuna.
    # TODO: TBD how they are gonna handle this upstream
    instruct_inp_prefix: str="\n\n### Instruction:\n\n"
    instruct_inp_suffix: str="\n\n### Response:\n\n"


def gpt_params_parse(argv = None, params: Optional[GptParams] = None):
    if params is None:
        params = GptParams()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--seed", type=int, default=-1, help="RNG seed (use random seed for <= 0)",dest="seed")
    parser.add_argument("-t", "--threads", type=int, default=min(4, os.cpu_count() or 1), help="number of threads to use during computation",dest="n_threads")
    parser.add_argument("-p", "--prompt", type=str, default="", help="initial prompt",dest="prompt")
    parser.add_argument("-f", "--file", type=str, default=None, help="file containing initial prompt to load",dest="file")
    parser.add_argument("-c", "--ctx_size", type=int, default=512, help="size of the prompt context",dest="n_ctx")
    parser.add_argument("--memory_f32", action="store_false", help="use f32 instead of f16 for memory key+value",dest="memory_f16")
    parser.add_argument("--top_p", type=float, default=0.95, help="top-p samplin",dest="top_p")
    parser.add_argument("--top_k", type=int, default=40, help="top-k sampling",dest="top_k")
    parser.add_argument("--temp", type=float, default=0.80, help="temperature",dest="temp")
    parser.add_argument("--n_predict", type=int, default=128, help="number of tokens to predict (-1 = infinity)",dest="n_predict")
    parser.add_argument("--repeat_last_n", type=int, default=64, help="last n tokens to consider for penalize ",dest="repeat_last_n")
    parser.add_argument("--repeat_penalty", type=float, default=1.10, help="penalize repeat sequence of tokens",dest="repeat_penalty")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size for prompt processing",dest="n_batch")
    parser.add_argument("--keep", type=int, default=0, help="number of tokens to keep from the initial prompt",dest="n_keep")
    parser.add_argument("-m", "--model", type=str, default="./models/llama-7B/ggml-model.bin", help="model path",dest="model")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="run in interactive mode", dest="interactive"
    )
    parser.add_argument("--embedding", action="store_true", help="", dest="embedding")
    parser.add_argument(
        "--interactive-start",
        action="store_true",
        help="run in interactive mode",
        dest="interactive"
    )
    parser.add_argument(
        "--interactive-first",
        action="store_true",
        help="run in interactive mode and wait for input right away",
        dest="interactive_start"
    )
    parser.add_argument(
        "-ins",
        "--instruct",
        action="store_true",
        help="run in instruction mode (use with Alpaca or Vicuna models)",
        dest="instruct"
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="colorise output to distinguish prompt and user input from generations",
        dest="use_color"
    )
    parser.add_argument("--mlock", action="store_true",help="force system to keep model in RAM rather than swapping or compressing",dest="use_mlock")
    parser.add_argument("--no-mmap", action="store_false",help="do not memory-map model (slower load but may reduce pageouts if not using mlock)",dest="use_mmap")
    parser.add_argument("--mtest", action="store_true",help="compute maximum memory usage",dest="mem_test")
    parser.add_argument("--verbose-prompt", action="store_true",help="print prompt before generation",dest="verbose_prompt")
    parser.add_argument(
        "-r",
        "--reverse-prompt",
        type=str,
        action='append',
        help="poll user input upon seeing PROMPT (can be\nspecified more than once for multiple prompts).",
        dest="antiprompt"
    )
    parser.add_argument("--perplexity", action="store_true", help="compute perplexity over the prompt", dest="perplexity")
    parser.add_argument("--ignore-eos", action="store_true", help="ignore end of stream token and continue generating", dest="ignore_eos")
    parser.add_argument("--n_parts", type=int, default=-1, help="number of model parts", dest="n_parts")
    parser.add_argument("--random-prompt", action="store_true", help="start with a randomized prompt.", dest="random_prompt")
    parser.add_argument("--in-prefix", type=str, default="", help="string to prefix user inputs with", dest="input_prefix")
    parser.add_argument("--fix-prefix", type=str, default="", help="append to input when generated n_predict tokens", dest="fix_prefix")
    parser.add_argument("--out-postfix", type=str, default="", help="append to input", dest="output_postfix")
    parser.add_argument("--input-noecho", action="store_false", help="dont output the input", dest="input_echo")
    args = parser.parse_args(argv)
    return args

def gpt_random_prompt(rng):
    return [
        "So",
        "Once upon a time",
        "When",
        "The",
        "After",
        "If",
        "import",
        "He",
        "She",
        "They",
    ][rng % 10]

if __name__ == "__main__":
    print(GptParams(gpt_params_parse()))
