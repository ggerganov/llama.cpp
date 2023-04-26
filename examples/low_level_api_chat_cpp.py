"""
This is an example implementation of main.cpp from llama.cpp
Quirks:
 * Its not exactly alike since this port is designed around programmatic I/O
 * Input is always echoed if on, so it should be turned off when using "input()"
 * The first antiprompt should be the userprompt like "\nUser:",
   because its added when n_predict is reached (aka generation ended prematurely)
 * n_predict can be set to -1 for unlimited length responses (or just a really high value)
 * Instruction mode adds its own antiprompt.
   You should also still be feeding the model with a "primer" prompt that
   shows it the expected format.
"""
import sys
from time import time
from os import cpu_count

import llama_cpp
from common import GptParams, gpt_params_parse, gpt_random_prompt

ANSI_COLOR_RESET = "\x1b[0m"
ANSI_COLOR_YELLOW = "\x1b[33m"
ANSI_BOLD = "\x1b[1m"
ANSI_COLOR_GREEN = "\x1b[32m"

CONSOLE_COLOR_DEFAULT = ANSI_COLOR_RESET
CONSOLE_COLOR_PROMPT = ANSI_COLOR_YELLOW
CONSOLE_COLOR_USER_INPUT = ANSI_BOLD + ANSI_COLOR_GREEN

# Iterative search
# Actively searches and prevents a pattern from being returned
class IterSearch:
    def __init__(self, pattern):
        self.pattern = list(pattern)
        self.buffer = []

    def __call__(self, char):
        self.buffer += [char]

        if (self.pattern[:len(self.buffer)] == self.buffer):
            if (len(self.buffer) >= len(self.pattern)):
                self.buffer.clear()
            return []

        _tmp = self.buffer[:]
        self.buffer.clear()
        return _tmp

# A LLaMA interactive session
class LLaMAInteract:
    def __init__(self, params: GptParams) -> None:
        # input args
        self.params = params

        if (self.params.perplexity):
            raise NotImplementedError("""************
please use the 'perplexity' tool for perplexity calculations
************""")

        if (self.params.embedding):
            raise NotImplementedError("""************
please use the 'embedding' tool for embedding calculations
************""")

        if (self.params.n_ctx > 2048):
            print(f"""warning: model does not support \
context sizes greater than 2048 tokens ({self.params.n_ctx} \
specified) expect poor results""", file=sys.stderr)

        if (self.params.seed <= 0):
            self.params.seed = int(time())

        print(f"seed = {self.params.seed}", file=sys.stderr)

        if (self.params.random_prompt):
            self.params.prompt = gpt_random_prompt(self.params.seed)

        # runtime args
        self.input_consumed = 0
        self.n_past = 0
        self.first_antiprompt = []
        self.remaining_tokens = self.params.n_predict
        self.output_echo = self.params.input_echo

        # model load
        self.lparams = llama_cpp.llama_context_default_params()
        self.lparams.n_ctx = self.params.n_ctx
        self.lparams.n_parts = self.params.n_parts
        self.lparams.seed = self.params.seed
        self.lparams.memory_f16 = self.params.memory_f16
        self.lparams.use_mlock = self.params.use_mlock
        self.lparams.use_mmap = self.params.use_mmap

        self.ctx = llama_cpp.llama_init_from_file(self.params.model.encode("utf8"), self.lparams)
        if (not self.ctx):
            raise RuntimeError(f"error: failed to load model '{self.params.model}'")

        print(file=sys.stderr)
        print(f"system_info: n_threads = {self.params.n_threads} / {cpu_count()} \
| {llama_cpp.llama_print_system_info().decode('utf8', errors='ignore')}", file=sys.stderr)

        # determine the required inference memory per token:
        if (self.params.mem_test):
            tmp = [0, 1, 2, 3]
            llama_cpp.llama_eval(self.ctx, (llama_cpp.c_int * len(tmp))(*tmp), len(tmp), 0, self.n_threads)
            llama_cpp.llama_print_timings(self.ctx)
            self.exit()
            return

        # create internal context
        self.n_ctx = llama_cpp.llama_n_ctx(self.ctx)

        # Add a space in front of the first character to match OG llama tokenizer behavior
        self.params.prompt = " " + self.params.prompt

        # Load prompt file
        if (self.params.file):
            with open(self.params.file) as f:
                self.params.prompt = f.read()

        # tokenize the prompt
        self.embd = []
        self.embd_inp = self._tokenize(self.params.prompt)

        if (len(self.embd_inp) > self.params.n_ctx - 4):
            raise RuntimeError(f"error: prompt is too long ({len(self.embd_inp)} tokens, max {self.params.n_ctx - 4})")

        # number of tokens to keep when resetting context
        if (self.params.n_keep < 0 or self.params.n_keep > len(self.embd_inp) or self.params.instruct):
            self.params.n_keep = len(self.embd_inp)

        self.inp_prefix = self._tokenize(self.params.instruct_inp_prefix)
        self.inp_suffix = self._tokenize(self.params.instruct_inp_suffix, False)

        # in instruct mode, we inject a prefix and a suffix to each input by the user
        if (self.params.instruct):
            self.params.interactive_start = True
            _ptn = self._tokenize(self.params.instruct_inp_prefix.strip(), False)
            self.first_antiprompt.append(_ptn)
            self.antiecho = IterSearch(_ptn)

        # enable interactive mode if reverse prompt or interactive start is specified
        if (len(self.params.antiprompt) != 0 or self.params.interactive_start):
            self.params.interactive = True

        # determine newline token
        self.llama_token_newline = self._tokenize("\n", False)
        self.llama_token_eot = self._tokenize(" [end of text]\n", False)

        if (self.params.verbose_prompt):
            print(f"""
prompt: '{self.params.prompt}'
number of tokens in prompt = {len(self.embd_inp)}""", file=sys.stderr)

            for i in range(len(self.embd_inp)):
                print(f"{self.embd_inp[i]} -> '{llama_cpp.llama_token_to_str(self.ctx, self.embd_inp[i])}'", file=sys.stderr)

            if (self.params.n_keep > 0):
                print("static prompt based on n_keep: '")
                for i in range(self.params.n_keep):
                    print(llama_cpp.llama_token_to_str(self.ctx, self.embd_inp[i]), file=sys.stderr)
                print("'", file=sys.stderr)
            print(file=sys.stderr)

        if (self.params.interactive):
            print("interactive mode on.", file=sys.stderr)

            if (len(self.params.antiprompt) > 0):
                for antiprompt in self.params.antiprompt:
                    print(f"Reverse prompt: '{antiprompt}'", file=sys.stderr)

            if len(self.params.input_prefix) > 0:
                print(f"Input prefix: '{self.params.input_prefix}'", file=sys.stderr)

        print(f"""sampling: temp = {self.params.temp},\
top_k = {self.params.top_k},\
top_p = {self.params.top_p},\
repeat_last_n = {self.params.repeat_last_n},\
repeat_penalty = {self.params.repeat_penalty}

generate: n_ctx = {self.n_ctx}, \
n_batch = {self.params.n_batch}, \
n_predict = {self.params.n_predict}, \
n_keep = {self.params.n_keep}
""", file=sys.stderr)

        # determine antiprompt tokens
        for i in self.params.antiprompt:
            self.first_antiprompt.append(self._tokenize(i, False))

        self.last_n_tokens = [0]*self.n_ctx #TODO: deque doesnt support slices

        if (params.interactive):
            print("""== Running in interactive mode. ==
 - Press Ctrl+C to interject at any time.
 - Press Return to return control to LLaMa.
 - If you want to submit another line, end your input in '\\'.

""", file=sys.stderr)
        self.set_color(CONSOLE_COLOR_PROMPT)

    # tokenize a prompt
    def _tokenize(self, prompt, bos=True):
        _arr = (llama_cpp.llama_token * (len(prompt) + 1))()
        _n = llama_cpp.llama_tokenize(self.ctx, prompt.encode("utf8", errors="ignore"), _arr, len(_arr), bos)
        return _arr[:_n]

    def set_color(self, c):
        if (self.params.use_color):
            print(c, end="")

    def use_antiprompt(self):
        return len(self.first_antiprompt) > 0

    # generate tokens
    def generate(self):
        while self.remaining_tokens > 0 or self.params.interactive or self.params.n_predict == -1:
            # predict
            if len(self.embd) > 0:
                # infinite text generation via context swapping
                # if we run out of context:
                # - take the n_keep first tokens from the original prompt (via n_past)
                # - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
                if (self.n_past + len(self.embd) > self.n_ctx):
                    n_left = self.n_past - self.params.n_keep
                    self.n_past = self.params.n_keep

                    # insert n_left/2 tokens at the start of embd from last_n_tokens
                    _insert = self.last_n_tokens[
                        self.n_ctx - int(n_left/2) - len(self.embd):-len(self.embd)
                    ]
                    self.embd = _insert + self.embd

                if (llama_cpp.llama_eval(
                    self.ctx, (llama_cpp.llama_token * len(self.embd))(*self.embd), len(self.embd), self.n_past, self.params.n_threads
                ) != 0):
                    raise Exception("Failed to llama_eval!")

            self.n_past += len(self.embd)
            self.embd = []
            if len(self.embd_inp) <= self.input_consumed:
                # out of user input, sample next token

                if (self.params.ignore_eos):
                    logits = llama_cpp.llama_get_logits(self.ctx)
                    logits[llama_cpp.llama_token_eos()] = llama_cpp.c_float(0)

                _arr = self.last_n_tokens[-min(self.params.repeat_last_n, self.n_past):]
                id = llama_cpp.llama_sample_top_p_top_k(
                    self.ctx,
                    (llama_cpp.llama_token * len(_arr))(*_arr),
                    len(_arr),
                    self.params.top_k,
                    self.params.top_p,
                    self.params.temp,
                    self.params.repeat_penalty,
                )
                self.last_n_tokens.pop(0)
                self.last_n_tokens.append(id)

                # replace end of text token with newline token when in interactive mode
                if (id == llama_cpp.llama_token_eos() and self.params.interactive and not self.params.instruct):
                    id = self.llama_token_newline[0]
                    if (self.use_antiprompt()):
                        # tokenize and inject first reverse prompt
                        self.embd_inp += self.first_antiprompt[0]

                # add it to the context
                self.embd.append(id)

                # echo this to console
                self.output_echo = True

                # decrement remaining sampling budget
                self.remaining_tokens -= 1
            else:
                # output to console if input echo is on
                self.output_echo = self.params.input_echo

                # some user input remains from prompt or interaction, forward it to processing
                while len(self.embd_inp) > self.input_consumed:
                    self.embd.append(self.embd_inp[self.input_consumed])
                    self.last_n_tokens.pop(0)
                    self.last_n_tokens.append(self.embd_inp[self.input_consumed])
                    self.input_consumed += 1
                    if len(self.embd) >= self.params.n_batch:
                        break

            # display tokens
            if self.output_echo:
                for id in self.embd:
                    if self.params.instruct:
                        for r in self.antiecho(id):
                            yield r
                    else:
                        yield id

            # reset color to default if we there is no pending user input
            if (self.params.input_echo and len(self.embd_inp) == self.input_consumed):
                self.set_color(CONSOLE_COLOR_DEFAULT)

            if (self.params.interactive and len(self.embd_inp) <= self.input_consumed):
                # if antiprompt is present, stop
                if (self.use_antiprompt()):
                    if True in [
                        i == self.last_n_tokens[-len(i):]
                        for i in self.first_antiprompt
                    ]:
                        break

                # if we are using instruction mode, and we have processed the initial prompt
                if (self.params.interactive_start):
                    break

            # end of text token
            if len(self.embd) > 0 and self.embd[-1] == llama_cpp.llama_token_eos():
                if (not self.params.instruct):
                    for i in self.llama_token_eot:
                        yield i
                break

            # respect n_predict even if antiprompt is present
            if (self.params.interactive and self.remaining_tokens <= 0 and self.params.n_predict != -1):
                # If we arent in instruction mode, fix the current generation by appending the antiprompt.
                # Makes it so if chat ends prematurely you dont append the AI's text etc.
                if not self.params.instruct:
                    self.embd_inp += self.first_antiprompt[0]
                self.n_remain = self.params.n_predict
                break

        self.params.interactive_start = False

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.exit()

    def exit(self):
        llama_cpp.llama_free(self.ctx)
        self.set_color(CONSOLE_COLOR_DEFAULT)

    # return past text
    def past(self):
        for id in self.last_n_tokens[-self.n_past:]:
            yield llama_cpp.llama_token_to_str(self.ctx, id).decode("utf-8", errors="ignore")

    # write input
    def input(self, prompt: str):
        if (self.params.instruct and self.last_n_tokens[-len(self.inp_prefix):] != self.inp_prefix):
            self.embd_inp += self.inp_prefix
        self.embd_inp += self._tokenize(prompt)
        if (self.params.instruct):
            self.embd_inp += self.inp_suffix

    # write output
    def output(self):
        self.remaining_tokens = self.params.n_predict
        for id in self.generate():
            yield llama_cpp.llama_token_to_str(self.ctx, id).decode("utf-8", errors="ignore")

    # read user input
    def read_input(self):
        out = ""
        while (t := input()).endswith("\\"):
            out += t[:-1] + "\n"
        return out + t + "\n"

    # interactive mode
    def interact(self):
        for i in self.output():
            print(i,end="",flush=True)
        self.params.input_echo = False

        while self.params.interactive:
            self.set_color(CONSOLE_COLOR_USER_INPUT)
            if (self.params.instruct):
                print('\n> ', end="")
                self.input(self.read_input())
            else:
                print(self.params.input_prefix, end="")
                self.input(f"{self.params.input_prefix}{self.read_input()}{self.params.output_postfix}")
                print(self.params.output_postfix,end="")
            self.set_color(CONSOLE_COLOR_DEFAULT)

            try:
                for i in self.output():
                    print(i,end="",flush=True)
            except KeyboardInterrupt:
                self.set_color(CONSOLE_COLOR_DEFAULT)
                if not self.params.instruct:
                    print(self.params.fix_prefix,end="")
                    self.input(self.params.fix_prefix)

if __name__ == "__main__":
    from datetime import datetime

    USER_NAME="User"
    AI_NAME="ChatLLaMa"

    time_now = datetime.now()
    prompt = f"""Text transcript of a never ending dialog, where {USER_NAME} interacts with an AI assistant named {AI_NAME}.
{AI_NAME} is helpful, kind, honest, friendly, good at writing and never fails to answer {USER_NAME}’s requests immediately and with details and precision.
There are no annotations like (30 seconds passed...) or (to himself), just what {USER_NAME} and {AI_NAME} say aloud to each other.
The dialog lasts for years, the entirety of it is shared below. It's 10000 pages long.
The transcript only includes text, it does not include markup like HTML and Markdown.

{USER_NAME}: Hello, {AI_NAME}!
{AI_NAME}: Hello {USER_NAME}! How may I help you today?
{USER_NAME}: What time is it?
{AI_NAME}: It is {time_now.strftime("%H:%M")}.
{USER_NAME}: What year is it?
{AI_NAME}: We are in {time_now.strftime("%Y")}.
{USER_NAME}: What is a cat?
{AI_NAME}: A cat is a domestic species of small carnivorous mammal. It is the only domesticated species in the family Felidae.
{USER_NAME}: Name a color.
{AI_NAME}: Blue
{USER_NAME}:"""
    args = gpt_params_parse()
    params = GptParams(**vars(args))

    with LLaMAInteract(params) as m:
        m.interact()
