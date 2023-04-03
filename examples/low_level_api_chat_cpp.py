"""
This is an example implementation of main.cpp from llama.cpp
Quirks:
 * Its not exactly alike since this port is designed around programmatic I/O
 * Input is always echoed if on, so it should be turned off when using "input()"
 * The first antiprompt should be the userprompt like "\nUser:",
   because its added when n_predict is reached (aka generation ended prematurely)
 * n_predict can be set to -1 for unlimited length responses
"""
import llama_cpp

def toIntArray(lst):
    return [int(i) for i in lst]

# A LLaMA interactive session
class LLaMAInteract:
    def __init__(self,
        primer: str="",
        model: str="./models/30B/ggml-model-q4_0.bin",
        n_ctx: int=1024,
        seed: int=0,
        n_threads: int=8,
        antiprompt: list[str]=[],
        input_echo: bool=True,
        n_predict: int=20,
        n_batch: int=8,
        repeat_last_n: int=64,
        top_k: int=50,
        top_p: float=1.,
        temp: float=1.0,
        repeat_penalty: float=1,
    ) -> None:
        # input args
        self.n_threads = n_threads
        self.input_echo = input_echo
        self.n_predict = n_predict
        self.n_batch = n_batch
        self.repeat_last_n = repeat_last_n
        self.top_k=top_k
        self.top_p=top_p
        self.temp=temp
        self.repeat_penalty=repeat_penalty
        self.n_ctx = n_ctx
        self.seed = seed

        # runtime args
        self.input_consumed = 0
        self.embd = []
        self.embd_inp = []
        self.n_past = 0
        self.first_antiprompt = []
        self.remaining_tokens = self.n_predict
        self.output_echo = input_echo

        # model load
        self.lparams = llama_cpp.llama_context_default_params()
        self.lparams.n_ctx = self.n_ctx
        self.lparams.seed = self.seed
        self.ctx = llama_cpp.llama_init_from_file(model.encode("utf8"), self.lparams)

        # determine the required inference memory per token:
        tmp = [0, 1, 2, 3]
        llama_cpp.llama_eval(self.ctx, (llama_cpp.c_int * len(tmp))(*tmp), len(tmp), 0, self.n_threads)

        # determine newline token
        self.llama_token_newline = (llama_cpp.llama_token * 1)()
        llama_cpp.llama_tokenize(self.ctx, b"\n", self.llama_token_newline, len(self.llama_token_newline), False)
        self.llama_token_newline = toIntArray(self.llama_token_newline)

        # primer feed
        if (len(primer) > 0):
            self.input(primer)
        self.n_keep = len(self.embd_inp)

        # create internal context
        self.n_ctx = int(llama_cpp.llama_n_ctx(self.ctx))
        self.last_n_tokens = [0]*self.n_ctx #TODO: deque doesnt support slices

        # determine antiprompt tokens
        for i in antiprompt:
            d_antiprompt = (llama_cpp.llama_token * (len(i) + 1))()
            n_antiprompt = llama_cpp.llama_tokenize(self.ctx, i.encode("utf8"), d_antiprompt, len(d_antiprompt), False)
            self.first_antiprompt.append(toIntArray(d_antiprompt[:n_antiprompt]))

    # if an antiprompt is present
    def use_antiprompt(self):
        return len(self.first_antiprompt) > 0

    def generate(self):
        while self.remaining_tokens > 0 or self.use_antiprompt():
            # predict
            if len(self.embd) > 0:
                # infinite text generation via context swapping
                # if we run out of context:
                # - take the n_keep first tokens from the original prompt (via n_past)
                # - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
                if (self.n_past + len(self.embd) > self.n_ctx):
                    n_left = self.n_past - self.n_keep
                    self.n_past = self.n_keep

                    # insert n_left/2 tokens at the start of embd from last_n_tokens
                    _insert = self.last_n_tokens[
                        -(int(n_left/2) - len(self.embd)):-len(self.embd)
                    ]
                    self.embd[:len(_insert)] = _insert
                    #TODO: Still untested

                if (llama_cpp.llama_eval(
                    self.ctx, (llama_cpp.llama_token * len(self.embd))(*self.embd), len(self.embd), self.n_past, self.n_threads
                ) != 0):
                    raise Exception("Failed to llama_eval!")

            self.n_past += len(self.embd)
            self.embd = []
            if len(self.embd_inp) <= self.input_consumed:
                # out of user input, sample next token
                _arr = self.last_n_tokens[-min(self.repeat_last_n, self.n_past):]
                id = llama_cpp.llama_sample_top_p_top_k(
                    self.ctx,
                    (llama_cpp.llama_token * len(_arr))(*_arr),
                    len(_arr),
                    self.top_k,
                    self.top_p,
                    self.temp,
                    self.repeat_penalty,
                )
                self.last_n_tokens.pop(0)
                self.last_n_tokens.append(int(id))

                # replace end of text token with newline token when in interactive mode
                if (id == llama_cpp.llama_token_eos() and self.use_antiprompt()):
                    id = self.llama_token_newline[0]
                    # tokenize and inject first reverse prompt
                    self.embd_inp += self.first_antiprompt[0]

                # add it to the context
                self.embd.append(int(id))

                # echo this to console
                self.output_echo = True

                # decrement remaining sampling budget
                self.remaining_tokens -= 1
            else:
                # output to console if input echo is on
                self.output_echo = self.input_echo

                # some user input remains from prompt or interaction, forward it to processing
                while len(self.embd_inp) > self.input_consumed:
                    self.embd.append(int(self.embd_inp[self.input_consumed]))
                    self.last_n_tokens.pop(0)
                    self.last_n_tokens.append(int(self.embd_inp[self.input_consumed]))
                    self.input_consumed += 1
                    if len(self.embd) >= self.n_batch:
                        break

            # display tokens
            if self.output_echo:
                for id in self.embd:
                    yield id

            # if antiprompt is present, stop
            if (self.use_antiprompt() and len(self.embd_inp) <= self.input_consumed):
                for i in self.first_antiprompt:
                    if i == self.last_n_tokens[-len(i):]:
                        return

            # if end of generation
            if len(self.embd) > 0 and self.embd[-1] == llama_cpp.llama_token_eos():
                break

            # respect n_predict even if antiprompt is present
            if (self.use_antiprompt() and self.remaining_tokens <= 0 and self.n_predict != -1):
                self.embd_inp += self.first_antiprompt[0]
                break

    def past(self):
        for id in self.last_n_tokens[-self.n_past:]:
            yield llama_cpp.llama_token_to_str(self.ctx, id).decode("utf-8")

    def input(self, prompt: str):
        embd_arr = (llama_cpp.llama_token * (len(prompt) + 1))()
        n_of_tok = llama_cpp.llama_tokenize(self.ctx, prompt.encode("utf8"), embd_arr, len(embd_arr), True)
        self.embd_inp += toIntArray(embd_arr[:n_of_tok])

    def output(self):
        self.remaining_tokens = self.n_predict
        for id in self.generate():
            yield llama_cpp.llama_token_to_str(self.ctx, id).decode("utf-8")

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

    print("Loading model...")
    ll = LLaMAInteract(prompt,
        model="./models/30B/ggml-model-q4_0.bin",
        n_ctx=2048,
        antiprompt=[f"\n{USER_NAME}:"],
        repeat_last_n=256,
        n_predict=2048,
        temp=0.7, top_p=0.5, top_k=40, repeat_penalty=1.17647
    )
    print("Loaded model!")

    for i in ll.output():
        print(i,end="",flush=True)
    ll.input_echo = False

    inp = lambda x: f" {x}\n"
    while True:
        ll.input(inp(input(' ')))
        for i in ll.output():
            print(i,end="",flush=True)