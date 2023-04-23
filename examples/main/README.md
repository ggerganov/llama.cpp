# main

This example shows how to run a LLaMA-like model for chatting or generating text. There are two basic modes of operation:

- Text generation: starting from an initial prompt, produce additional text as predicted by the model
- Chat: alternate between user input to complete a prompt and generating some text based on the input

## Essential parameters

- ``--model``: the pathname of the model file to load (if not specified: ``models/llama-7B/ggml-model.bin``)
- ``--prompt``: the first prompt used to initialize the model.
- Alternatively, you can use ``--file`` to load the prompt from a file (UTF-8 encoded, may include line breaks).

**Note**: Most parameters also have short forms (such as ``-m`` for the model filepath). For clarity, all examples use the long form.

## Text generation

The most basic application for an LLM is producing more text from a given prompt.

    ./main --model models/alpaca-7B/ggml-alpaca-7b-q4.bin --prompt "Hello World"

This will run until the model predicts an end-of-text token, or until ``--n_predict`` tokens have been generated (a value of ``-1`` means unlimited). If you want the model to keep going without ever predicting end-of-text on its own, use the ``--ignore-eos`` parameter.

When generating "infinite" text, the model will at some point exhaust its context size, that is, the number of past tokens it can remember. When this happens, the oldest half of the context is forgotten, and the most recent half is used to seamlessly continue the text. Text generation will temporarily
slow down until the context is recomputed. If you want the model to remember the initial prompt (rather than just continuing from the most recent text), you can pass ``--keep -1`` on the command line (to remember the full prompt), or give a specific number of tokens to remember. In theory, this should lead to more consistency across longer texts.

So, a useful starting point for long text generation (using the default llama-7B model) would be:

    ./main --ignore-eos --n_predict -1 --keep -1 --prompt "Hello World"

## Chat mode

Chat can be viewed as "co-creating" a stream of text between the user and the model: first, the context is initialized with a starting prompt (``--prompt`` or ``--file``, for example describing the character of the assistant, and giving a brief example of a typical conversation). Control is then passed to the user, who completes this prompt with their request. From this point, the model is used to generate the tokens of the reply, until the model predicts either end-of-text or emits a phrase that is defined with the ``--reverse-prompt`` parameter. This is usually a phrase such as "User:" that indicates a dialog turn:

    User: Please tell me the largest city in Europe.
    Bob: Sure. The largest city in Europe is Moscow, the capital of Russia.
    User:

There are three different chat modes, mostly with small differences in how phrases are handled that indicate conversation turns:

1. ``--interactive-first``: initializes the context with the starting prompt, then passes control to the user. From the user's input, keeps generating a response until a reverse prompt or end-of-text has been predicted, and then passes control back.
   - If the response ends with an end-of-text token, the user's next input is prefixed with the reverse prompt, if any, to preserve the turn structure of the dialog.
2. ``--interactive``: same as ``--interactive-first``, but immediately generates text after the initial prompt, until a reverse prompt or end-of-text is encountered.
3. ``--instruct``: same as ``--interactive-first``, but with the following differences:
   - When the context is full (see above), the initial prompt is remembered when switching to the new context (same as ``--keep -1``)
   - The user input is quietly prefixed with the reverse prompt (or ``### Instruction:`` as the default), and followed by ``### Response:`` (except if you just press Return without any input, to keep generating a longer response).

The ``--n_predict`` parameter limits the length of the model's responses before giving control back to the user. In *interactive* mode, this does not insert the reverse prompt before the user's input. The same happens if you use Ctrl-C to interrupt a response.

If your terminal supports it, you can use ``--color`` to visually distinguish user input from model output.

Example:

    ./main --file .\prompts\chat-with-bob.txt --interactive -r "User:" --keep -1

runs a basic chat where the prompt primes the model to expect one of the dialog participants to be named "User:", and the initial prompt is kept on context switches. In many cases, model authors suggest reasonable values for these parameters. However, the defaults are usually fine to get started.

## Sampling parameters

The following parameters control sampling, that is, how a token is randomly selected from the most probable candidates that are predicted by the model. For more background, see [https://huggingface.co/blog/how-to-generate](https://huggingface.co/blog/how-to-generate).

- ``--seed``: the starting value of the random number generator. If you use a positive value, a given prompt will produce the same output in each run. A negative value (the default) will usually produce somewhat different output in each run.
- ``--temp``: the "temperature" parameter of the softmax function. A higher temperature means less likely words are being picked more often. Conversely, a temperature of 0 will always pick the most likely next token, leading to identical outputs in each run.
- ``--top_k``, ``--top_p``: restrict the selection of the final token candidates to the *k* most likely, or the tokens that combine to a probability mass of at least *p*.
- ``--repeat_last_n``, ``--repeat_penalty``: reduces repetitions within the last *n* tokes by applying a penalty to repeated tokens.
