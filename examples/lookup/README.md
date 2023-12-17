# llama.cpp/examples/lookup

Demonstration of Prompt Lookup Decoding

https://github.com/apoorvumang/prompt-lookup-decoding

The two key parameters for lookup decoding are `max_ngram_size` and `n_draft`. The first, determines how many ngrams to use when searching through the prompt for a match and the second specifies how many subsequent tokens to draft if a match is found.

More info:

https://github.com/ggerganov/llama.cpp/pull/4484
https://github.com/ggerganov/llama.cpp/issues/4226

