# jarvis.cpp/examples/lookup

Demonstration of Prompt Lookup Decoding

https://github.com/apoorvumang/prompt-lookup-decoding

The key parameters for lookup decoding are `ngram_min`, `ngram_max` and `n_draft`. The first two determine the size of the ngrams to search for in the prompt for a match. The latter specifies how many subsequent tokens to draft if a match is found.

More info:

https://github.com/ggerganov/jarvis.cpp/pull/4484
https://github.com/ggerganov/jarvis.cpp/issues/4226
