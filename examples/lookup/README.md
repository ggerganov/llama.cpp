# llama.cpp/examples/lookup

Demonstration of speculative decoding using n-gram lookup.
Initial version was based on https://github.com/apoorvumang/prompt-lookup-decoding .
The current version uses three separate types of "n-gram caches".
Each of these caches maps how frequently a given n-gram is followed by a specific token.
The difference between the caches lies in what data is used to build them:

* The "context" cache is built using the tokens in the current context of a user generation.
* The "dynamic" cache is built by merging the context caches of previous user generations.
* The "static" cache is built from a large text corpus with no relation to the current context.

The tradeoff between these caches lies in relevance to the current context vs. the emount of input data.
When trying to draft a new token using n-gram lookup the algorithm is as follows:

* Try to draft a suitable token from the context cache. If a static cache is available, use it to validate the draft candidates. This is done by simply multiplying the frequencies of the two caches.
* Try to draft a suitable token from the dynamic cache, validate with static cache if available.
* Try to draft a suitable token from the static cache.

Only a single token sequence with the most likely token candidates is drafted.
All tokens must pass thresholds for frequency and sample size in order to be drafted.

Relevant command line arguments:

- `--draft`: maximum number of additional tokens to draft using n-gram lookup. Default: 5. Set to 0 to disable n-gram lookup. **Results are not deterministic with n-gram lookup enabled due to varying batch size.**
- `-lcs FNAME, --lookup-cache-static FNAME`: optional path to static lookup cache to use for n-gram lookup. Created from a large, unspecific text corpus using `lookup-create`.
- `-lcd FNAME, --lookup-cache-dynamic FNAME`: optional path to dynamic lookup cache to use for n-gram lookup. Contains data from previous generations. Automatically created and filled while the server is running but by default discarded on server exit. Setting this argument tries to initialize the dynamic cache from a file and saves it to said file on server shutdown.

N-gram lookup caches saved to disk are compatible between models as long as they use the same tokenizer
(but for dynamic caches the resulting drafted tokens may be wrong which means there is no speedup).
Furthermore, the data format for both types of caches is the same so they can be used interchangeably (but probably not with good results).

## Usage Examples

### `lookup`

Generation using n-gram lookup:

``` sh
./lookup --model models/opt/llama_2-7b-q4_0.gguf -ngl 99 --n-predict 256 --ignore-eos --draft 3 --color --prompt "Write a love story about two stars that tragically ends in a type Ia supernova. Use a lot of emotional and dramatic language."
```

The `--color` flag highlights the successfully predicted tokens.
The `--lookup-cache-static` and `--lookup-cache-dynamic` arguments can be set to provide static/dynamic caches.

### `lookup-stats`

Determine n-gram lookup effectiveness for a given text corpus (similar to `perplexity`):

``` sh
./lookup-stats --model /opt/models/llama_2-7b-q4_0.gguf --file wikitext-2-raw/wiki.test.raw --draft 3
```

The `--lookup-cache-static` and `--lookup-cache-dynamic` arguments can be set to provide static/dynamic caches.

### `lookup-create`

Create a static lookup cache from a text corpus:

``` sh
./lookup-create --model /opt/models/llama_2-7b-q4_0.gguf --lookup-cache-static wt103-llama_2.lcs --file wikitext-103-raw/wiki.train.raw
```

The `--lookup-cache-static` argument must be set to provide the path to which the static lookup cache will be saved.
The tokenizer for which to create the cache is taken from the provided model.

### `lookup-merge`

Merge two lookup caches into one:

``` sh
./lookup-merge cache_1.lcs cache_2.lcs cache_merged.lcs
```

Can be used for both static and dynamic lookup caches.

## More info:

* https://github.com/ggerganov/llama.cpp/pull/4484
* https://github.com/ggerganov/llama.cpp/issues/4226
* https://github.com/ggerganov/llama.cpp/pull/5479
* https://github.com/ggerganov/llama.cpp/pull/6828
