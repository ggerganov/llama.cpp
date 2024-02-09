# llama.cpp/examples/imatrix

Compute an importance matrix for a model and given text dataset. Can be used during quantization to enchance the quality of the quantum models.
More information is available here: https://github.com/ggerganov/llama.cpp/pull/4861

## Usage

```
./imatrix -m <some_fp_model> -f <some_training_data> [-o <output_file>] [--verbosity <verbosity_level>]
        [-ofreq num_chunks] [-ow <0 or 1>] [other common params]
```

Here `-m` with a model name and `-f` with a file containing training data (such as e.g. `wiki.train.raw`) are mandatory.
The parameters in square brackets are optional and have the following meaning:
* `-o` (or `--output-file`) specifies the name of the file where the computed data will be stored. If missing `imatrix.dat` is used.
* `--verbosity` specifies the verbosity level. If set to `0`, no output other than the perplexity of the processed chunks will be generated. If set to `1`, each time the results are saved a message is written to `stderr`. If `>=2`, a message is output each time data is collected for any tensor. Default verbosity level is `1`.
* `-ofreq` (or `--output-frequency`) specifies how often the so far computed result is saved to disk. Default is 10 (i.e., every 10 chunks)
* `-ow` (or `--output-weight`) specifies if data will be collected for the `output.weight` tensor. My experience is that it is better to not utilize the importance matrix when quantizing `output.weight`, so this is set to `false` by default.

For faster computation, make sure to use GPU offloading via the `-ngl` argument

## Example

```bash
LLAMA_CUBLAS=1 make -j

# generate importance matrix (imatrix.dat)
./imatrix -m ggml-model-f16.gguf -f train-data.txt -ngl 99

# use the imatrix to perform a Q4_K_M quantization
./quantize --imatrix imatrix.dat ggml-model-f16.gguf ./ggml-model-q4_k_m.gguf q4_k_m
```
