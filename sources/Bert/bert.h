#ifndef BERT_H
#define BERT_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(_WIN32)
#define BERT_API __declspec(dllexport)
#else
#define BERT_API __attribute__ ((visibility ("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct bert_params
{
    int32_t n_threads = 6;
    int32_t port = 8080; // server mode port to bind

    const char* model = "models/all-MiniLM-L6-v2/ggml-model-q4_0.bin"; // model path
    const char* prompt = "test prompt";
};

BERT_API bool bert_params_parse(int argc, char **argv, bert_params &params);

struct bert_ctx;

typedef int32_t bert_vocab_id;

BERT_API struct bert_ctx * bert_load_from_file(const char * fname);
BERT_API void bert_free(bert_ctx * ctx);

// Main api, does both tokenizing and evaluation

BERT_API void bert_encode(
    struct bert_ctx * ctx,
    int32_t n_threads,
    const char * texts,
    float * embeddings);

// n_batch_size - how many to process at a time
// n_inputs     - total size of texts and embeddings arrays
BERT_API void bert_encode_batch(
    struct bert_ctx * ctx,
    int32_t n_threads,
    int32_t n_batch_size,
    int32_t n_inputs,
    const char ** texts,
    float ** embeddings);

// Api for separate tokenization & eval

BERT_API void bert_tokenize(
    struct bert_ctx * ctx,
    const char * text,
    bert_vocab_id * tokens,
    int32_t * n_tokens,
    int32_t n_max_tokens);

BERT_API void bert_eval(
    struct bert_ctx * ctx,
    int32_t n_threads,
    bert_vocab_id * tokens,
    int32_t n_tokens,
    float * embeddings);

// NOTE: for batch processing the longest input must be first
BERT_API void bert_eval_batch(
    struct bert_ctx * ctx,
    int32_t n_threads,
    int32_t n_batch_size,
    bert_vocab_id ** batch_tokens,
    int32_t * n_tokens,
    float ** batch_embeddings);

BERT_API int32_t bert_n_embd(bert_ctx * ctx);
BERT_API int32_t bert_n_max_tokens(bert_ctx * ctx);

BERT_API const char* bert_vocab_id_to_token(bert_ctx * ctx, bert_vocab_id id);

#ifdef __cplusplus
}
#endif

#endif // BERT_H
