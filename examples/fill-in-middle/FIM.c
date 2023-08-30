#include <stdlib.h>
#include <stdio.h>
#include "../../llama.h"

/*
The FIM (Fill-In-Middle) objective is useful for generating text conditioned on a prefix and a suffix.
For a quick summary of what's going on here, see issue #2818.
*/


static inline struct llama_context* codellama_create_fim_context(const char* model_path, const char** error_message) {
    struct llama_context_params params = llama_context_default_params();
    struct llama_model* model = llama_load_model_from_file(model_path,  params);
    if (!model) {
        *error_message = "Failed to load model.";
        return NULL;
    }

    struct llama_context* context = llama_new_context_with_model(model, params);
    if (!context) {
        *error_message = "Failed to create context.";
        llama_free_model(model);
        return NULL;
    }

    return context;
}

static inline char*
codellama_fill_in_middle(struct llama_context* ctx, const char* prefix, const char* suffix, size_t n_max_tokens, int n_threads, bool spm, const char** error_message) {

    int num_tokens;
    llama_token* tokens_end = (llama_token*)malloc(sizeof(llama_token) * n_max_tokens);
    llama_token* tokens = tokens_end;
    if (!tokens) {
        *error_message = "Failed to allocate memory for tokens.";
        return NULL;
    }

    // Append first part of prompt
    *tokens_end++ = spm ? llama_token_suffix(ctx) : llama_token_prefix(ctx);
    tokens_end += num_tokens = llama_tokenize(ctx, spm ? suffix : prefix, tokens_end, n_max_tokens, 1);
    if (num_tokens < 0) {
        *error_message = "Failed to tokenize the prompt.";
        free(tokens);
        return NULL;
    }

    // Append second part of prompt
    *tokens_end++ = spm ? llama_token_prefix(ctx) : llama_token_suffix(ctx);
    tokens_end += num_tokens = llama_tokenize(ctx, spm ? prefix : suffix, tokens_end, n_max_tokens, 1);
    if (num_tokens < 0) {
        *error_message = "Failed to tokenize the prompt.";
        free(tokens);
        return NULL;
    }

    // Append middle token
    *tokens_end++ = llama_token_middle(ctx);

    // Evaluate the LM on the prompt.
    if (llama_eval(ctx, tokens, (int)(tokens_end - tokens), 0, n_threads)) {
        *error_message = "Failed to evaluate the prompt.";
        free(tokens);
        return NULL;
    }

    // Generate tokens until n_max_tokens or the <EOT> token is generated.
    llama_token* generated_tokens = NULL;
    size_t num_generated_tokens = 0;
    int vocab_size = llama_n_vocab(ctx);
    for (size_t i = 0; i < n_max_tokens; i++) {
        // Evaluate the LM for a single token, obtaining the logits and probabilities.
        if (llama_eval(ctx, &generated_tokens[num_generated_tokens], 1, (int)num_generated_tokens, n_threads)) {
            *error_message = "Failed to evaluate the prompt.";
            free(tokens);
            break;
        }
        float* logits = llama_get_logits(ctx);

        // From the logits, select the most likely token.
        float highest_log_likelihood = -1;
        llama_token likeliest_token = -1;
        for (llama_token token_id = 0; token_id < vocab_size; token_id++) {
            if (logits[token_id] > highest_log_likelihood) {
                highest_log_likelihood = logits[token_id];
                likeliest_token = token_id;
            }
        }

        // Don't add the token if it's <EOT>.
        if (likeliest_token == llama_token_eot(ctx)) {
            break;
        }

        // Append the token, so it's there for subsequent evaluations.
        generated_tokens[num_generated_tokens++] = likeliest_token;
    }

    // Allocate memory for the final result
    size_t result_length = 0;
    size_t result_capacity = 4096;
    char* result = (char*)malloc(sizeof(char) * result_capacity);
    if (!result) {
        *error_message = "Failed to allocate memory for result.";
        free(tokens);
        return NULL;
    }

    // Translate tokens to string, growing the allocation if it's too small.
    for (size_t i = 0; i < num_generated_tokens; i++) {
        int appended = llama_token_to_piece(ctx, generated_tokens[i], result, result_capacity - result_length);
        if (appended < 0) {
            // retry the token with a larger buffer
            i--;
            size_t new_capacity = result_capacity * 2;
            char* new_result = (char*)realloc(result, sizeof(char) * new_capacity);
            if (!new_result) {
                *error_message = "Failed to allocate memory for result.";
                free(tokens);
                free(result);
                return NULL;
            }
            result = new_result;
            result_capacity = new_capacity;
        }

        result_length += appended;
    }

    free(tokens);
    *error_message = NULL;
    return result;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <model> <prefix> <suffix> <n_max_tokens> <n_threads>\n", argv[0]);
        return 1;
    }

    char* model = argv[1];
    char* prefix = argv[2];
    char* suffix = argv[3];
    size_t n_max_tokens = atoi(argv[4]);
    int n_threads = atoi(argv[5]);
    bool spm = false;
    const char* error_message = NULL;

    struct llama_context* ctx = codellama_create_fim_context(model, &error_message);
    if (error_message) {
        fprintf(stderr, "Error: %s\n", error_message);
        return 1;
    }

    char* result = codellama_fill_in_middle(ctx, prefix, suffix, n_max_tokens, n_threads, spm, &error_message);
    if (error_message) {
        fprintf(stderr, "Error: %s\n", error_message);
        return 1;
    }

    printf("%s%s%s\n", prefix, result, suffix);

    llama_free(ctx);
}
