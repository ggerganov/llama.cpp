#include <stdlib.h>
#include <stdio.h>
#include "../../llama.h"

/*
The FIM (Fill-In-Middle) objective is useful for generating text conditioned on a prefix and a suffix.
For a quick summary of what's going on here, see issue #2818.
*/


static inline struct llama_context* create_codellama_fim_context(const char* model_path) {
    struct llama_context_params params = llama_context_default_params();
    struct llama_model* model = llama_load_model_from_file(model_path,  params);
    struct llama_context* context = llama_new_context_with_model(model, params);
    return context;
}

static inline char*
codellama_fill_in_middle(struct llama_context* ctx, const char* prefix, const char* suffix, size_t n_max_tokens, int n_threads, bool spm, char** error_message) {

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
    int num_prompt_tokens = (int)(tokens_end - tokens);
    *tokens_end++ = llama_token_middle(ctx);

    // Evaluate the LM on the prompt.
    if (llama_eval(ctx, tokens, num_prompt_tokens, 0, n_threads)) {
        *error_message = "Failed to evaluate the prompt.";
        free(tokens);
        return NULL;
    }

    // Generate tokens until n_max_tokens or the <EOT> token is generated.
    int num_generated_tokens = 0;
    llama_token* generated_tokens = NULL;
    for (size_t i = 0; i < n_max_tokens; i++) {
        // Evaluate the LM for a single token
        llama_token* current_token = generated_tokens + num_generated_tokens;
        if (llama_eval(ctx, current_token, 1, num_generated_tokens, n_threads)) {
            *error_message = "Failed to evaluate the prompt.";
            free(tokens);
            break;
        }

        if (*current_token == llama_token_eot(ctx)) {
            break;
        }

        num_generated_tokens++;
    }

    // Allocate memory for the final result
    size_t result_length = 0;
    size_t result_capacity = 4096;
    char* result = (char*)malloc(sizeof(char) * 4096);
    if (!result) {
        *error_message = "Failed to allocate memory for result.";
        free(tokens);
        return NULL;
    }

    // Translate tokens to string
    for (size_t i = 0; i < num_generated_tokens; i++) {
        int appended = llama_token_to_str(ctx, generated_tokens[i], result, result_capacity - result_length);
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

    struct llama_context* ctx = create_codellama_fim_context(argv[1]);

    size_t n_max_tokens = atoi(argv[4]);
    int n_threads = atoi(argv[5]);
    bool spm = false;
    char* error_message = NULL;
    char* result = codellama_fill_in_middle(ctx, argv[2], argv[3], n_max_tokens, n_threads, spm, &error_message);

    if (error_message) {
        fprintf(stderr, "Error: %s\n", error_message);
        return 1;
    }

    printf("%s\n", result);

    llama_free(ctx);
}
