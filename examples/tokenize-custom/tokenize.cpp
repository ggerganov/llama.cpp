#include "common.h"
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <unordered_map>

// Token -> Word map for demonstration
std::unordered_map<llama_token, std::string> token_to_word_map = {
    {400, "hello"},
    {960, "world"},
    {74, "example"}
};

// Custom tokenizer that splits words by spaces and generates a "token" for each word
int32_t custom_tokenizer(const char * text, int32_t text_len, llama_token * tokens, int32_t max_tokens) {
    std::istringstream stream(std::string(text, text_len));
    std::string word;
    int32_t token_count = 0;

    while (stream >> word) {
        if (token_count >= max_tokens) {
            return -token_count; // Indicate error: too many tokens
        }
        // Generate a "fake" token as an integer hash of the word
        tokens[token_count++] = std::hash<std::string>{}(word) % 1000;
    }

    return token_count;
}

// Custom detokenizer that maps tokens back to words
int32_t custom_detokenizer(const llama_token * tokens, int32_t n_tokens, char * text, int32_t text_len_max) {
    std::ostringstream result;

    for (int i = 0; i < n_tokens; i++) {
        auto it = token_to_word_map.find(tokens[i]);
        if (it != token_to_word_map.end()) {
            result << it->second << " ";
        } else {
            result << "<UNK> ";
        }
    }

    std::string result_str = result.str();
    if (result_str.length() >= (size_t)text_len_max) {
        return -1; // Error: result too long
    }
    std::strncpy(text, result_str.c_str(), text_len_max);
    return result_str.size();
}

void print_usage_information(const char * program_name) {
    printf("Usage: %s --model MODEL_PATH -p \"PROMPT_TEXT\"\n", program_name);
    printf("Options:\n");
    printf("    --model MODEL_PATH     Path to the model file.\n");
    printf("    -p, --prompt PROMPT    Text prompt to tokenize and detokenize.\n");
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * prompt = nullptr;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        } else {
            print_usage_information(argv[0]);
            return 1;
        }
    }

    // Validate arguments
    if (!model_path || !prompt) {
        print_usage_information(argv[0]);
        return 1;
    }

    llama_backend_init();

    // Load model with provided path
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "Error: Could not load model from '%s'.\n", model_path);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Error: Could not create context for the model.\n");
        llama_free_model(model);
        return 1;
    }

    // Set custom tokenizer and detokenizer
    llama_set_custom_tokenizer(custom_tokenizer, custom_detokenizer);

    // Tokenize the prompt using the custom tokenizer
    std::vector<llama_token> tokens(100); // Allocate space for tokens
    int32_t token_count = llama_tokenize(model, prompt, std::strlen(prompt), tokens.data(), tokens.size(), true, false);
    if (token_count < 0) {
        fprintf(stderr, "Error: custom tokenizer produced too many tokens.\n");
        llama_free(ctx);
        llama_free_model(model);
        return 1;
    }

    // Detokenize the tokens back into text
    char detokenized_text[256];
    int32_t detokenized_length = llama_detokenize(model, tokens.data(), token_count, detokenized_text, sizeof(detokenized_text), true, true);
    if (detokenized_length < 0) {
        fprintf(stderr, "Error: Detokenized text exceeds buffer size.\n");
        llama_free(ctx);
        llama_free_model(model);
        return 1;
    }

    // Display tokenized and detokenized output
    printf("Tokenized output:\n");
    for (int i = 0; i < token_count; i++) {
        printf("Token %d -> %d\n", i, tokens[i]);
    }

    printf("\nDetokenized output:\n%s\n", detokenized_text);

    // Clean up
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
