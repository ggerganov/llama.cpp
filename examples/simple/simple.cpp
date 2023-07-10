#include <stdio.h>
#include <string>
#include <vector>

#include "llama.h"


void generate_sequence(llama_context * ctx, int n_ctx, const std::vector<llama_token>& prompt_tokens, float temperature) {
    // print the tokens from the prompt
    for (llama_token id : prompt_tokens) {
        printf("%s", llama_token_to_str(ctx, id));
    }
    fflush(stdout);

    // the maximum number of tokens to generate at a time
    // TODO: not supported, remove
    const int CUDA_MAX_TOKENS = 1;
    llama_token tokens_out[CUDA_MAX_TOKENS];

    // current position in the context window
    int n_past = 0;

    // number of tokens to generate
    int n_tokens_out;

    // list of tokens to evaluate
    // note that at most llama_context_params::n_batch tokens can be evaluated at a time
    std::vector<llama_token> token_list = prompt_tokens;

    while (n_past < n_ctx) {
        // evaluate the tokens

        // llama_eval generates one token at a time
        n_tokens_out = 1;

        // number of threads to use for CPU evaluation - ignored if compiled with CUDA support
        const int n_threads = 4;
        // note: llama_eval is not compatible with GPU sampling
        if (llama_eval(ctx, token_list.data(), token_list.size(), n_past, n_threads)) {
            fprintf(stderr, "%s : failed to eval\n", __func__ );
            exit(1);
        }

        // perform sampling on the CPU
        float * logits  = llama_get_logits(ctx);
        auto n_vocab = llama_n_vocab(ctx);

        // initialize candidate array from logits
        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for(llama_token token_id = 0 ; token_id < n_vocab ; token_id++) {
            candidates.push_back(llama_token_data{ token_id, logits[token_id], 0.0f});
        }

        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

        // sample token
        llama_sample_temperature(ctx, &candidates_p, temperature);
        tokens_out[0] = llama_sample_token(ctx, &candidates_p);

        // increment the position in the context window
        n_past += token_list.size() + n_tokens_out - 1;

        token_list.clear();

        // print the new tokens
        for (int i = 0; i < n_tokens_out; i++) {
            llama_token new_token_id = tokens_out[i];

            // is it an end of stream ?
            if (new_token_id == llama_token_eos()) {
                fprintf(stderr, " [end of text]\n");
                //return;
            }

            // print the new token :
            printf("%s", llama_token_to_str(ctx, new_token_id));
        }
        fflush(stdout);

        // push the last new token for the next evaluation
        token_list.push_back(tokens_out[n_tokens_out - 1]);
    }
}

int main(int argc, char ** argv) {
    if (argc < 2 || argv[1][0] == '-') {
        printf("usage: %s <model> <n_ctx> <n_gens> <temp> [prompt]\n", argv[0]);
        printf(" note: passing a temp parameter will enable GPU sampling\n");
        return 1 ;
    }

    std::string model = argv[1];
    struct llama_context_params lparams = llama_context_default_params();

    if (argc >= 3) {
        lparams.n_ctx = std::stoi(argv[2]);
    } else {
        lparams.n_ctx = 512;
    }

    int n_gens;
    if (argc >= 4) {
        n_gens = std::stoi(argv[3]);
    } else {
        n_gens = 1;
    }

    float temperature;

    if (argc >= 5) {
        temperature = std::stof(argv[4]);
    } else {
        temperature = 0.8f;
    }

    std::string prompt;
    if (argc >= 6) {
        prompt = argv[5];
    } else {
        prompt = "Hello my name is";
    }

    // initialize llama.cpp
    bool numa = false;
    llama_init_backend(numa);

    llama_model * lmodel  = llama_load_model_from_file(model.c_str(), lparams);
    if (lmodel == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, model.c_str());
        return 1;
    }

    llama_context * ctx = llama_new_context_with_model(lmodel, lparams);
    if (ctx == NULL) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, model.c_str());
        llama_free_model(lmodel);
        return 1;
    }

    // tokenize the prompt
    std::vector<llama_token> token_list(lparams.n_ctx);
    int prompt_tokens = llama_tokenize(ctx, prompt.c_str(), token_list.data(), token_list.size(), true);
    if (prompt_tokens <= 0) {
        fprintf(stderr, "%s: error: unable to tokenize prompt\n", __func__);
        return 1;
    }

    token_list.resize(prompt_tokens);

    const int max_context_size     = llama_n_ctx(ctx);
    const int max_tokens_list_size = max_context_size - 4 ;

    if ((int)token_list.size() > max_tokens_list_size) {
        fprintf( stderr, "%s: error: prompt too long (%d tokens, max %d)\n" ,
             __func__, (int)token_list.size(), max_tokens_list_size );
        return 1;
    }

    fprintf(stderr, "\n\n");

    // generate the sequences
    for (int i = 0; i < n_gens; i++) {
        printf("==== GENERATION %d ====\n", i + 1);
        generate_sequence(ctx, max_context_size, token_list, temperature);
        printf("\n\n");
    }

    llama_print_timings(ctx);
    llama_free(ctx);

    return 0;
}
