#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3 || argv[1][0] == '-') {
        printf("usage: %s MODEL_PATH PROMPT [--ids]\n" , argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    const char * prompt     = argv[2];

    const bool printing_ids = argc > 3 && std::string(argv[3]) == "--ids";

    llama_backend_init(false);

    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = true;
    llama_model * model = llama_load_model_from_file(model_path, model_params);

    llama_context_params ctx_params = llama_context_default_params();
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    const bool add_bos = llama_should_add_bos_token(model);

    std::vector<llama_token> tokens;

    tokens = ::llama_tokenize(model, prompt, add_bos, true);

    for (int token : tokens) {
        if (printing_ids) {
            printf("%d\n", token);
        } else {
            printf("%6d -> '%s'\n", token, llama_token_to_piece(ctx, token).c_str());
        }
    }

    return 0;
}
