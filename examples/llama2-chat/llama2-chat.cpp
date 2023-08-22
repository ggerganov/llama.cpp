#include "build-info.h"
#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <iostream>

static const std::string B_INST = "[INST]";
static const std::string E_INST = "[/INST]";
static const std::string B_SYS = "<<SYS>>\n";
static const std::string E_SYS = "\n<<SYS>>\n\n";

struct chat {
    llama_context_params lparams;
    llama_model * model;
    llama_context * ctx;

    std::string system;
    int n_threads = 8;

    chat(const std::string & model_file, const std::string & system) : system(system) {
        lparams = llama_context_default_params();
        lparams.n_ctx = 4096;
        lparams.n_gpu_layers = 99;

        model = llama_load_model_from_file(model_file.c_str(), lparams);
        if (model == NULL) {
            fprintf(stderr , "%s: error: unable to load model\n" , __func__);
            exit(1);
        }

        ctx = llama_new_context_with_model(model, lparams);
        if (ctx == NULL) {
            fprintf(stderr , "%s: error: unable to create context\n" , __func__);
            exit(1);
        }
    }

    std::vector<llama_token> tokenize_dialog(const std::string & user, const std::string & assistant = "") {
        std::string content;
        // B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"],
        if (!system.empty()) {
            content = B_SYS + system + E_SYS + user;
            system.clear();
        } else {
            content = user;
        }
        // f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"
        std::string prompt;
        prompt = B_INST + " " + content + " " + E_INST;

        // f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
        if (!assistant.empty()) {
            prompt += " " + assistant;
        }

        // printf("prompt: %s\n", prompt.c_str());

        auto tokens = ::llama_tokenize(ctx, prompt, true);

        if (!assistant.empty()) {
            tokens.push_back(llama_token_eos(ctx));
        }

        return tokens;
    }

    void eval_prompt(std::vector<llama_token> prompt) {
        while (!prompt.empty()) {
            int n_tokens = std::min(lparams.n_batch, (int)prompt.size());
            llama_eval(ctx, prompt.data(), n_tokens, llama_get_kv_cache_token_count(ctx), n_threads);
            prompt.erase(prompt.begin(), prompt.begin() + n_tokens);
        }
    }

    llama_token sample_token() {
        auto logits  = llama_get_logits(ctx);
        auto n_vocab = llama_n_vocab(ctx);

        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);

        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
        }

        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

        llama_sample_temperature(ctx, &candidates_p, 0.7f);

        llama_token token = llama_sample_token(ctx , &candidates_p);

        return token;
    }

    void eval_answer() {
        std::string answer;
        do {
            llama_token id = sample_token();
            llama_eval(ctx, &id, 1, llama_get_kv_cache_token_count(ctx), n_threads);

            //printf("[%d]%s", id, llama_token_to_str(ctx, id).c_str());

            if (id == llama_token_eos(ctx)) {
                break;
            }

            printf("%s", llama_token_to_str(ctx, id).c_str());

        } while (true);
    }

    void chat_loop() {
        std::string user_prompt;

        while (true) {
            printf("\nUser: ");
            std::getline(std::cin, user_prompt);
            if (user_prompt.empty())
                break;

            auto prompt = tokenize_dialog(user_prompt);
            eval_prompt(prompt);

            eval_answer();
        }
    }
};

int main(int argc, char ** argv) {
    if (argc == 1 || argv[1][0] == '-') {
        printf("usage: %s MODEL_PATH [\"system prompt\"]", argv[0]);
        return 1 ;
    }

    std::string model_file = argv[1];
    std::string system;

    if (argc > 2) {
        system = argv[2];
    }
    llama_backend_init(false);

    chat c(model_file, system);

    c.chat_loop();
}
