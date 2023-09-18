// A basic application simulating a server with multiple clients.
// The clients submite requests to the server and they are processed in parallel.

#include "build-info.h"

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// trim whitespace from the beginning and end of a string
static std::string trim(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();

    while (start < end && isspace(str[start])) {
        start += 1;
    }

    while (end > start && isspace(str[end - 1])) {
        end -= 1;
    }

    return str.substr(start, end - start);
}

static std::string k_system = R"(
Transcript of a dialog, where the User interacts with an Assistant.
The Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User: Hello, what is the temperature outside?
Assistant: It is 72 degrees Fahrenheit.
User: What is the definition of a prime number?
Assistant: A prime number is a number that is divisible only by itself and 1.
User: )";

static std::vector<std::string> k_prompts = {
    "What is the meaning of life?",
    "What is the population of Europe?",
    "List all planets in the Solar System.",
    "What is the capital of France?",
    "Tell me an interesting fact about llamas.",
    "What is the best way to cook a steak?",
    "Are you familiar with the Special Theory of Relativity and can you explain it to me?",
    "Recommend some interesting books to read.",
    "What is the best way to learn a new language?",
    "How to get a job at Google?",
    "If you could have any superpower, what would it be?",
    "I want to learn how to play the piano.",
};

struct client {
    int32_t id = 0;

    llama_seq_id seq_id = -1;

    llama_token sampled;

    int32_t n_prompt  = 0;
    int32_t n_decoded = 0;
    int32_t i_batch   = -1;

    std::string input;
    std::string prompt;
    std::string response;

    std::vector<llama_token> last_tokens;
};

int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    const int n_clients = 16;

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("parallel", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
#endif // LOG_DISABLE_LOGS

    // init llama.cpp
    llama_backend_init(params.numa);

    llama_model * model = NULL;

    llama_context * ctx = NULL;

    // load the target model
    params.logits_all = true;
    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    fprintf(stderr, "\n\n");
    fflush(stderr);

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_vocab = llama_n_vocab(ctx);

    std::vector<client> clients(n_clients);
    for (size_t i = 0; i < clients.size(); ++i) {
        auto & client = clients[i];
        client.id = i;
        client.last_tokens.resize(n_ctx);
        std::fill(client.last_tokens.begin(), client.last_tokens.end(), 0);
    }

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);

    auto t_main_start = ggml_time_us();

    int64_t n_tokens_total = 0;

    llama_seq_id g_seq_id = 0;

    std::vector<llama_token>  batch_token;
    std::vector<llama_pos>    batch_pos;
    std::vector<llama_seq_id> batch_seq_id;
    std::vector<client *>     batch_clients;

    while (true) {
        uint32_t n_tokens = 0;

        batch_token.clear();
        batch_pos.clear();
        batch_seq_id.clear();

        for (auto & client : clients) {
            if (client.seq_id == -1) {
                client.seq_id = g_seq_id;
                client.input = k_prompts[rand() % k_prompts.size()];
                client.prompt = k_system + client.input + "\nAssistant:";
                client.response = "";
                std::fill(client.last_tokens.begin(), client.last_tokens.end(), 0);

                std::vector<llama_token> prompt_tokens;
                prompt_tokens = ::llama_tokenize(ctx, client.prompt, true);

                for (size_t i = 0; i < prompt_tokens.size(); ++i) {
                    batch_token.push_back(prompt_tokens[i]);
                    batch_pos.push_back(i);
                    batch_seq_id.push_back(client.seq_id);
                    batch_clients.push_back(&client);
                }
                client.n_prompt  = prompt_tokens.size();
                client.n_decoded = prompt_tokens.size();
                client.i_batch   = batch_token.size() - 1;

                g_seq_id += 1;
            } else {
                batch_token.push_back(client.sampled);
                batch_pos.push_back(client.n_decoded);
                batch_seq_id.push_back(client.seq_id);
                batch_clients.push_back(&client);
                client.n_decoded += 1;
                client.i_batch = batch_token.size() - 1;
            }
        }

        // process in chunks of params.n_batch
        for (size_t i = 0; i < batch_token.size(); i += params.n_batch) {
            n_tokens = std::min(params.n_batch, (int32_t) (batch_token.size() - i));

            llama_batch batch = {
                n_tokens,
                batch_token.data() + i,
                nullptr,
                batch_pos.data() + i,
                batch_seq_id.data() + i,
                0, 0, 0, // unused
            };

            if (llama_decode(ctx, batch, params.n_threads)) {
                LOG_TEE("%s : failed to decode batch\n", __func__);
                return 1;
            }

            for (auto & client : clients) {
                if (client.i_batch < (int) i || client.i_batch >= (int) (i + n_tokens)) {
                    continue;
                }

                const llama_token id = llama_sample_token(ctx, NULL, NULL, params, client.last_tokens, candidates, client.i_batch - i);

                // remember which tokens were sampled - used for repetition penalties during sampling
                client.last_tokens.erase(client.last_tokens.begin());
                client.last_tokens.push_back(id);

                const std::string token_str = llama_token_to_piece(ctx, id);
                client.response += token_str;
                client.sampled = id;

                //printf("client %d, seq %d, token %d, pos %d, batch %d: %s\n",
                //        client.id, client.seq_id, id, client.n_decoded, client.i_batch, token_str.c_str());

                if (id == llama_token_eos(ctx) || client.n_decoded > params.n_predict || client.response.find("User:") != std::string::npos) {
                    const size_t pos = client.response.find("User:");
                    if (pos != std::string::npos) {
                        client.response = client.response.substr(0, pos);
                    }

                    llama_kv_cache_rm_seq(ctx, client.seq_id, 0, n_ctx);

                    const auto t_main_end = ggml_time_us();

                    n_tokens_total += client.n_decoded - client.n_prompt;

                    printf("\033[1mClient %d, seq %d, prompt %d t, response %d t, speed: %.2f t/s\033[0m: \n\nInput:    %s\nResponse: %s\n\n",
                            client.id, client.seq_id, client.n_prompt, client.n_decoded - client.n_prompt,
                            (double) n_tokens_total / (t_main_end - t_main_start) * 1e6,
                            client.input.c_str(), ::trim(client.response).c_str());

                    client.seq_id = -1;
                }
            }
        }

        static bool is_first = true;
        if (is_first) {
            t_main_start = ggml_time_us();
            n_tokens_total = 0;
            is_first = false;
        }
    }

    LOG_TEE("\n\n");

    llama_print_timings(ctx);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
