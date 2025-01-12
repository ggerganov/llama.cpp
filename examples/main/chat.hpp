#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include <fstream>

struct llama_cli_chat {
    struct llama_context     * ctx;
    const struct llama_model * model;
    const struct llama_vocab * vocab;
    struct common_sampler    * smpl;
    struct common_params       params;

    bool interacting = false;
    std::vector<common_chat_msg> chat_msgs;
    std::ostringstream pending_input;

    struct llama_batch batch;
    llama_tokens cache_tokens;
    int n_past = 0;

    llama_cli_chat(
            struct common_params & params,
            struct llama_context * ctx,
            struct common_sampler * smpl) : ctx(ctx), smpl(smpl), params(params) {
        model = llama_get_model(ctx);
        vocab = llama_model_get_vocab(model);
        batch = llama_batch_init(params.n_batch, 0, 1);
    }

    void decode(llama_tokens & eval_tokens, bool is_generating) {
        if (is_generating) {
            GGML_ASSERT(eval_tokens.size() == 1);
        } else {
            n_past = common_lcp(cache_tokens, eval_tokens);
            // in case we do a re-generation, we need to prevent eval_tokens from being empty
            if ((int) eval_tokens.size() == n_past) {
                n_past--;
            }
            if (n_past > 0) {
                eval_tokens.erase(eval_tokens.begin(), eval_tokens.begin() + n_past);
                cache_tokens.erase(cache_tokens.begin() + n_past, cache_tokens.end());
                LOG_DBG("remove from cache [%d, inf)\n", n_past);
                LOG_DBG("in cache: %s\n", common_detokenize(ctx, cache_tokens, true).c_str());
                LOG_DBG("to decode %d tokens\n", (int) eval_tokens.size());
                llama_kv_cache_seq_rm(ctx, 0, n_past, -1);
            }
        }

        // decode
        for (size_t i = 0; i < eval_tokens.size(); i += params.n_batch) {
            if (interacting) {
                break;
            }

            common_batch_clear(batch);
            for (int j = 0; j < params.n_batch && i + j < eval_tokens.size(); ++j) {
                n_past++;
                bool is_last_token = i + j == eval_tokens.size() - 1;
                common_batch_add(batch, eval_tokens[i + j], n_past, {0}, is_last_token);
            }

            if (llama_decode(ctx, batch)) {
                GGML_ABORT("failed to decode\n");
            }
        }

        // update cache tokens
        if (is_generating) {
            cache_tokens.push_back(eval_tokens[0]);
        } else {
            cache_tokens.insert(cache_tokens.end(), eval_tokens.begin(), eval_tokens.end());
        }
    }

    [[noreturn]] void run() {
        while (true) {
            interacting = true;
            LOG("\n> ");

            // color user input only
            console::set_display(console::user_input);
            std::string line;
            bool another_line = true;
            bool continue_input = false;
            do {
                another_line = console::readline(line, params.multiline_input);
                if (handle_command(line, continue_input)) {
                    continue; // do not add this line to pending_input
                }
                pending_input << line;
            } while (another_line);

            if (continue_input) {
                continue;
            }

            if (pending_input.tellp() == 0) {
                LOG_DBG("empty line, passing control back\n");
                continue;
            }

            // done taking input, reset color
            console::set_display(console::reset);
            interacting = false;

            // add message and format chat
            if (!chat_msgs.empty() && chat_msgs.back().role == "user") {
                chat_msgs.pop_back();
            }
            chat_msgs.push_back({"user", string_strip(pending_input.str())});
            pending_input.str(""); // clear
            auto formatted = common_chat_apply_template(model, params.chat_template, chat_msgs, true);

            // tokenize the new chat history and decode
            llama_tokens prompt_tokens = common_tokenize(ctx, formatted, true, true);
            decode(prompt_tokens, false);

            // generate response
            llama_token new_token_id = LLAMA_TOKEN_NULL;
            llama_tokens generated_tokens;
            common_sampler_reset(smpl);
            while (true) {
                if (interacting) {
                    break;
                }

                // sample the next token
                new_token_id = common_sampler_sample(smpl, ctx, -1);

                // is it an end of generation?
                if (llama_vocab_is_eog(vocab, new_token_id)) {
                    break;
                }

                // print the token, then decode it
                printf("%s", common_token_to_piece(ctx, new_token_id, params.special).c_str());
                fflush(stdout);
                generated_tokens.push_back(new_token_id);
                llama_tokens new_tok = {new_token_id};
                decode(new_tok, true);
            }

            // add the generated tokens to the chat history
            std::string response = common_detokenize(ctx, generated_tokens, true);
            chat_msgs.push_back({"assistant", response});

            // print a new line if needed
            if (!response.empty() && response.back() != '\n') {
                printf("\n");
            }
        }
    }

    void interrupt() {
        if (interacting) {
            // exit
            printf("\n");
            console::cleanup();
            common_perf_print(ctx, smpl);
            common_log_pause(common_log_main());
            exit(0);
        }
        interacting = true;
    }

    bool handle_command(std::string & inp, bool & continue_input) {
        if (inp.empty() || inp[0] != '/') {
            return false; // not a command
        }
        auto parts = string_split<std::string>(string_strip(inp), ' ');
        std::string & cmd = parts[0];
        if (cmd == "/help") {
            LOG("TODO\n");
            continue_input = true;
        } else if (cmd == "/history") {
            display_history();
            continue_input = true;
        } else if (cmd == "/regen") {
            if (chat_msgs.empty()) {
                LOG_ERR("no chat history to regenerate\n");
                continue_input = true;
                return true;
            }
            if (chat_msgs.back().role == "assistant") {
                chat_msgs.pop_back();
            }
            if (chat_msgs.back().role == "user") {
                pending_input.str(""); // clear
                pending_input << chat_msgs.back().content;
                chat_msgs.pop_back();
            }
            continue_input = false;
        } else if (cmd == "/readfile") {
            const std::string filename = parts[1];
            LOG_DBG("reading file: '%s'\n", filename.c_str());
            std::ifstream text_file(filename);
            if (!text_file) {
                LOG("failed to open file '%s'\n", filename.c_str());
            } else {
                pending_input << text_file.rdbuf() << "\n\n";
                LOG("read %zu characters from file\n", (size_t) text_file.tellg());
            }
            continue_input = true;
        } else {
            LOG_ERR("unknown command: %s\n", cmd.c_str());
            continue_input = true;
        }
        return true;
    }

    void display_history() {
        for (const auto & msg : chat_msgs) {
            LOG("%s: %s\n\n", msg.role.c_str(), msg.content.c_str());
        }
    }

    ~llama_cli_chat() {
        llama_batch_free(batch);
    }
};
