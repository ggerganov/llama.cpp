#if defined(_WIN32)
#    include <windows.h>
#else
#    include <unistd.h>
#endif

#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "llama-cpp.h"

typedef std::unique_ptr<char[]> char_array_ptr;

class Opt {
  public:
    int init_opt(int argc, const char ** argv) {
        construct_help_str_();
        // Parse arguments
        if (parse(argc, argv)) {
            fprintf(stderr, "Error: Failed to parse arguments.\n");
            help();
            return 1;
        }

        // If help is requested, show help and exit
        if (help_) {
            help();
            return 2;
        }

        return 0;  // Success
    }

    std::string model_;
    std::string user_;
    int         context_size_ = 2048, ngl_ = 0;

  private:
    std::string help_str_;
    bool        help_ = false;

    void construct_help_str_() {
        help_str_ =
            "Description:\n"
            "  Runs a llm\n"
            "\n"
            "Usage:\n"
            "  llama-run [options] MODEL [PROMPT]\n"
            "\n"
            "Options:\n"
            "  -c, --context-size <value>\n"
            "      Context size (default: " +
            std::to_string(context_size_);
        help_str_ +=
            ")\n"
            "  -n, --ngl <value>\n"
            "      Number of GPU layers (default: " +
            std::to_string(ngl_);
        help_str_ +=
            ")\n"
            "  -h, --help\n"
            "      Show help message\n"
            "\n"
            "Examples:\n"
            "  llama-run your_model.gguf\n"
            "  llama-run --ngl 99 your_model.gguf\n"
            "  llama-run --ngl 99 your_model.gguf Hello World\n";
    }

    int parse(int argc, const char ** argv) {
        int positional_args_i = 0;
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--context-size") == 0) {
                if (i + 1 >= argc) {
                    return 1;
                }

                context_size_ = std::atoi(argv[++i]);
            } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--ngl") == 0) {
                if (i + 1 >= argc) {
                    return 1;
                }

                ngl_ = std::atoi(argv[++i]);
            } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
                help_ = true;
                return 0;
            } else if (!positional_args_i) {
                ++positional_args_i;
                model_ = std::string(argv[i]);
            } else if (positional_args_i == 1) {
                ++positional_args_i;
                user_ = argv[i];
            } else {
                user_ += " " + std::string(argv[i]);
            }
        }

        return model_.empty();  // model_ is the only required value
    }

    void help() const { printf("%s", help_str_.c_str()); }
};

class LlamaData {
  public:
    llama_model_ptr                 model;
    llama_sampler_ptr               sampler;
    llama_context_ptr               context;
    std::vector<llama_chat_message> messages;
    std::vector<char_array_ptr>     owned_content;
    std::vector<char>               fmtted;

    int init(Opt & opt) {
        model = initialize_model(opt);
        if (!model) {
            return 1;
        }

        context = initialize_context(model, opt.context_size_);
        if (!context) {
            return 1;
        }

        sampler = initialize_sampler();
        return 0;
    }

  private:
    int remove_proto(std::string & model_) {
        const std::string::size_type pos = model_.find("://");
        if (pos == std::string::npos) {
            return 1;
        }

        model_ = model_.substr(pos + 3);  // Skip past "://"
        return 0;
    }

    int huggingface_dl(std::string & model_, const struct llama_model_params & params) {
        // Find the second occurrence of '/' after protocol string
        const size_t start = 0;
        size_t       pos   = model_.find('/', start + 1);
        pos                = model_.find('/', pos + 1);
        if (pos == std::string::npos) {
            return 1;
        }

        const std::string hfr = model_.substr(start, pos - start);
        const std::string hff = model_.substr(pos + 1);
        common_load_model_from_hf(hfr, hff, "", "", params);

        return 0;
    }

    int resolve_model(std::string & model_, const struct llama_model_params & params) {
        if (starts_with(model_, "hf://") || starts_with(model_, "huggingface://")) {
            remove_proto(model_);
            huggingface_dl(model_, params);
        } else if (starts_with(model_, "https://")) {
            common_load_model_from_url(model_, "", "", params);
        } else if (starts_with(model_, "file://")) {
            remove_proto(model_);
        }

        // Also implement ollama://, if file doesn't exist, assume ollama str

        return 0;
    }

    // Initializes the model and returns a unique pointer to it
    llama_model_ptr initialize_model(Opt & opt) {
        ggml_backend_load_all();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers       = opt.ngl_;
        resolve_model(opt.model_, model_params);
        llama_model_ptr model(llama_load_model_from_file(opt.model_.c_str(), model_params));
        if (!model) {
            fprintf(stderr, "%s: error: unable to load model\n", __func__);
        }

        return model;
    }

    // Initializes the context with the specified parameters
    llama_context_ptr initialize_context(const llama_model_ptr & model, const int n_ctx) {
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx                = n_ctx;
        ctx_params.n_batch              = n_ctx;
        llama_context_ptr context(llama_new_context_with_model(model.get(), ctx_params));
        if (!context) {
            fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        }

        return context;
    }

    // Initializes and configures the sampler
    llama_sampler_ptr initialize_sampler() {
        llama_sampler_ptr sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        return sampler;
    }
};

// Add a message to `messages` and store its content in `owned_content`
static void add_message(const char * role, const std::string & text, LlamaData & llama_data) {
    char_array_ptr content(new char[text.size() + 1]);
    std::strcpy(content.get(), text.c_str());
    llama_data.messages.push_back({ role, content.get() });
    llama_data.owned_content.push_back(std::move(content));
}

// Function to apply the chat template and resize `formatted` if needed
static int apply_chat_template(LlamaData & llama_data, const bool append) {
    int result = llama_chat_apply_template(llama_data.model.get(), nullptr, llama_data.messages.data(),
                                           llama_data.messages.size(), append, llama_data.fmtted.data(),
                                           llama_data.fmtted.size());
    if (result > static_cast<int>(llama_data.fmtted.size())) {
        llama_data.fmtted.resize(result);
        result = llama_chat_apply_template(llama_data.model.get(), nullptr, llama_data.messages.data(),
                                           llama_data.messages.size(), append, llama_data.fmtted.data(),
                                           llama_data.fmtted.size());
    }

    return result;
}

// Function to tokenize the prompt
static int tokenize_prompt(const llama_model_ptr & model, const std::string & prompt,
                           std::vector<llama_token> & prompt_tokens) {
    const int n_prompt_tokens = -llama_tokenize(model.get(), prompt.c_str(), prompt.size(), NULL, 0, true, true);
    prompt_tokens.resize(n_prompt_tokens);
    if (llama_tokenize(model.get(), prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true,
                       true) < 0) {
        fprintf(stderr, "failed to tokenize the prompt\n");
        return -1;
    }

    return n_prompt_tokens;
}

// Check if we have enough space in the context to evaluate this batch
static int check_context_size(const llama_context_ptr & ctx, const llama_batch & batch) {
    const int n_ctx      = llama_n_ctx(ctx.get());
    const int n_ctx_used = llama_get_kv_cache_used_cells(ctx.get());
    if (n_ctx_used + batch.n_tokens > n_ctx) {
        printf("\033[0m\n");
        fprintf(stderr, "context size exceeded\n");
        return 1;
    }

    return 0;
}

// convert the token to a string
static int convert_token_to_string(const llama_model_ptr & model, const llama_token token_id, std::string & piece) {
    char buf[256];
    int  n = llama_token_to_piece(model.get(), token_id, buf, sizeof(buf), 0, true);
    if (n < 0) {
        fprintf(stderr, "failed to convert token to piece\n");
        return 1;
    }

    piece = std::string(buf, n);
    return 0;
}

static void print_word_and_concatenate_to_response(const std::string & piece, std::string & response) {
    printf("%s", piece.c_str());
    fflush(stdout);
    response += piece;
}

// helper function to evaluate a prompt and generate a response
static int generate(LlamaData & llama_data, const std::string & prompt, std::string & response) {
    std::vector<llama_token> tokens;
    const int                n_prompt_tokens = tokenize_prompt(llama_data.model, prompt, tokens);
    if (n_prompt_tokens < 0) {
        return 1;
    }

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_token new_token_id;
    while (true) {
        check_context_size(llama_data.context, batch);
        if (llama_decode(llama_data.context.get(), batch)) {
            fprintf(stderr, "failed to decode\n");
            return 1;
        }

        // sample the next token, check is it an end of generation?
        new_token_id = llama_sampler_sample(llama_data.sampler.get(), llama_data.context.get(), -1);
        if (llama_token_is_eog(llama_data.model.get(), new_token_id)) {
            break;
        }

        std::string piece;
        if (convert_token_to_string(llama_data.model, new_token_id, piece)) {
            return 1;
        }

        print_word_and_concatenate_to_response(piece, response);

        // prepare the next batch with the sampled token
        batch = llama_batch_get_one(&new_token_id, 1);
    }

    return 0;
}

static int read_user_input(std::string & user) {
    std::getline(std::cin, user);
    return user.empty();  // Indicate an error or empty input
}

// Function to generate a response based on the prompt
static int generate_response(LlamaData & llama_data, const std::string & prompt, std::string & response) {
    // Set response color
    printf("\033[33m");
    if (generate(llama_data, prompt, response)) {
        fprintf(stderr, "failed to generate response\n");
        return 1;
    }

    // End response with color reset and newline
    printf("\n\033[0m");
    return 0;
}

// Helper function to apply the chat template and handle errors
static int apply_chat_template_with_error_handling(LlamaData & llama_data, const bool append, int & output_length) {
    const int new_len = apply_chat_template(llama_data, append);
    if (new_len < 0) {
        fprintf(stderr, "failed to apply the chat template\n");
        return -1;
    }

    output_length = new_len;
    return 0;
}

// Helper function to handle user input
static bool handle_user_input(std::string & user_input, const std::string & user_) {
    if (!user_.empty()) {
        user_input = user_;
        return true;  // No need for interactive input
    }

    printf("\033[32m> \033[0m");
    return !read_user_input(user_input);  // Returns false if input ends the loop
}

// Function to tokenize the prompt
static int chat_loop(LlamaData & llama_data, std::string & user_) {
    int                         prev_len = 0;
    llama_data.fmtted.resize(llama_n_ctx(llama_data.context.get()));
    while (true) {
        // Get user input
        std::string user_input;
        if (!handle_user_input(user_input, user_)) {
            break;
        }

        add_message("user", user_.empty() ? user_input : user_, llama_data);
        int new_len;
        if (apply_chat_template_with_error_handling(llama_data, true, new_len) < 0) {
            return 1;
        }

        std::string prompt(llama_data.fmtted.begin() + prev_len, llama_data.fmtted.begin() + new_len);
        std::string response;
        if (generate_response(llama_data, prompt, response)) {
            return 1;
        }

        if (!user_.empty()) {
            break;
        }

        add_message("assistant", response, llama_data);
        if (apply_chat_template_with_error_handling(llama_data, false, prev_len) < 0) {
            return 1;
        }
    }

    return 0;
}

static void log_callback(const enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    }
}

static bool is_stdin_a_terminal() {
#if defined(_WIN32)
    HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
    DWORD  mode;
    return GetConsoleMode(hStdin, &mode);
#else
    return isatty(STDIN_FILENO);
#endif
}

static std::string read_pipe_data() {
    std::ostringstream result;
    result << std::cin.rdbuf();  // Read all data from std::cin
    return result.str();
}

int main(int argc, const char ** argv) {
    Opt       opt;
    const int opt_ret = opt.init_opt(argc, argv);
    if (opt_ret == 2) {
        return 0;
    } else if (opt_ret) {
        return 1;
    }

    if (!is_stdin_a_terminal()) {
        if (!opt.user_.empty()) {
            opt.user_ += "\n\n";
        }

        opt.user_ += read_pipe_data();
    }

    llama_log_set(log_callback, nullptr);
    LlamaData llama_data;
    if (llama_data.init(opt)) {
        return 1;
    }

    if (chat_loop(llama_data, opt.user_)) {
        return 1;
    }

    return 0;
}
