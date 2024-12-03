#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <climits>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "llama-cpp.h"

typedef std::unique_ptr<char[]> char_array_ptr;

struct Argument {
    std::string flag;
    std::string help_text;
};

struct Options {
    std::string model_path, prompt_non_interactive;
    int ngl = 99;
    int n_ctx = 2048;
};

class ArgumentParser {
   public:
    ArgumentParser(const char * program_name) : program_name(program_name) {}

    void add_argument(const std::string & flag, std::string & var, const std::string & help_text = "") {
        string_args[flag] = &var;
        arguments.push_back({flag, help_text});
    }

    void add_argument(const std::string & flag, int & var, const std::string & help_text = "") {
        int_args[flag] = &var;
        arguments.push_back({flag, help_text});
    }

    int parse(int argc, const char ** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (string_args.count(arg)) {
                if (i + 1 < argc) {
                    *string_args[arg] = argv[++i];
                } else {
                    fprintf(stderr, "error: missing value for %s\n", arg.c_str());
                    print_usage();
                    return 1;
                }
            } else if (int_args.count(arg)) {
                if (i + 1 < argc) {
                    if (parse_int_arg(argv[++i], *int_args[arg]) != 0) {
                        fprintf(stderr, "error: invalid value for %s: %s\n", arg.c_str(), argv[i]);
                        print_usage();
                        return 1;
                    }
                } else {
                    fprintf(stderr, "error: missing value for %s\n", arg.c_str());
                    print_usage();
                    return 1;
                }
            } else {
                fprintf(stderr, "error: unrecognized argument %s\n", arg.c_str());
                print_usage();
                return 1;
            }
        }

        if (string_args["-m"]->empty()) {
            fprintf(stderr, "error: -m is required\n");
            print_usage();
            return 1;
        }

        return 0;
    }

   private:
    const char * program_name;
    std::unordered_map<std::string, std::string *> string_args;
    std::unordered_map<std::string, int *> int_args;
    std::vector<Argument> arguments;

    int parse_int_arg(const char * arg, int & value) {
        char * end;
        const long val = std::strtol(arg, &end, 10);
        if (*end == '\0' && val >= INT_MIN && val <= INT_MAX) {
            value = static_cast<int>(val);
            return 0;
        }
        return 1;
    }

    void print_usage() const {
        printf("\nUsage:\n");
        printf("  %s [OPTIONS]\n\n", program_name);
        printf("Options:\n");
        for (const auto & arg : arguments) {
            printf("  %-10s %s\n", arg.flag.c_str(), arg.help_text.c_str());
        }

        printf("\n");
    }
};

class LlamaData {
   public:
    llama_model_ptr model;
    llama_sampler_ptr sampler;
    llama_context_ptr context;
    std::vector<llama_chat_message> messages;

    int init(const Options & opt) {
        model = initialize_model(opt.model_path, opt.ngl);
        if (!model) {
            return 1;
        }

        context = initialize_context(model, opt.n_ctx);
        if (!context) {
            return 1;
        }

        sampler = initialize_sampler();
        return 0;
    }

   private:
    // Initializes the model and returns a unique pointer to it
    llama_model_ptr initialize_model(const std::string & model_path, const int ngl) {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = ngl;

        llama_model_ptr model(llama_load_model_from_file(model_path.c_str(), model_params));
        if (!model) {
            fprintf(stderr, "%s: error: unable to load model\n", __func__);
        }

        return model;
    }

    // Initializes the context with the specified parameters
    llama_context_ptr initialize_context(const llama_model_ptr & model, const int n_ctx) {
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = n_ctx;

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
static void add_message(const char * role, const std::string & text, LlamaData & llama_data,
                        std::vector<char_array_ptr> & owned_content) {
    char_array_ptr content(new char[text.size() + 1]);
    std::strcpy(content.get(), text.c_str());
    llama_data.messages.push_back({role, content.get()});
    owned_content.push_back(std::move(content));
}

// Function to apply the chat template and resize `formatted` if needed
static int apply_chat_template(const LlamaData & llama_data, std::vector<char> & formatted, const bool append) {
    int result = llama_chat_apply_template(llama_data.model.get(), nullptr, llama_data.messages.data(),
                                           llama_data.messages.size(), append, formatted.data(), formatted.size());
    if (result > static_cast<int>(formatted.size())) {
        formatted.resize(result);
        result = llama_chat_apply_template(llama_data.model.get(), nullptr, llama_data.messages.data(),
                                           llama_data.messages.size(), append, formatted.data(), formatted.size());
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
        GGML_ABORT("failed to tokenize the prompt\n");
    }

    return n_prompt_tokens;
}

// Check if we have enough space in the context to evaluate this batch
static int check_context_size(const llama_context_ptr & ctx, const llama_batch & batch) {
    const int n_ctx = llama_n_ctx(ctx.get());
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
    int n = llama_token_to_piece(model.get(), token_id, buf, sizeof(buf), 0, true);
    if (n < 0) {
        GGML_ABORT("failed to convert token to piece\n");
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
    std::vector<llama_token> prompt_tokens;
    const int n_prompt_tokens = tokenize_prompt(llama_data.model, prompt, prompt_tokens);
    if (n_prompt_tokens < 0) {
        return 1;
    }

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token new_token_id;
    while (true) {
        check_context_size(llama_data.context, batch);
        if (llama_decode(llama_data.context.get(), batch)) {
            GGML_ABORT("failed to decode\n");
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

static int parse_arguments(const int argc, const char ** argv, Options & opt) {
    ArgumentParser parser(argv[0]);
    parser.add_argument("-m", opt.model_path, "model");
    parser.add_argument("-p", opt.prompt_non_interactive, "prompt");
    parser.add_argument("-c", opt.n_ctx, "context_size");
    parser.add_argument("-ngl", opt.ngl, "n_gpu_layers");
    if (parser.parse(argc, argv)) {
        return 1;
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
static int apply_chat_template_with_error_handling(const LlamaData & llama_data, std::vector<char> & formatted,
                                                   const bool is_user_input, int & output_length) {
    const int new_len = apply_chat_template(llama_data, formatted, is_user_input);
    if (new_len < 0) {
        fprintf(stderr, "failed to apply the chat template\n");
        return -1;
    }

    output_length = new_len;
    return 0;
}

// Helper function to handle user input
static bool handle_user_input(std::string & user_input, const std::string & prompt_non_interactive) {
    if (!prompt_non_interactive.empty()) {
        user_input = prompt_non_interactive;
        return true;  // No need for interactive input
    }

    printf("\033[32m> \033[0m");
    return !read_user_input(user_input);  // Returns false if input ends the loop
}

// Function to tokenize the prompt
static int chat_loop(LlamaData & llama_data, std::string & prompt_non_interactive) {
    std::vector<char_array_ptr> owned_content;
    std::vector<char> fmtted(llama_n_ctx(llama_data.context.get()));
    int prev_len = 0;

    while (true) {
        // Get user input
        std::string user_input;
        if (!handle_user_input(user_input, prompt_non_interactive)) {
            break;
        }

        add_message("user", prompt_non_interactive.empty() ? user_input : prompt_non_interactive, llama_data,
                    owned_content);

        int new_len;
        if (apply_chat_template_with_error_handling(llama_data, fmtted, true, new_len) < 0) {
            return 1;
        }

        std::string prompt(fmtted.begin() + prev_len, fmtted.begin() + new_len);
        std::string response;
        if (generate_response(llama_data, prompt, response)) {
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
    DWORD mode;
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
    Options opt;
    if (parse_arguments(argc, argv, opt)) {
        return 1;
    }

    if (!is_stdin_a_terminal()) {
        if (!opt.prompt_non_interactive.empty()) {
            opt.prompt_non_interactive += "\n\n";
        }

        opt.prompt_non_interactive += read_pipe_data();
    }

    llama_log_set(log_callback, nullptr);
    LlamaData llama_data;
    if (llama_data.init(opt)) {
        return 1;
    }

    if (chat_loop(llama_data, opt.prompt_non_interactive)) {
        return 1;
    }

    return 0;
}
