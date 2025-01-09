#include "opt.h"

#include <cstdarg>

int printe(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    const int ret = vfprintf(stderr, fmt, args);
    va_end(args);

    return ret;
}

int Opt::init(int argc, const char ** argv) {
    ctx_params           = llama_context_default_params();
    model_params         = llama_model_default_params();
    context_size_default = ctx_params.n_batch;
    ngl_default          = model_params.n_gpu_layers;
    common_params_sampling sampling;
    temperature_default = sampling.temp;

    if (argc < 2) {
        printe("Error: No arguments provided.\n");
        print_help();
        return 1;
    }

    // Parse arguments
    if (parse(argc, argv)) {
        printe("Error: Failed to parse arguments.\n");
        print_help();
        return 1;
    }

    // If help is requested, show help and exit
    if (help) {
        print_help();
        return 2;
    }

    ctx_params.n_batch        = context_size >= 0 ? context_size : context_size_default;
    ctx_params.n_ctx          = ctx_params.n_batch;
    model_params.n_gpu_layers = ngl >= 0 ? ngl : ngl_default;
    temperature               = temperature >= 0 ? temperature : temperature_default;

    return 0;  // Success
}

bool Opt::parse_flag(const char ** argv, int i, const char * short_opt, const char * long_opt) {
    return strcmp(argv[i], short_opt) == 0 || strcmp(argv[i], long_opt) == 0;
}

int Opt::handle_option_with_value(int argc, const char ** argv, int & i, int & option_value) {
    if (i + 1 >= argc) {
        return 1;
    }

    option_value = std::atoi(argv[++i]);

    return 0;
}

int Opt::handle_option_with_value(int argc, const char ** argv, int & i, float & option_value) {
    if (i + 1 >= argc) {
        return 1;
    }

    option_value = std::atof(argv[++i]);

    return 0;
}

int Opt::parse(int argc, const char ** argv) {
    bool options_parsing = true;
    for (int i = 1, positional_args_i = 0; i < argc; ++i) {
        if (options_parsing && (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--context-size") == 0)) {
            if (handle_option_with_value(argc, argv, i, context_size) == 1) {
                return 1;
            }
        } else if (options_parsing && (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--ngl") == 0)) {
            if (handle_option_with_value(argc, argv, i, ngl) == 1) {
                return 1;
            }
        } else if (options_parsing && strcmp(argv[i], "--temp") == 0) {
            if (handle_option_with_value(argc, argv, i, temperature) == 1) {
                return 1;
            }
        } else if (options_parsing &&
                   (parse_flag(argv, i, "-v", "--verbose") || parse_flag(argv, i, "-v", "--log-verbose"))) {
            verbose = true;
        } else if (options_parsing && parse_flag(argv, i, "-h", "--help")) {
            help = true;
            return 0;
        } else if (options_parsing && strcmp(argv[i], "--") == 0) {
            options_parsing = false;
        } else if (positional_args_i == 0) {
            if (!argv[i][0] || argv[i][0] == '-') {
                return 1;
            }

            ++positional_args_i;
            model_ = argv[i];
        } else if (positional_args_i == 1) {
            ++positional_args_i;
            user = argv[i];
        } else {
            user += " " + std::string(argv[i]);
        }
    }

    return 0;
}

void Opt::print_help() const {
    printf(
        "Description:\n"
        "  Runs a llm\n"
        "\n"
        "Usage:\n"
        "  llama-run [options] model [prompt]\n"
        "\n"
        "Options:\n"
        "  -c, --context-size <value>\n"
        "      Context size (default: %d)\n"
        "  -n, --ngl <value>\n"
        "      Number of GPU layers (default: %d)\n"
        "  --temp <value>\n"
        "      Temperature (default: %.1f)\n"
        "  -v, --verbose, --log-verbose\n"
        "      Set verbosity level to infinity (i.e. log all messages, useful for debugging)\n"
        "  -h, --help\n"
        "      Show help message\n"
        "\n"
        "Commands:\n"
        "  model\n"
        "      Model is a string with an optional prefix of \n"
        "      huggingface:// (hf://), ollama://, https:// or file://.\n"
        "      If no protocol is specified and a file exists in the specified\n"
        "      path, file:// is assumed, otherwise if a file does not exist in\n"
        "      the specified path, ollama:// is assumed. Models that are being\n"
        "      pulled are downloaded with .partial extension while being\n"
        "      downloaded and then renamed as the file without the .partial\n"
        "      extension when complete.\n"
        "\n"
        "Examples:\n"
        "  llama-run llama3\n"
        "  llama-run ollama://granite-code\n"
        "  llama-run ollama://smollm:135m\n"
        "  llama-run hf://QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q2_K.gguf\n"
        "  llama-run "
        "huggingface://bartowski/SmolLM-1.7B-Instruct-v0.2-GGUF/SmolLM-1.7B-Instruct-v0.2-IQ3_M.gguf\n"
        "  llama-run https://example.com/some-file1.gguf\n"
        "  llama-run some-file2.gguf\n"
        "  llama-run file://some-file3.gguf\n"
        "  llama-run --ngl 999 some-file4.gguf\n"
        "  llama-run --ngl 999 some-file5.gguf Hello World\n",
        context_size_default, ngl_default, temperature_default);
}
