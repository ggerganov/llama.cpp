#include "common.h"

#include <cassert>
#include <iostream>
#include <cstring>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <sstream>
#include <unordered_set>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#include <wchar.h>
#endif

int32_t get_num_physical_cores() {
#ifdef __linux__
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (siblings.size() > 0) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#elif defined(_WIN32)
    //TODO: Implement
#endif
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

void process_escapes(std::string& input) {
    std::size_t input_len = input.length();
    std::size_t output_idx = 0;

    for (std::size_t input_idx = 0; input_idx < input_len; ++input_idx) {
        if (input[input_idx] == '\\' && input_idx + 1 < input_len) {
            switch (input[++input_idx]) {
                case 'n':  input[output_idx++] = '\n'; break;
                case 'r':  input[output_idx++] = '\r'; break;
                case 't':  input[output_idx++] = '\t'; break;
                case '\'': input[output_idx++] = '\''; break;
                case '\"': input[output_idx++] = '\"'; break;
                case '\\': input[output_idx++] = '\\'; break;
                default:   input[output_idx++] = '\\';
                           input[output_idx++] = input[input_idx]; break;
            }
        } else {
            input[output_idx++] = input[input_idx];
        }
    }

    input.resize(output_idx);
}

bool gpt_params_parse(int argc, char ** argv, gpt_params & params) {
    bool invalid_param = false;
    bool escape_prompt = false;
    std::string arg;
    gpt_params default_params;
    const std::string arg_prefix = "--";

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "-s" || arg == "--seed") {
#if defined(GGML_USE_CUBLAS)
            fprintf(stderr, "WARNING: when using cuBLAS generation results are NOT guaranteed to be reproducible.\n");
#endif
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.seed = std::stoi(argv[i]);
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_threads = std::stoi(argv[i]);
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.prompt = argv[i];
        } else if (arg == "-e") {
            escape_prompt = true;
        } else if (arg == "--prompt-cache") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.path_prompt_cache = argv[i];
        } else if (arg == "--prompt-cache-all") {
            params.prompt_cache_all = true;
        } else if (arg == "-f" || arg == "--file") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::ifstream file(argv[i]);
            if (!file) {
                fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
                invalid_param = true;
                break;
            }
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
            if (params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        } else if (arg == "-n" || arg == "--n-predict") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_predict = std::stoi(argv[i]);
        } else if (arg == "--top-k") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.top_k = std::stoi(argv[i]);
        } else if (arg == "-c" || arg == "--ctx-size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_ctx = std::stoi(argv[i]);
        } else if (arg == "--memory-f32") {
            params.memory_f16 = false;
        } else if (arg == "--top-p") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.top_p = std::stof(argv[i]);
        } else if (arg == "--temp") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.temp = std::stof(argv[i]);
        } else if (arg == "--tfs") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.tfs_z = std::stof(argv[i]);
        } else if (arg == "--typical") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.typical_p = std::stof(argv[i]);
        } else if (arg == "--repeat-last-n") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.repeat_last_n = std::stoi(argv[i]);
        } else if (arg == "--repeat-penalty") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.repeat_penalty = std::stof(argv[i]);
        } else if (arg == "--frequency-penalty") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.frequency_penalty = std::stof(argv[i]);
        } else if (arg == "--presence-penalty") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.presence_penalty = std::stof(argv[i]);
        } else if (arg == "--mirostat") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.mirostat = std::stoi(argv[i]);
        } else if (arg == "--mirostat-lr") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.mirostat_eta = std::stof(argv[i]);
        } else if (arg == "--mirostat-ent") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.mirostat_tau = std::stof(argv[i]);
        } else if (arg == "-b" || arg == "--batch-size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_batch = std::stoi(argv[i]);
            params.n_batch = std::min(512, params.n_batch);
        } else if (arg == "--keep") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_keep = std::stoi(argv[i]);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model = argv[i];
        } else if (arg == "--lora") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.lora_adapter = argv[i];
            params.use_mmap = false;
        } else if (arg == "--lora-base") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.lora_base = argv[i];
        } else if (arg == "-i" || arg == "--interactive") {
            params.interactive = true;
        } else if (arg == "--embedding") {
            params.embedding = true;
        } else if (arg == "--interactive-first") {
            params.interactive_first = true;
        } else if (arg == "-ins" || arg == "--instruct") {
            params.instruct = true;
        } else if (arg == "--multiline-input") {
            params.multiline_input = true;
        } else if (arg == "--color") {
            params.use_color = true;
        } else if (arg == "--mlock") {
            params.use_mlock = true;
        } else if (arg == "--gpu-layers" || arg == "-ngl" || arg == "--n-gpu-layers") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_gpu_layers = std::stoi(argv[i]);
        } else if (arg == "--no-mmap") {
            params.use_mmap = false;
        } else if (arg == "--mtest") {
            params.mem_test = true;
        } else if (arg == "--verbose-prompt") {
            params.verbose_prompt = true;
        } else if (arg == "-r" || arg == "--reverse-prompt") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.antiprompt.push_back(argv[i]);
        } else if (arg == "--perplexity") {
            params.perplexity = true;
        } else if (arg == "--ignore-eos") {
            params.logit_bias[llama_token_eos()] = -INFINITY;
        } else if (arg == "--no-penalize-nl") {
            params.penalize_nl = false;
        } else if (arg == "-l" || arg == "--logit-bias") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::stringstream ss(argv[i]);
            llama_token key;
            char sign;
            std::string value_str;
            try {
                if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
                    params.logit_bias[key] = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
                } else {
                    throw std::exception();
                }
            } catch (const std::exception &e) {
                invalid_param = true;
                break;
            }
        } else if (arg == "--n-parts") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_parts = std::stoi(argv[i]);
        } else if (arg == "-h" || arg == "--help") {
            gpt_print_usage(argc, argv, default_params);
            exit(0);
        } else if (arg == "--random-prompt") {
            params.random_prompt = true;
        } else if (arg == "--in-prefix") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.input_prefix = argv[i];
        } else if (arg == "--in-suffix") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.input_suffix = argv[i];
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            gpt_print_usage(argc, argv, default_params);
            exit(1);
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        gpt_print_usage(argc, argv, default_params);
        exit(1);
    }
    if (params.prompt_cache_all &&
            (params.interactive || params.interactive_first ||
             params.instruct || params.antiprompt.size())) {
        fprintf(stderr, "error: --prompt-cache-all not supported in interactive mode yet\n");
        gpt_print_usage(argc, argv, default_params);
        exit(1);
    }
    if (escape_prompt) {
        process_escapes(params.prompt);
    }

    return true;
}

void gpt_print_usage(int /*argc*/, char ** argv, const gpt_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -i, --interactive     run in interactive mode\n");
    fprintf(stderr, "  --interactive-first   run in interactive mode and wait for input right away\n");
    fprintf(stderr, "  -ins, --instruct      run in instruction mode (use with Alpaca models)\n");
    fprintf(stderr, "  --multiline-input     allows you to write or paste multiple lines without ending each in '\\'\n");
    fprintf(stderr, "  -r PROMPT, --reverse-prompt PROMPT\n");
    fprintf(stderr, "                        run in interactive mode and poll user input upon seeing PROMPT (can be\n");
    fprintf(stderr, "                        specified more than once for multiple prompts).\n");
    fprintf(stderr, "  --color               colorise output to distinguish prompt and user input from generations\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for < 0)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: empty)\n");
    fprintf(stderr, "  -e                    process prompt escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\)\n");
    fprintf(stderr, "  --prompt-cache FNAME  file to cache prompt state for faster startup (default: none)\n");
    fprintf(stderr, "  --prompt-cache-all    if specified, saves user input and generations to cache as well.\n");
    fprintf(stderr, "                        not supported with --interactive or other interactive options\n");
    fprintf(stderr, "  --random-prompt       start with a randomized prompt.\n");
    fprintf(stderr, "  --in-prefix STRING    string to prefix user inputs with (default: empty)\n");
    fprintf(stderr, "  --in-suffix STRING    string to suffix after user inputs with (default: empty)\n");
    fprintf(stderr, "  -f FNAME, --file FNAME\n");
    fprintf(stderr, "                        prompt file to start generation.\n");
    fprintf(stderr, "  -n N, --n-predict N   number of tokens to predict (default: %d, -1 = infinity)\n", params.n_predict);
    fprintf(stderr, "  --top-k N             top-k sampling (default: %d, 0 = disabled)\n", params.top_k);
    fprintf(stderr, "  --top-p N             top-p sampling (default: %.1f, 1.0 = disabled)\n", (double)params.top_p);
    fprintf(stderr, "  --tfs N               tail free sampling, parameter z (default: %.1f, 1.0 = disabled)\n", (double)params.tfs_z);
    fprintf(stderr, "  --typical N           locally typical sampling, parameter p (default: %.1f, 1.0 = disabled)\n", (double)params.typical_p);
    fprintf(stderr, "  --repeat-last-n N     last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)\n", params.repeat_last_n);
    fprintf(stderr, "  --repeat-penalty N    penalize repeat sequence of tokens (default: %.1f, 1.0 = disabled)\n", (double)params.repeat_penalty);
    fprintf(stderr, "  --presence-penalty N  repeat alpha presence penalty (default: %.1f, 0.0 = disabled)\n", (double)params.presence_penalty);
    fprintf(stderr, "  --frequency-penalty N repeat alpha frequency penalty (default: %.1f, 0.0 = disabled)\n", (double)params.frequency_penalty);
    fprintf(stderr, "  --mirostat N          use Mirostat sampling.\n");
    fprintf(stderr, "                        Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.\n");
    fprintf(stderr, "                        (default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)\n", params.mirostat);
    fprintf(stderr, "  --mirostat-lr N       Mirostat learning rate, parameter eta (default: %.1f)\n", (double)params.mirostat_eta);
    fprintf(stderr, "  --mirostat-ent N      Mirostat target entropy, parameter tau (default: %.1f)\n", (double)params.mirostat_tau);
    fprintf(stderr, "  -l TOKEN_ID(+/-)BIAS, --logit-bias TOKEN_ID(+/-)BIAS\n");
    fprintf(stderr, "                        modifies the likelihood of token appearing in the completion,\n");
    fprintf(stderr, "                        i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n");
    fprintf(stderr, "                        or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'\n");
    fprintf(stderr, "  -c N, --ctx-size N    size of the prompt context (default: %d)\n", params.n_ctx);
    fprintf(stderr, "  --ignore-eos          ignore end of stream token and continue generating (implies --logit-bias 2-inf)\n");
    fprintf(stderr, "  --no-penalize-nl      do not penalize newline token\n");
    fprintf(stderr, "  --memory-f32          use f32 instead of f16 for memory key+value\n");
    fprintf(stderr, "  --temp N              temperature (default: %.1f)\n", (double)params.temp);
    fprintf(stderr, "  --n-parts N           number of model parts (default: -1 = determine from dimensions)\n");
    fprintf(stderr, "  -b N, --batch-size N  batch size for prompt processing (default: %d)\n", params.n_batch);
    fprintf(stderr, "  --perplexity          compute perplexity over the prompt\n");
    fprintf(stderr, "  --keep                number of tokens to keep from the initial prompt (default: %d, -1 = all)\n", params.n_keep);
    if (llama_mlock_supported()) {
        fprintf(stderr, "  --mlock               force system to keep model in RAM rather than swapping or compressing\n");
    }
    if (llama_mmap_supported()) {
        fprintf(stderr, "  --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)\n");
    }
    fprintf(stderr, "  -ngl N, --n-gpu-layers N\n");
    fprintf(stderr, "                        number of layers to store in VRAM\n");
    fprintf(stderr, "  --mtest               compute maximum memory usage\n");
    fprintf(stderr, "  --verbose-prompt      print prompt before generation\n");
    fprintf(stderr, "  --lora FNAME          apply LoRA adapter (implies --no-mmap)\n");
    fprintf(stderr, "  --lora-base FNAME     optional model to use as a base for the layers modified by the LoRA adapter\n");
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "\n");
}

std::string gpt_random_prompt(std::mt19937 & rng) {
    const int r = rng() % 10;
    switch (r) {
        case 0: return "So";
        case 1: return "Once upon a time";
        case 2: return "When";
        case 3: return "The";
        case 4: return "After";
        case 5: return "If";
        case 6: return "import";
        case 7: return "He";
        case 8: return "She";
        case 9: return "They";
        default: return "To";
    }

    return "The";
}

// TODO: not great allocating this every time
std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos) {
    // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
    std::vector<llama_token> res(text.size() + (int) add_bos);
    const int n = llama_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
    assert(n >= 0);
    res.resize(n);

    return res;
}

struct llama_context * llama_init_from_gpt_params(const gpt_params & params) {
    auto lparams = llama_context_default_params();

    lparams.n_ctx        = params.n_ctx;
    lparams.n_parts      = params.n_parts;
    lparams.n_gpu_layers = params.n_gpu_layers;
    lparams.seed         = params.seed;
    lparams.f16_kv       = params.memory_f16;
    lparams.use_mmap     = params.use_mmap;
    lparams.use_mlock    = params.use_mlock;
    lparams.logits_all   = params.perplexity;
    lparams.embedding    = params.embedding;

    llama_context * lctx = llama_init_from_file(params.model.c_str(), lparams);

    if (lctx == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
        return NULL;
    }

    if (!params.lora_adapter.empty()) {
        int err = llama_apply_lora_from_file(lctx,
                                             params.lora_adapter.c_str(),
                                             params.lora_base.empty() ? NULL : params.lora_base.c_str(),
                                             params.n_threads);
        if (err != 0) {
            fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
            return NULL;
        }
    }

    return lctx;
}

void console_init(console_state & con_st) {
#if defined(_WIN32)
    // Windows-specific console initialization
    DWORD dwMode = 0;
    con_st.hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    if (con_st.hConsole == INVALID_HANDLE_VALUE || !GetConsoleMode(con_st.hConsole, &dwMode)) {
        con_st.hConsole = GetStdHandle(STD_ERROR_HANDLE);
        if (con_st.hConsole != INVALID_HANDLE_VALUE && (!GetConsoleMode(con_st.hConsole, &dwMode))) {
            con_st.hConsole = NULL;
        }
    }
    if (con_st.hConsole) {
        // Enable ANSI colors on Windows 10+
        if (con_st.use_color && !(dwMode & ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
            SetConsoleMode(con_st.hConsole, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
        // Set console output codepage to UTF8
        SetConsoleOutputCP(CP_UTF8);
    }
    HANDLE hConIn = GetStdHandle(STD_INPUT_HANDLE);
    if (hConIn != INVALID_HANDLE_VALUE && GetConsoleMode(hConIn, &dwMode)) {
        // Set console input codepage to UTF16
        _setmode(_fileno(stdin), _O_WTEXT);

        // Turn off ICANON (ENABLE_LINE_INPUT) and ECHO (ENABLE_ECHO_INPUT)
        dwMode &= ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT);
        SetConsoleMode(hConIn, dwMode);
    }
#else
    // POSIX-specific console initialization
    struct termios new_termios;
    tcgetattr(STDIN_FILENO, &con_st.prev_state);
    new_termios = con_st.prev_state;
    new_termios.c_lflag &= ~(ICANON | ECHO);
    new_termios.c_cc[VMIN] = 1;
    new_termios.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);

    con_st.tty = fopen("/dev/tty", "w+");
    if (con_st.tty != nullptr) {
        con_st.out = con_st.tty;
    }

    setlocale(LC_ALL, "");
#endif
}

void console_cleanup(console_state & con_st) {
    // Reset console color
    console_set_color(con_st, CONSOLE_COLOR_DEFAULT);

#if !defined(_WIN32)
    if (con_st.tty != nullptr) {
        con_st.out = stdout;
        fclose(con_st.tty);
        con_st.tty = nullptr;
    }
    // Restore the terminal settings on POSIX systems
    tcsetattr(STDIN_FILENO, TCSANOW, &con_st.prev_state);
#endif
}

/* Keep track of current color of output, and emit ANSI code if it changes. */
void console_set_color(console_state & con_st, console_color_t color) {
    if (con_st.use_color && con_st.color != color) {
        fflush(stdout);
        switch(color) {
            case CONSOLE_COLOR_DEFAULT:
                fprintf(con_st.out, ANSI_COLOR_RESET);
                break;
            case CONSOLE_COLOR_PROMPT:
                fprintf(con_st.out, ANSI_COLOR_YELLOW);
                break;
            case CONSOLE_COLOR_USER_INPUT:
                fprintf(con_st.out, ANSI_BOLD ANSI_COLOR_GREEN);
                break;
        }
        con_st.color = color;
        fflush(con_st.out);
    }
}

char32_t getchar32() {
    wchar_t wc = getwchar();
    if (static_cast<wint_t>(wc) == WEOF) {
        return WEOF;
    }

#if WCHAR_MAX == 0xFFFF
    if ((wc >= 0xD800) && (wc <= 0xDBFF)) { // Check if wc is a high surrogate
        wchar_t low_surrogate = getwchar();
        if ((low_surrogate >= 0xDC00) && (low_surrogate <= 0xDFFF)) { // Check if the next wchar is a low surrogate
            return (static_cast<char32_t>(wc & 0x03FF) << 10) + (low_surrogate & 0x03FF) + 0x10000;
        }
    }
    if ((wc >= 0xD800) && (wc <= 0xDFFF)) { // Invalid surrogate pair
        return 0xFFFD; // Return the replacement character U+FFFD
    }
#endif

    return static_cast<char32_t>(wc);
}

void pop_cursor(console_state & con_st) {
#if defined(_WIN32)
    if (con_st.hConsole != NULL) {
        CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
        GetConsoleScreenBufferInfo(con_st.hConsole, &bufferInfo);

        COORD newCursorPosition = bufferInfo.dwCursorPosition;
        if (newCursorPosition.X == 0) {
            newCursorPosition.X = bufferInfo.dwSize.X - 1;
            newCursorPosition.Y -= 1;
        } else {
            newCursorPosition.X -= 1;
        }

        SetConsoleCursorPosition(con_st.hConsole, newCursorPosition);
        return;
    }
#endif
    putc('\b', con_st.out);
}

int estimateWidth(char32_t codepoint) {
#if defined(_WIN32)
    return 1;
#else
    return wcwidth(codepoint);
#endif
}

int put_codepoint(console_state & con_st, const char* utf8_codepoint, size_t length, int expectedWidth) {
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
    if (!GetConsoleScreenBufferInfo(con_st.hConsole, &bufferInfo)) {
        // go with the default
        return expectedWidth;
    }
    COORD initialPosition = bufferInfo.dwCursorPosition;
    DWORD nNumberOfChars = length;
    WriteConsole(con_st.hConsole, utf8_codepoint, nNumberOfChars, &nNumberOfChars, NULL);

    CONSOLE_SCREEN_BUFFER_INFO newBufferInfo;
    GetConsoleScreenBufferInfo(con_st.hConsole, &newBufferInfo);

    // Figure out our real position if we're in the last column
    if (utf8_codepoint[0] != 0x09 && initialPosition.X == newBufferInfo.dwSize.X - 1) {
        DWORD nNumberOfChars;
        WriteConsole(con_st.hConsole, &" \b", 2, &nNumberOfChars, NULL);
        GetConsoleScreenBufferInfo(con_st.hConsole, &newBufferInfo);
    }

    int width = newBufferInfo.dwCursorPosition.X - initialPosition.X;
    if (width < 0) {
        width += newBufferInfo.dwSize.X;
    }
    return width;
#else
    // we can trust expectedWidth if we've got one
    if (expectedWidth >= 0 || con_st.tty == nullptr) {
        fwrite(utf8_codepoint, length, 1, con_st.out);
        return expectedWidth;
    }

    fputs("\033[6n", con_st.tty); // Query cursor position
    int x1, x2, y1, y2;
    int results = 0;
    results = fscanf(con_st.tty, "\033[%d;%dR", &y1, &x1);

    fwrite(utf8_codepoint, length, 1, con_st.tty);

    fputs("\033[6n", con_st.tty); // Query cursor position
    results += fscanf(con_st.tty, "\033[%d;%dR", &y2, &x2);

    if (results != 4) {
        return expectedWidth;
    }

    int width = x2 - x1;
    if (width < 0) {
        // Calculate the width considering text wrapping
        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        width += w.ws_col;
    }
    return width;
#endif
}

void replace_last(console_state & con_st, char ch) {
#if defined(_WIN32)
    pop_cursor(con_st);
    put_codepoint(con_st, &ch, 1, 1);
#else
    fprintf(con_st.out, "\b%c", ch);
#endif
}

void append_utf8(char32_t ch, std::string & out) {
    if (ch <= 0x7F) {
        out.push_back(static_cast<unsigned char>(ch));
    } else if (ch <= 0x7FF) {
        out.push_back(static_cast<unsigned char>(0xC0 | ((ch >> 6) & 0x1F)));
        out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
    } else if (ch <= 0xFFFF) {
        out.push_back(static_cast<unsigned char>(0xE0 | ((ch >> 12) & 0x0F)));
        out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 6) & 0x3F)));
        out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
    } else if (ch <= 0x10FFFF) {
        out.push_back(static_cast<unsigned char>(0xF0 | ((ch >> 18) & 0x07)));
        out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 12) & 0x3F)));
        out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 6) & 0x3F)));
        out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
    } else {
        // Invalid Unicode code point
    }
}

// Helper function to remove the last UTF-8 character from a string
void pop_back_utf8_char(std::string & line) {
    if (line.empty()) {
        return;
    }

    size_t pos = line.length() - 1;

    // Find the start of the last UTF-8 character (checking up to 4 bytes back)
    for (size_t i = 0; i < 3 && pos > 0; ++i, --pos) {
        if ((line[pos] & 0xC0) != 0x80) break; // Found the start of the character
    }
    line.erase(pos);
}

bool console_readline(console_state & con_st, std::string & line) {
    console_set_color(con_st, CONSOLE_COLOR_USER_INPUT);
    if (con_st.out != stdout) {
        fflush(stdout);
    }

    line.clear();
    std::vector<int> widths;
    bool is_special_char = false;
    bool end_of_stream = false;

    char32_t input_char;
    while (true) {
        fflush(con_st.out); // Ensure all output is displayed before waiting for input
        input_char = getchar32();

        if (input_char == '\r' || input_char == '\n') {
            break;
        }

        if (input_char == WEOF || input_char == 0x04 /* Ctrl+D*/) {
            end_of_stream = true;
            break;
        }

        if (is_special_char) {
            console_set_color(con_st, CONSOLE_COLOR_USER_INPUT);
            replace_last(con_st, line.back());
            is_special_char = false;
        }

        if (input_char == '\033') { // Escape sequence
            char32_t code = getchar32();
            if (code == '[' || code == 0x1B) {
                // Discard the rest of the escape sequence
                while ((code = getchar32()) != WEOF) {
                    if ((code >= 'A' && code <= 'Z') || (code >= 'a' && code <= 'z') || code == '~') {
                        break;
                    }
                }
            }
        } else if (input_char == 0x08 || input_char == 0x7F) { // Backspace
            if (!widths.empty()) {
                int count;
                do {
                    count = widths.back();
                    widths.pop_back();
                    // Move cursor back, print space, and move cursor back again
                    for (int i = 0; i < count; i++) {
                        replace_last(con_st, ' ');
                        pop_cursor(con_st);
                    }
                    pop_back_utf8_char(line);
                } while (count == 0 && !widths.empty());
            }
        } else {
            int offset = line.length();
            append_utf8(input_char, line);
            int width = put_codepoint(con_st, line.c_str() + offset, line.length() - offset, estimateWidth(input_char));
            if (width < 0) {
                width = 0;
            }
            widths.push_back(width);
        }

        if (!line.empty() && (line.back() == '\\' || line.back() == '/')) {
            console_set_color(con_st, CONSOLE_COLOR_PROMPT);
            replace_last(con_st, line.back());
            is_special_char = true;
        }
    }

    bool has_more = con_st.multiline_input;
    if (is_special_char) {
        replace_last(con_st, ' ');
        pop_cursor(con_st);

        char last = line.back();
        line.pop_back();
        if (last == '\\') {
            line += '\n';
            fputc('\n', con_st.out);
            has_more = !has_more;
        } else {
            // llama will just eat the single space, it won't act as a space
            if (line.length() == 1 && line.back() == ' ') {
                line.clear();
                pop_cursor(con_st);
            }
            has_more = false;
        }
    } else {
        if (end_of_stream) {
            has_more = false;
        } else {
            line += '\n';
            fputc('\n', con_st.out);
        }
    }

    fflush(con_st.out);
    return has_more;
}
