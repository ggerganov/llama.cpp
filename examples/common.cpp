#include "common.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <regex>

#if defined (_WIN32)
#include <fcntl.h>
#include <io.h>
#pragma comment(lib,"kernel32.lib")
extern "C" __declspec(dllimport) void* __stdcall GetStdHandle(unsigned long nStdHandle);
extern "C" __declspec(dllimport) int __stdcall GetConsoleMode(void* hConsoleHandle, unsigned long* lpMode);
extern "C" __declspec(dllimport) int __stdcall SetConsoleMode(void* hConsoleHandle, unsigned long dwMode);
extern "C" __declspec(dllimport) int __stdcall SetConsoleCP(unsigned int wCodePageID);
extern "C" __declspec(dllimport) int __stdcall SetConsoleOutputCP(unsigned int wCodePageID);
extern "C" __declspec(dllimport) int __stdcall WideCharToMultiByte(unsigned int CodePage, unsigned long dwFlags,
                                                                   const wchar_t * lpWideCharStr, int cchWideChar,
                                                                   char * lpMultiByteStr, int cbMultiByte,
                                                                   const char * lpDefaultChar, bool * lpUsedDefaultChar);
#define CP_UTF8 65001
#endif

void split_args(const std::string & args_string, std::vector<std::string> & output_args)
{
    std::string current_arg = "";
    bool in_quotes = false;
    char quote_type;

    for (char c : args_string) {
        if (c == '"' || c == '\'') {
            if (!in_quotes) {
                in_quotes = true;
                quote_type = c;
            } else if (quote_type == c) {
                in_quotes = false;
            } else {
                current_arg += c;
            }
        } else if (in_quotes) {
            current_arg += c;
        } else if (std::isspace(c)) {
            if (current_arg != "") {
                output_args.push_back(current_arg);
                current_arg = "";
            }
        } else {
            current_arg += c;
        }
    }

    if (current_arg != "") {
        output_args.push_back(current_arg);
    }
}

std::string unescape(const std::string & str) {
    return std::regex_replace(str, std::regex("\\\\n"), "\n");
}

bool gpt_params_parse(int argc, char ** argv, gpt_params & params) {
    // determine sensible default number of threads.
    // std::thread::hardware_concurrency may not be equal to the number of cores, or may return 0.
#ifdef __linux__
    std::ifstream cpuinfo("/proc/cpuinfo");
    params.n_threads = std::count(std::istream_iterator<std::string>(cpuinfo),
                                  std::istream_iterator<std::string>(),
                                  std::string("processor"));
#endif
    if (params.n_threads == 0) {
        params.n_threads = std::max(1, (int32_t) std::thread::hardware_concurrency());
    }

    bool invalid_param = false;
    std::string arg;
    gpt_params default_params;

    // get additional arguments from config files
    std::vector<std::string> args;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg == "--config") {
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
            std::string args_string;
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(args_string));
            if (args_string.back() == '\n') {
                args_string.pop_back();
            }
            split_args(args_string, args);
            for (int j = 0; j < args.size(); j++) {
                args[j] = unescape(args[j]);
            }
        } else {
            args.emplace_back(argv[i]);
        }
    }

    // parse args
    int args_c = static_cast<int>(args.size());
    for (int i = 0; i < args_c && !invalid_param; i++) {
        arg = args[i];

        if (arg == "-s" || arg == "--seed") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.seed = std::stoi(args[i]);
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.n_threads = std::stoi(args[i]);
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.prompt = args[i];
        } else if (arg == "-f" || arg == "--file") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            std::ifstream file(args[i]);
            if (!file) {
                fprintf(stderr, "error: failed to open file '%s'\n", args[i].c_str());
                invalid_param = true;
                break;
            }
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
            if (params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        } else if (arg == "-n" || arg == "--n_predict") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.n_predict = std::stoi(args[i]);
        } else if (arg == "--top_k") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.top_k = std::stoi(args[i]);
        } else if (arg == "-c" || arg == "--ctx_size") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.n_ctx = std::stoi(args[i]);
        } else if (arg == "--memory_f32") {
            params.memory_f16 = false;
        } else if (arg == "--top_p") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.top_p = std::stof(args[i]);
        } else if (arg == "--temp") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.temp = std::stof(args[i]);
        } else if (arg == "--repeat_last_n") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.repeat_last_n = std::stoi(args[i]);
        } else if (arg == "--repeat_penalty") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.repeat_penalty = std::stof(args[i]);
        } else if (arg == "-b" || arg == "--batch_size") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.n_batch = std::stoi(args[i]);
            params.n_batch = std::min(512, params.n_batch);
        } else if (arg == "--keep") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.n_keep = std::stoi(args[i]);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.model = args[i];
        } else if (arg == "-i" || arg == "--interactive") {
            params.interactive = true;
        } else if (arg == "--embedding") {
            params.embedding = true;
        } else if (arg == "--clean-interface") {
            params.clean_interface = true;
        } else if (arg == "--interactive-start") {
            params.interactive = true;
        } else if (arg == "--interactive-first") {
            params.interactive_start = true;
        } else if (arg == "-ins" || arg == "--instruct") {
            fprintf(stderr, "\n\nWarning: instruct mode is deprecated! Use: \n"
                "--clean-interface "
                "--interactive-first "
                "--keep -1 "
                "--ins-prefix-bos "
                "--ins-prefix \"\\n\\n### Instruction:\\n\\n\" "
                "--ins-suffix \"\\n\\n### Response:\\n\\n\" "
                "-r \"### Instruction:\\n\\n\" "
            "\n\n");
            // params.instruct = true;
            params.clean_interface = true;
            params.interactive_start = true;
            params.n_keep = -1;
            params.instruct_prefix_bos = true;
            params.instruct_prefix = "\n\n### Instruction:\n\n";
            params.instruct_suffix = "\n\n### Response:\n\n";
            params.antiprompt.push_back("### Instruction:\n\n");
        } else if (arg == "--color") {
            params.use_color = true;
        } else if (arg == "--disable-multiline") {
            params.multiline_mode = false;
        } else if (arg == "--mlock") {
            params.use_mlock = true;
        } else if (arg == "--no-mmap") {
            params.use_mmap = false;
        } else if (arg == "--mtest") {
            params.mem_test = true;
        } else if (arg == "--verbose-prompt") {
            params.verbose_prompt = true;
        } else if (arg == "-r" || arg == "--reverse-prompt") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.antiprompt.push_back(args[i]);
        } else if (arg == "--stop-prompt") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.stopprompt.push_back(args[i]);
        } else if (arg == "--rm-trailing-space-workaround") {
            params.rm_trailing_space_workaround = true;
        } else if (arg == "--perplexity") {
            params.perplexity = true;
        } else if (arg == "--ignore-eos") {
            params.ignore_eos = true;
        } else if (arg == "--n_parts") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.n_parts = std::stoi(args[i]);
        } else if (arg == "-h" || arg == "--help") {
            gpt_print_usage(argv[0], default_params);
            exit(0);
        } else if (arg == "--random-prompt") {
            params.random_prompt = true;
        } else if (arg == "--in-prefix") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.input_prefix = args[i];
        } else if (arg == "--ins-prefix-bos") {
            params.instruct_prefix_bos = true;
        } else if (arg == "--ins-prefix") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.instruct_prefix = args[i];
        } else if (arg == "--ins-suffix-bos") {
            params.instruct_suffix_bos = true;
        } else if (arg == "--ins-suffix") {
            if (++i >= args_c) {
                invalid_param = true;
                break;
            }
            params.instruct_suffix = args[i];
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            gpt_print_usage(argv[0], default_params);
            exit(1);
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        gpt_print_usage(argv[0], default_params);
        exit(1);
    }

    return true;
}

void gpt_print_usage(char * argv_0, const gpt_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv_0);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -i, --interactive     run in interactive mode\n");
    fprintf(stderr, "  --interactive-first   run in interactive mode and wait for input right away\n");
    fprintf(stderr, "  --clean-interface     hides input prefix & suffix and displays '>' instead\n");
    fprintf(stderr, "  -r PROMPT, --reverse-prompt PROMPT\n");
    fprintf(stderr, "                        run in interactive mode and poll user input upon seeing PROMPT (can be\n");
    fprintf(stderr, "                        specified more than once for multiple prompts).\n");
    fprintf(stderr, "  --color               colorise output to distinguish prompt and user input from generations\n");
    fprintf(stderr, "  --disable-multiline   disable multiline mode (use Ctrl+D on Linux/Mac and Ctrl+Z then Return on Windows to toggle multiline)\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for <= 0)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: empty)\n");
    fprintf(stderr, "  --random-prompt       start with a randomized prompt.\n");
    fprintf(stderr, "  --in-prefix STRING    string to prefix user inputs with (default: empty)\n");
    fprintf(stderr, "  --ins-prefix STRING   (instruct) prefix user inputs with tokenized string (default: empty)\n");
    fprintf(stderr, "  --ins-prefix-bos      (instruct) prepend bos token to instruct prefix.\n");
    fprintf(stderr, "  --ins-suffix STRING   (instruct) suffix user inputs with tokenized string (default: empty)\n");
    fprintf(stderr, "  --ins-suffix-bos      (instruct) prepend bos token to instruct suffix.\n");
    fprintf(stderr, "  -f FNAME, --file FNAME\n");
    fprintf(stderr, "                        prompt file to start generation.\n");
    fprintf(stderr, "  -n N, --n_predict N   number of tokens to predict (default: %d, -1 = infinity)\n", params.n_predict);
    fprintf(stderr, "  --top_k N             top-k sampling (default: %d)\n", params.top_k);
    fprintf(stderr, "  --top_p N             top-p sampling (default: %.1f)\n", (double)params.top_p);
    fprintf(stderr, "  --repeat_last_n N     last n tokens to consider for penalize (default: %d)\n", params.repeat_last_n);
    fprintf(stderr, "  --repeat_penalty N    penalize repeat sequence of tokens (default: %.1f)\n", (double)params.repeat_penalty);
    fprintf(stderr, "  -c N, --ctx_size N    size of the prompt context (default: %d)\n", params.n_ctx);
    fprintf(stderr, "  --ignore-eos          ignore end of stream token and continue generating\n");
    fprintf(stderr, "  --memory_f32          use f32 instead of f16 for memory key+value\n");
    fprintf(stderr, "  --temp N              temperature (default: %.1f)\n", (double)params.temp);
    fprintf(stderr, "  --n_parts N           number of model parts (default: -1 = determine from dimensions)\n");
    fprintf(stderr, "  -b N, --batch_size N  batch size for prompt processing (default: %d)\n", params.n_batch);
    fprintf(stderr, "  --perplexity          compute perplexity over the prompt\n");
    fprintf(stderr, "  --keep                number of tokens to keep from the initial prompt (default: %d, -1 = all)\n", params.n_keep);
    if (llama_mlock_supported()) {
        fprintf(stderr, "  --mlock               force system to keep model in RAM rather than swapping or compressing\n");
    }
    if (llama_mmap_supported()) {
        fprintf(stderr, "  --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)\n");
    }
    fprintf(stderr, "  --mtest               compute maximum memory usage\n");
    fprintf(stderr, "  --verbose-prompt      print prompt before generation\n");
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
    std::vector<llama_token> res(text.size() + (int)add_bos);
    int n = llama_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
    assert(n >= 0);
    res.resize(n);

    return res;
}

/* Keep track of current color of output, and emit ANSI code if it changes. */
void set_console_color(console_state & con_st, console_color_t color) {
    if (con_st.use_color && con_st.color != color) {
        switch(color) {
            case CONSOLE_COLOR_DEFAULT:
                printf(ANSI_COLOR_RESET);
                break;
            case CONSOLE_COLOR_PROMPT:
                printf(ANSI_COLOR_YELLOW);
                break;
            case CONSOLE_COLOR_USER_INPUT:
                printf(ANSI_BOLD ANSI_COLOR_GREEN);
                break;
        }
        con_st.color = color;
    }
}

#if defined (_WIN32)
void win32_console_init(bool enable_color) {
    unsigned long dwMode = 0;
    void* hConOut = GetStdHandle((unsigned long)-11); // STD_OUTPUT_HANDLE (-11)
    if (!hConOut || hConOut == (void*)-1 || !GetConsoleMode(hConOut, &dwMode)) {
        hConOut = GetStdHandle((unsigned long)-12); // STD_ERROR_HANDLE (-12)
        if (hConOut && (hConOut == (void*)-1 || !GetConsoleMode(hConOut, &dwMode))) {
            hConOut = 0;
        }
    }
    if (hConOut) {
        // Enable ANSI colors on Windows 10+
        if (enable_color && !(dwMode & 0x4)) {
            SetConsoleMode(hConOut, dwMode | 0x4); // ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x4)
        }
        // Set console output codepage to UTF8
        SetConsoleOutputCP(CP_UTF8);
    }
    void* hConIn = GetStdHandle((unsigned long)-10); // STD_INPUT_HANDLE (-10)
    if (hConIn && hConIn != (void*)-1 && GetConsoleMode(hConIn, &dwMode)) {
        // Set console input codepage to UTF16
        _setmode(_fileno(stdin), _O_WTEXT);
    }
}

// Convert a wide Unicode string to an UTF8 string
void win32_utf8_encode(const std::wstring & wstr, std::string & str) {
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
    str = strTo;
}
#endif

bool get_input_text(std::string & input_text, bool eof_toggled_multiline_mode) {
    bool another_line = true;
    bool is_eof_multiline_toggled = false;
    do {
        std::string line;
#if defined(_WIN32)
        auto & stdcin = std::wcin;
        std::wstring wline;
        if (!std::getline(stdcin, wline)) {
            // input stream is bad or EOF received
            if (stdcin.bad()) {
                fprintf(stderr, "%s: error: input stream bad\n", __func__);
                return 1;
            }
        }
        win32_utf8_encode(wline, line);
#else
        auto & stdcin = std::cin;
        if (!std::getline(stdcin, line)) {
            // input stream is bad or EOF received
            if (stdcin.bad()) {
                fprintf(stderr, "%s: error: input stream bad\n", __func__);
                return 1;
            }
        }
#endif
        if (stdcin.eof()) {
            stdcin.clear();
            stdcin.seekg(0, std::ios::beg);
            if (!eof_toggled_multiline_mode) {
                another_line = false;
            } else {
                is_eof_multiline_toggled = !is_eof_multiline_toggled;
                if (is_eof_multiline_toggled) {
                    input_text += line;
                    continue;
                }
            }
        }
        if (!eof_toggled_multiline_mode) {
            if (line.empty() || line.back() != '\\') {
                another_line = false;
            } else {
                line.pop_back(); // Remove the continue character
            }
        } else {
            if (!is_eof_multiline_toggled) {
                another_line = false;
            }
        }
        input_text += line;
        if (another_line) {
            input_text += '\n'; // Append the line to the result
        }
    } while (another_line);
    return true;
}
