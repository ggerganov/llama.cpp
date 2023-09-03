#include "common.h"
#include "build-info.h"
#include "llama.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iterator>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <cinttypes>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <codecvt>
#include <locale>
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
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
    std::string arg;
    gpt_params default_params;
    const std::string arg_prefix = "--";

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "-s" || arg == "--seed") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.seed = std::stoul(argv[i]);
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_threads = std::stoi(argv[i]);
            if (params.n_threads <= 0) {
                params.n_threads = std::thread::hardware_concurrency();
            }
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.prompt = argv[i];
        } else if (arg == "-e" || arg == "--escape") {
            params.escape = true;
        } else if (arg == "--prompt-cache") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.path_prompt_cache = argv[i];
        } else if (arg == "--prompt-cache-all") {
            params.prompt_cache_all = true;
        } else if (arg == "--prompt-cache-ro") {
            params.prompt_cache_ro = true;
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
        } else if (arg == "--rope-freq-base") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.rope_freq_base = std::stof(argv[i]);
        } else if (arg == "--rope-freq-scale") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.rope_freq_scale = std::stof(argv[i]);
        } else if (arg == "--rope-scale") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.rope_freq_scale = 1.0f/std::stof(argv[i]);
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
        } else if (arg == "--cfg-negative-prompt") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.cfg_negative_prompt = argv[i];
        } else if (arg == "--cfg-negative-prompt-file") {
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
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.cfg_negative_prompt));
            if (params.cfg_negative_prompt.back() == '\n') {
                params.cfg_negative_prompt.pop_back();
            }
        } else if (arg == "--cfg-scale") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.cfg_scale = std::stof(argv[i]);
        } else if (arg == "-b" || arg == "--batch-size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_batch = std::stoi(argv[i]);
        } else if (arg == "--keep") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_keep = std::stoi(argv[i]);
        } else if (arg == "--chunks") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_chunks = std::stoi(argv[i]);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model = argv[i];
        } else if (arg == "-a" || arg == "--alias") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model_alias = argv[i];
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
        } else if (arg == "--simple-io") {
            params.simple_io = true;
        } else if (arg == "--color") {
            params.use_color = true;
        } else if (arg == "--mlock") {
            params.use_mlock = true;
        } else if (arg == "--gpu-layers" || arg == "-ngl" || arg == "--n-gpu-layers") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
#ifdef LLAMA_SUPPORTS_GPU_OFFLOAD
            params.n_gpu_layers = std::stoi(argv[i]);
#else
            fprintf(stderr, "warning: not compiled with GPU offload support, --n-gpu-layers option will be ignored\n");
            fprintf(stderr, "warning: see main README.md for information on enabling GPU BLAS support\n");
#endif
        } else if (arg == "--main-gpu" || arg == "-mg") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
#ifdef GGML_USE_CUBLAS
            params.main_gpu = std::stoi(argv[i]);
#else
            fprintf(stderr, "warning: llama.cpp was compiled without cuBLAS. It is not possible to set a main GPU.\n");
#endif
        } else if (arg == "--tensor-split" || arg == "-ts") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
#ifdef GGML_USE_CUBLAS
            std::string arg_next = argv[i];

            // split string by , and /
            const std::regex regex{R"([,/]+)"};
            std::sregex_token_iterator it{arg_next.begin(), arg_next.end(), regex, -1};
            std::vector<std::string> split_arg{it, {}};
            GGML_ASSERT(split_arg.size() <= LLAMA_MAX_DEVICES);

            for (size_t i = 0; i < LLAMA_MAX_DEVICES; ++i) {
                if (i < split_arg.size()) {
                    params.tensor_split[i] = std::stof(split_arg[i]);
                } else {
                    params.tensor_split[i] = 0.0f;
                }
            }
#else
            fprintf(stderr, "warning: llama.cpp was compiled without cuBLAS. It is not possible to set a tensor split.\n");
#endif // GGML_USE_CUBLAS
        } else if (arg == "--no-mul-mat-q" || arg == "-nommq") {
#ifdef GGML_USE_CUBLAS
            params.mul_mat_q = false;
#else
            fprintf(stderr, "warning: llama.cpp was compiled without cuBLAS. Disabling mul_mat_q kernels has no effect.\n");
#endif // GGML_USE_CUBLAS
        } else if (arg == "--low-vram" || arg == "-lv") {
#ifdef GGML_USE_CUBLAS
            params.low_vram = true;
#else
            fprintf(stderr, "warning: llama.cpp was compiled without cuBLAS. It is not possible to set lower vram usage.\n");
#endif // GGML_USE_CUBLAS
        } else if (arg == "--no-mmap") {
            params.use_mmap = false;
        } else if (arg == "--mtest") {
            params.mem_test = true;
        } else if (arg == "--numa") {
            params.numa = true;
        } else if (arg == "--export") {
            params.export_cgraph = true;
        } else if (arg == "--verbose-prompt") {
            params.verbose_prompt = true;
        } else if (arg == "-r" || arg == "--reverse-prompt") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.antiprompt.push_back(argv[i]);
        } else if (arg == "-ld" || arg == "--logdir") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.logdir = argv[i];

            if (params.logdir.back() != DIRECTORY_SEPARATOR) {
                params.logdir += DIRECTORY_SEPARATOR;
            }
        } else if (arg == "--perplexity") {
            params.perplexity = true;
        } else if (arg == "--ppl-stride") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.ppl_stride = std::stoi(argv[i]);
        } else if (arg == "--ppl-output-type") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.ppl_output_type = std::stoi(argv[i]);
        } else if (arg == "--hellaswag") {
            params.hellaswag = true;
        } else if (arg == "--hellaswag-tasks") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.hellaswag_tasks = std::stoi(argv[i]);
        } else if (arg == "--ignore-eos") {
            params.ignore_eos = true;
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
            } catch (const std::exception&) {
                invalid_param = true;
                break;
            }
        } else if (arg == "-h" || arg == "--help") {
            gpt_print_usage(argc, argv, default_params);
#ifndef LOG_DISABLE_LOGS
            log_print_usage();
#endif // LOG_DISABLE_LOGS
            exit(0);
        } else if (arg == "--random-prompt") {
            params.random_prompt = true;
        } else if (arg == "--in-prefix-bos") {
            params.input_prefix_bos = true;
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
        } else if (arg == "--grammar") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.grammar = argv[i];
        } else if (arg == "--grammar-file") {
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
            std::copy(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(),
                std::back_inserter(params.grammar)
            );
#ifndef LOG_DISABLE_LOGS
        // Parse args for logging parameters
        } else if ( log_param_single_parse( argv[i] ) ) {
            // Do nothing, log_param_single_parse automatically does it's thing
            //  and returns if a match was found and parsed.
        } else if ( log_param_pair_parse( /*check_but_dont_parse*/ true, argv[i] ) ) {
            // We have a matching known parameter requiring an argument,
            //  now we need to check if there is anything after this argv
            //  and flag invalid_param or parse it.
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            if( !log_param_pair_parse( /*check_but_dont_parse*/ false, argv[i-1], argv[i]) ) {
                invalid_param = true;
                break;
            }
        // End of Parse args for logging parameters
#endif // LOG_DISABLE_LOGS
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
             params.instruct)) {
        fprintf(stderr, "error: --prompt-cache-all not supported in interactive mode yet\n");
        gpt_print_usage(argc, argv, default_params);
        exit(1);
    }

    if (params.escape) {
        process_escapes(params.prompt);
        process_escapes(params.input_prefix);
        process_escapes(params.input_suffix);
    }

    return true;
}

void gpt_print_usage(int /*argc*/, char ** argv, const gpt_params & params) {
    fprintf(stdout, "usage: %s [options]\n", argv[0]);
    fprintf(stdout, "\n");
    fprintf(stdout, "options:\n");
    fprintf(stdout, "  -h, --help            show this help message and exit\n");
    fprintf(stdout, "  -i, --interactive     run in interactive mode\n");
    fprintf(stdout, "  --interactive-first   run in interactive mode and wait for input right away\n");
    fprintf(stdout, "  -ins, --instruct      run in instruction mode (use with Alpaca models)\n");
    fprintf(stdout, "  --multiline-input     allows you to write or paste multiple lines without ending each in '\\'\n");
    fprintf(stdout, "  -r PROMPT, --reverse-prompt PROMPT\n");
    fprintf(stdout, "                        halt generation at PROMPT, return control in interactive mode\n");
    fprintf(stdout, "                        (can be specified more than once for multiple prompts).\n");
    fprintf(stdout, "  --color               colorise output to distinguish prompt and user input from generations\n");
    fprintf(stdout, "  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for < 0)\n");
    fprintf(stdout, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stdout, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stdout, "                        prompt to start generation with (default: empty)\n");
    fprintf(stdout, "  -e, --escape          process prompt escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\)\n");
    fprintf(stdout, "  --prompt-cache FNAME  file to cache prompt state for faster startup (default: none)\n");
    fprintf(stdout, "  --prompt-cache-all    if specified, saves user input and generations to cache as well.\n");
    fprintf(stdout, "                        not supported with --interactive or other interactive options\n");
    fprintf(stdout, "  --prompt-cache-ro     if specified, uses the prompt cache but does not update it.\n");
    fprintf(stdout, "  --random-prompt       start with a randomized prompt.\n");
    fprintf(stdout, "  --in-prefix-bos       prefix BOS to user inputs, preceding the `--in-prefix` string\n");
    fprintf(stdout, "  --in-prefix STRING    string to prefix user inputs with (default: empty)\n");
    fprintf(stdout, "  --in-suffix STRING    string to suffix after user inputs with (default: empty)\n");
    fprintf(stdout, "  -f FNAME, --file FNAME\n");
    fprintf(stdout, "                        prompt file to start generation.\n");
    fprintf(stdout, "  -n N, --n-predict N   number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)\n", params.n_predict);
    fprintf(stdout, "  -c N, --ctx-size N    size of the prompt context (default: %d)\n", params.n_ctx);
    fprintf(stdout, "  -b N, --batch-size N  batch size for prompt processing (default: %d)\n", params.n_batch);
    fprintf(stdout, "  --top-k N             top-k sampling (default: %d, 0 = disabled)\n", params.top_k);
    fprintf(stdout, "  --top-p N             top-p sampling (default: %.1f, 1.0 = disabled)\n", (double)params.top_p);
    fprintf(stdout, "  --tfs N               tail free sampling, parameter z (default: %.1f, 1.0 = disabled)\n", (double)params.tfs_z);
    fprintf(stdout, "  --typical N           locally typical sampling, parameter p (default: %.1f, 1.0 = disabled)\n", (double)params.typical_p);
    fprintf(stdout, "  --repeat-last-n N     last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)\n", params.repeat_last_n);
    fprintf(stdout, "  --repeat-penalty N    penalize repeat sequence of tokens (default: %.1f, 1.0 = disabled)\n", (double)params.repeat_penalty);
    fprintf(stdout, "  --presence-penalty N  repeat alpha presence penalty (default: %.1f, 0.0 = disabled)\n", (double)params.presence_penalty);
    fprintf(stdout, "  --frequency-penalty N repeat alpha frequency penalty (default: %.1f, 0.0 = disabled)\n", (double)params.frequency_penalty);
    fprintf(stdout, "  --mirostat N          use Mirostat sampling.\n");
    fprintf(stdout, "                        Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.\n");
    fprintf(stdout, "                        (default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)\n", params.mirostat);
    fprintf(stdout, "  --mirostat-lr N       Mirostat learning rate, parameter eta (default: %.1f)\n", (double)params.mirostat_eta);
    fprintf(stdout, "  --mirostat-ent N      Mirostat target entropy, parameter tau (default: %.1f)\n", (double)params.mirostat_tau);
    fprintf(stdout, "  -l TOKEN_ID(+/-)BIAS, --logit-bias TOKEN_ID(+/-)BIAS\n");
    fprintf(stdout, "                        modifies the likelihood of token appearing in the completion,\n");
    fprintf(stdout, "                        i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n");
    fprintf(stdout, "                        or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'\n");
    fprintf(stdout, "  --grammar GRAMMAR     BNF-like grammar to constrain generations (see samples in grammars/ dir)\n");
    fprintf(stdout, "  --grammar-file FNAME  file to read grammar from\n");
    fprintf(stdout, "  --cfg-negative-prompt PROMPT\n");
    fprintf(stdout, "                        negative prompt to use for guidance. (default: empty)\n");
    fprintf(stdout, "  --cfg-negative-prompt-file FNAME\n");
    fprintf(stdout, "                        negative prompt file to use for guidance. (default: empty)\n");
    fprintf(stdout, "  --cfg-scale N         strength of guidance (default: %f, 1.0 = disable)\n", params.cfg_scale);
    fprintf(stdout, "  --rope-scale N        RoPE context linear scaling factor, inverse of --rope-freq-scale (default: %g)\n", 1.0f/params.rope_freq_scale);
    fprintf(stdout, "  --rope-freq-base N    RoPE base frequency, used by NTK-aware scaling (default: %.1f)\n", params.rope_freq_base);
    fprintf(stdout, "  --rope-freq-scale N   RoPE frequency linear scaling factor, inverse of --rope-scale (default: %g)\n", params.rope_freq_scale);
    fprintf(stdout, "  --ignore-eos          ignore end of stream token and continue generating (implies --logit-bias 2-inf)\n");
    fprintf(stdout, "  --no-penalize-nl      do not penalize newline token\n");
    fprintf(stdout, "  --memory-f32          use f32 instead of f16 for memory key+value (default: disabled)\n");
    fprintf(stdout, "                        not recommended: doubles context memory required and no measurable increase in quality\n");
    fprintf(stdout, "  --temp N              temperature (default: %.1f)\n", (double)params.temp);
    fprintf(stdout, "  --perplexity          compute perplexity over each ctx window of the prompt\n");
    fprintf(stdout, "  --hellaswag           compute HellaSwag score over random tasks from datafile supplied with -f\n");
    fprintf(stdout, "  --hellaswag-tasks N   number of tasks to use when computing the HellaSwag score (default: %zu)\n", params.hellaswag_tasks);
    fprintf(stdout, "  --keep N              number of tokens to keep from the initial prompt (default: %d, -1 = all)\n", params.n_keep);
    fprintf(stdout, "  --chunks N            max number of chunks to process (default: %d, -1 = all)\n", params.n_chunks);
    if (llama_mlock_supported()) {
        fprintf(stdout, "  --mlock               force system to keep model in RAM rather than swapping or compressing\n");
    }
    if (llama_mmap_supported()) {
        fprintf(stdout, "  --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)\n");
    }
    fprintf(stdout, "  --numa                attempt optimizations that help on some NUMA systems\n");
    fprintf(stdout, "                        if run without this previously, it is recommended to drop the system page cache before using this\n");
    fprintf(stdout, "                        see https://github.com/ggerganov/llama.cpp/issues/1437\n");
#ifdef LLAMA_SUPPORTS_GPU_OFFLOAD
    fprintf(stdout, "  -ngl N, --n-gpu-layers N\n");
    fprintf(stdout, "                        number of layers to store in VRAM\n");
    fprintf(stdout, "  -ts SPLIT --tensor-split SPLIT\n");
    fprintf(stdout, "                        how to split tensors across multiple GPUs, comma-separated list of proportions, e.g. 3,1\n");
    fprintf(stdout, "  -mg i, --main-gpu i   the GPU to use for scratch and small tensors\n");
    fprintf(stdout, "  -lv, --low-vram       don't allocate VRAM scratch buffer\n");
#ifdef GGML_USE_CUBLAS
    fprintf(stdout, "  -nommq, --no-mul-mat-q\n");
    fprintf(stdout, "                        use " GGML_CUBLAS_NAME " instead of custom mul_mat_q " GGML_CUDA_NAME " kernels.\n");
    fprintf(stdout, "                        Not recommended since this is both slower and uses more VRAM.\n");
#endif // GGML_USE_CUBLAS
#endif
    fprintf(stdout, "  --mtest               compute maximum memory usage\n");
    fprintf(stdout, "  --export              export the computation graph to 'llama.ggml'\n");
    fprintf(stdout, "  --verbose-prompt      print prompt before generation\n");
    fprintf(stderr, "  --simple-io           use basic IO for better compatibility in subprocesses and limited consoles\n");
    fprintf(stdout, "  --lora FNAME          apply LoRA adapter (implies --no-mmap)\n");
    fprintf(stdout, "  --lora-base FNAME     optional model to use as a base for the layers modified by the LoRA adapter\n");
    fprintf(stdout, "  -m FNAME, --model FNAME\n");
    fprintf(stdout, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stdout, "  -ld LOGDIR, --logdir LOGDIR\n");
    fprintf(stdout, "                        path under which to save YAML logs (no logging if unset)\n");
    fprintf(stdout, "\n");
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

//
// Model utils
//

struct llama_context_params llama_context_params_from_gpt_params(const gpt_params & params) {
    auto lparams = llama_context_default_params();

    lparams.n_ctx           = params.n_ctx;
    lparams.n_batch         = params.n_batch;
    lparams.n_gpu_layers    = params.n_gpu_layers;
    lparams.main_gpu        = params.main_gpu;
    lparams.tensor_split    = params.tensor_split;
    lparams.low_vram        = params.low_vram;
    lparams.mul_mat_q       = params.mul_mat_q;
    lparams.seed            = params.seed;
    lparams.f16_kv          = params.memory_f16;
    lparams.use_mmap        = params.use_mmap;
    lparams.use_mlock       = params.use_mlock;
    lparams.logits_all      = params.perplexity;
    lparams.embedding       = params.embedding;
    lparams.rope_freq_base  = params.rope_freq_base;
    lparams.rope_freq_scale = params.rope_freq_scale;

    return lparams;
}

std::tuple<struct llama_model *, struct llama_context *> llama_init_from_gpt_params(gpt_params & params) {
    auto lparams = llama_context_params_from_gpt_params(params);

    llama_model * model  = llama_load_model_from_file(params.model.c_str(), lparams);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
        return std::make_tuple(nullptr, nullptr);
    }

    llama_context * lctx = llama_new_context_with_model(model, lparams);
    if (lctx == NULL) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params.model.c_str());
        llama_free_model(model);
        return std::make_tuple(nullptr, nullptr);
    }

    if (!params.lora_adapter.empty()) {
        int err = llama_model_apply_lora_from_file(model,
                                             params.lora_adapter.c_str(),
                                             params.lora_base.empty() ? NULL : params.lora_base.c_str(),
                                             params.n_threads);
        if (err != 0) {
            fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
            llama_free(lctx);
            llama_free_model(model);
            return std::make_tuple(nullptr, nullptr);
        }
    }

    if (params.ignore_eos) {
        params.logit_bias[llama_token_eos(lctx)] = -INFINITY;
    }

    {
        LOG("warming up the model with an empty run\n");

        const std::vector<llama_token> tmp = { llama_token_bos(lctx), };
        llama_eval(lctx, tmp.data(), tmp.size(), 0, params.n_threads);
        llama_reset_timings(lctx);
    }

    return std::make_tuple(model, lctx);
}

//
// Vocab utils
//

std::vector<llama_token> llama_tokenize(
        struct llama_context * ctx,
           const std::string & text,
                        bool   add_bos) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + add_bos;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(ctx, text.c_str(), result.data(), result.size(), add_bos);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(ctx, text.c_str(), result.data(), result.size(), add_bos);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(ctx, token, result.data(), result.size());
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(ctx, token, result.data(), result.size());
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

std::string llama_detokenize_spm(llama_context * ctx, const std::vector<llama_token> & tokens) {
    const llama_token bos_id = llama_token_bos(ctx);

    std::string piece;
    std::string result;

    for (size_t i = 0; i < tokens.size(); ++i) {
        piece = llama_token_to_piece(ctx, tokens[i]);

        // remove the leading space of the first non-BOS token
        if (((tokens[0] == bos_id && i == 1) || (tokens[0] != bos_id && i == 0)) && piece[0] == ' ') {
            piece = piece.substr(1);
        }

        result += piece;
    }

    return result;
}

std::string llama_detokenize_bpe(llama_context * ctx, const std::vector<llama_token> & tokens) {
    std::string piece;
    std::string result;

    for (size_t i = 0; i < tokens.size(); ++i) {
        piece = llama_token_to_piece(ctx, tokens[i]);

        result += piece;
    }

    return result;
}

// returns true if successful, false otherwise
bool create_directory_with_parents(const std::string & path) {
#ifdef _WIN32
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring wpath = converter.from_bytes(path);

    // if the path already exists, check whether it's a directory
    const DWORD attributes = GetFileAttributesW(wpath.c_str());
    if ((attributes != INVALID_FILE_ATTRIBUTES) && (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        return true;
    }

    size_t pos_slash = 0;

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('\\', pos_slash)) != std::string::npos) {
        const std::wstring subpath = wpath.substr(0, pos_slash);
        const wchar_t * test = subpath.c_str();

        const bool success = CreateDirectoryW(test, NULL);
        if (!success) {
            const DWORD error = GetLastError();

            // if the path already exists, ensure that it's a directory
            if (error == ERROR_ALREADY_EXISTS) {
                const DWORD attributes = GetFileAttributesW(subpath.c_str());
                if (attributes == INVALID_FILE_ATTRIBUTES || !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    return false;
                }
            } else {
                return false;
            }
        }

        pos_slash += 1;
    }

    return true;
#else
    // if the path already exists, check whether it's a directory
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return S_ISDIR(info.st_mode);
    }

    size_t pos_slash = 1; // skip leading slashes for directory creation

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('/', pos_slash)) != std::string::npos) {
        const std::string subpath = path.substr(0, pos_slash);
        struct stat info;

        // if the path already exists, ensure that it's a directory
        if (stat(subpath.c_str(), &info) == 0) {
            if (!S_ISDIR(info.st_mode)) {
                return false;
            }
        } else {
            // create parent directories
            const int ret = mkdir(subpath.c_str(), 0755);
            if (ret != 0) {
                return false;
            }
        }

        pos_slash += 1;
    }

    return true;
#endif // _WIN32
}

void dump_vector_float_yaml(FILE * stream, const char * prop_name, const std::vector<float> & data) {
    if (data.empty()) {
        fprintf(stream, "%s:\n", prop_name);
        return;
    }

    fprintf(stream, "%s: [", prop_name);
    for (size_t i = 0; i < data.size() - 1; ++i) {
        fprintf(stream, "%e, ", data[i]);
    }
    fprintf(stream, "%e]\n", data.back());
}

void dump_vector_int_yaml(FILE * stream, const char * prop_name, const std::vector<int> & data) {
    if (data.empty()) {
        fprintf(stream, "%s:\n", prop_name);
        return;
    }

    fprintf(stream, "%s: [", prop_name);
    for (size_t i = 0; i < data.size() - 1; ++i) {
        fprintf(stream, "%d, ", data[i]);
    }
    fprintf(stream, "%d]\n", data.back());
}

void dump_string_yaml_multiline(FILE * stream, const char * prop_name, const char * data) {
    std::string data_str(data == NULL ? "" : data);

    if (data_str.empty()) {
        fprintf(stream, "%s:\n", prop_name);
        return;
    }

    size_t pos_start = 0;
    size_t pos_found = 0;

    if (!data_str.empty() && (std::isspace(data_str[0]) || std::isspace(data_str.back()))) {
        data_str = std::regex_replace(data_str, std::regex("\n"), "\\n");
        data_str = std::regex_replace(data_str, std::regex("\""), "\\\"");
        data_str = "\"" + data_str + "\"";
        fprintf(stream, "%s: %s\n", prop_name, data_str.c_str());
        return;
    }

    if (data_str.find('\n') == std::string::npos) {
        fprintf(stream, "%s: %s\n", prop_name, data_str.c_str());
        return;
    }

    fprintf(stream, "%s: |\n", prop_name);
    while ((pos_found = data_str.find('\n', pos_start)) != std::string::npos) {
        fprintf(stream, "  %s\n", data_str.substr(pos_start, pos_found-pos_start).c_str());
        pos_start = pos_found + 1;
    }
}

std::string get_sortable_timestamp() {
    using clock = std::chrono::system_clock;

    const clock::time_point current_time = clock::now();
    const time_t as_time_t = clock::to_time_t(current_time);
    char timestamp_no_ns[100];
    std::strftime(timestamp_no_ns, 100, "%Y_%m_%d-%H_%M_%S", std::localtime(&as_time_t));

    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        current_time.time_since_epoch() % 1000000000).count();
    char timestamp_ns[11];
    snprintf(timestamp_ns, 11, "%09" PRId64, ns);

    return std::string(timestamp_no_ns) + "." + std::string(timestamp_ns);
}

void dump_non_result_info_yaml(FILE * stream, const gpt_params & params, const llama_context * lctx,
                               const std::string & timestamp, const std::vector<int> & prompt_tokens, const char * model_desc) {
    fprintf(stream, "build_commit: %s\n", BUILD_COMMIT);
    fprintf(stream, "build_number: %d\n", BUILD_NUMBER);
    fprintf(stream, "cpu_has_arm_fma: %s\n", ggml_cpu_has_arm_fma() ? "true" : "false");
    fprintf(stream, "cpu_has_avx: %s\n", ggml_cpu_has_avx() ? "true" : "false");
    fprintf(stream, "cpu_has_avx2: %s\n", ggml_cpu_has_avx2() ? "true" : "false");
    fprintf(stream, "cpu_has_avx512: %s\n", ggml_cpu_has_avx512() ? "true" : "false");
    fprintf(stream, "cpu_has_avx512_vbmi: %s\n", ggml_cpu_has_avx512_vbmi() ? "true" : "false");
    fprintf(stream, "cpu_has_avx512_vnni: %s\n", ggml_cpu_has_avx512_vnni() ? "true" : "false");
    fprintf(stream, "cpu_has_blas: %s\n", ggml_cpu_has_blas() ? "true" : "false");
    fprintf(stream, "cpu_has_cublas: %s\n", ggml_cpu_has_cublas() ? "true" : "false");
    fprintf(stream, "cpu_has_clblast: %s\n", ggml_cpu_has_clblast() ? "true" : "false");
    fprintf(stream, "cpu_has_fma: %s\n", ggml_cpu_has_fma() ? "true" : "false");
    fprintf(stream, "cpu_has_gpublas: %s\n", ggml_cpu_has_gpublas() ? "true" : "false");
    fprintf(stream, "cpu_has_neon: %s\n", ggml_cpu_has_neon() ? "true" : "false");
    fprintf(stream, "cpu_has_f16c: %s\n", ggml_cpu_has_f16c() ? "true" : "false");
    fprintf(stream, "cpu_has_fp16_va: %s\n", ggml_cpu_has_fp16_va() ? "true" : "false");
    fprintf(stream, "cpu_has_wasm_simd: %s\n", ggml_cpu_has_wasm_simd() ? "true" : "false");
    fprintf(stream, "cpu_has_blas: %s\n", ggml_cpu_has_blas() ? "true" : "false");
    fprintf(stream, "cpu_has_sse3: %s\n", ggml_cpu_has_sse3() ? "true" : "false");
    fprintf(stream, "cpu_has_vsx: %s\n", ggml_cpu_has_vsx() ? "true" : "false");

#ifdef NDEBUG
    fprintf(stream, "debug: false\n");
#else
    fprintf(stream, "debug: true\n");
#endif // NDEBUG

    fprintf(stream, "model_desc: %s\n", model_desc);
    fprintf(stream, "n_vocab: %d  # output size of the final layer, 32001 for some models\n", llama_n_vocab(lctx));

#ifdef __OPTIMIZE__
    fprintf(stream, "optimize: true\n");
#else
    fprintf(stream, "optimize: false\n");
#endif // __OPTIMIZE__

    fprintf(stream, "time: %s\n", timestamp.c_str());

    fprintf(stream, "\n");
    fprintf(stream, "###############\n");
    fprintf(stream, "# User Inputs #\n");
    fprintf(stream, "###############\n");
    fprintf(stream, "\n");

    fprintf(stream, "alias: %s # default: unknown\n", params.model_alias.c_str());
    fprintf(stream, "batch_size: %d # default: 512\n", params.n_batch);
    dump_string_yaml_multiline(stream, "cfg_negative_prompt", params.cfg_negative_prompt.c_str());
    fprintf(stream, "cfg_scale: %f # default: 1.0\n", params.cfg_scale);
    fprintf(stream, "chunks: %d # default: -1 (unlimited)\n", params.n_chunks);
    fprintf(stream, "color: %s # default: false\n", params.use_color ? "true" : "false");
    fprintf(stream, "ctx_size: %d # default: 512\n", params.n_ctx);
    fprintf(stream, "escape: %s # default: false\n", params.escape ? "true" : "false");
    fprintf(stream, "export: %s # default: false\n", params.export_cgraph ? "true" : "false");
    fprintf(stream, "file: # never logged, see prompt instead. Can still be specified for input.\n");
    fprintf(stream, "frequency_penalty: %f # default: 0.0 \n", params.frequency_penalty);
    dump_string_yaml_multiline(stream, "grammar", params.grammar.c_str());
    fprintf(stream, "grammar-file: # never logged, see grammar instead. Can still be specified for input.\n");
    fprintf(stream, "hellaswag: %s # default: false\n", params.hellaswag ? "true" : "false");
    fprintf(stream, "hellaswag_tasks: %zu # default: 400\n", params.hellaswag_tasks);

    const auto logit_bias_eos = params.logit_bias.find(llama_token_eos(lctx));
    const bool ignore_eos = logit_bias_eos != params.logit_bias.end() && logit_bias_eos->second == -INFINITY;
    fprintf(stream, "ignore_eos: %s # default: false\n", ignore_eos ? "true" : "false");

    dump_string_yaml_multiline(stream, "in_prefix", params.input_prefix.c_str());
    fprintf(stream, "in_prefix_bos: %s # default: false\n", params.input_prefix_bos ? "true" : "false");
    dump_string_yaml_multiline(stream, "in_suffix", params.input_prefix.c_str());
    fprintf(stream, "instruct: %s # default: false\n", params.instruct ? "true" : "false");
    fprintf(stream, "interactive: %s # default: false\n", params.interactive ? "true" : "false");
    fprintf(stream, "interactive_first: %s # default: false\n", params.interactive_first ? "true" : "false");
    fprintf(stream, "keep: %d # default: 0\n", params.n_keep);
    fprintf(stream, "logdir: %s # default: unset (no logging)\n", params.logdir.c_str());

    fprintf(stream, "logit_bias:\n");
    for (std::pair<llama_token, float> lb : params.logit_bias) {
        if (ignore_eos && lb.first == logit_bias_eos->first) {
            continue;
        }
        fprintf(stream, "  %d: %f", lb.first, lb.second);
    }

    fprintf(stream, "lora: %s\n", params.lora_adapter.c_str());
    fprintf(stream, "lora_base: %s\n", params.lora_base.c_str());
    fprintf(stream, "low_vram: %s # default: false\n", params.low_vram ? "true" : "false");
    fprintf(stream, "main_gpu: %d # default: 0\n", params.main_gpu);
    fprintf(stream, "memory_f32: %s # default: false\n", !params.memory_f16 ? "true" : "false");
    fprintf(stream, "mirostat: %d # default: 0 (disabled)\n", params.mirostat);
    fprintf(stream, "mirostat_ent: %f # default: 5.0\n", params.mirostat_tau);
    fprintf(stream, "mirostat_lr: %f # default: 0.1\n", params.mirostat_eta);
    fprintf(stream, "mlock: %s # default: false\n", params.use_mlock ? "true" : "false");
    fprintf(stream, "model: %s # default: models/7B/ggml-model.bin\n", params.model.c_str());
    fprintf(stream, "mtest: %s # default: false\n", params.mem_test ? "true" : "false");
    fprintf(stream, "multiline_input: %s # default: false\n", params.multiline_input ? "true" : "false");
    fprintf(stream, "n_gpu_layers: %d # default: 0\n", params.n_gpu_layers);
    fprintf(stream, "n_predict: %d # default: -1 (unlimited)\n", params.n_predict);
    fprintf(stream, "n_probs: %d # only used by server binary, default: 0\n", params.n_probs);
    fprintf(stream, "no_mmap: %s # default: false\n", !params.use_mmap ? "true" : "false");
    fprintf(stream, "no_mul_mat_q: %s # default: false\n", !params.mul_mat_q ? "true" : "false");
    fprintf(stream, "no_penalize_nl: %s # default: false\n", !params.penalize_nl ? "true" : "false");
    fprintf(stream, "numa: %s # default: false\n", params.numa ? "true" : "false");
    fprintf(stream, "ppl_output_type: %d # default: 0\n", params.ppl_output_type);
    fprintf(stream, "ppl_stride: %d # default: 0\n", params.ppl_stride);
    fprintf(stream, "presence_penalty: %f # default: 0.0\n", params.presence_penalty);
    dump_string_yaml_multiline(stream, "prompt", params.prompt.c_str());
    fprintf(stream, "prompt_cache: %s\n", params.path_prompt_cache.c_str());
    fprintf(stream, "prompt_cache_all: %s # default: false\n", params.prompt_cache_all ? "true" : "false");
    fprintf(stream, "prompt_cache_ro: %s # default: false\n", params.prompt_cache_ro ? "true" : "false");
    dump_vector_int_yaml(stream, "prompt_tokens", prompt_tokens);
    fprintf(stream, "random_prompt: %s # default: false\n", params.random_prompt ? "true" : "false");
    fprintf(stream, "repeat_penalty: %f # default: 1.1\n", params.repeat_penalty);

    fprintf(stream, "reverse_prompt:\n");
    for (std::string ap : params.antiprompt) {
        size_t pos = 0;
        while ((pos = ap.find('\n', pos)) != std::string::npos) {
            ap.replace(pos, 1, "\\n");
            pos += 1;
        }

        fprintf(stream, "  - %s\n", ap.c_str());
    }

    fprintf(stream, "rope_freq_base: %f # default: 10000.0\n", params.rope_freq_base);
    fprintf(stream, "rope_freq_scale: %f # default: 1.0\n", params.rope_freq_scale);
    fprintf(stream, "seed: %d # default: -1 (random seed)\n", params.seed);
    fprintf(stream, "simple_io: %s # default: false\n", params.simple_io ? "true" : "false");
    fprintf(stream, "temp: %f # default: 0.8\n", params.temp);

    const std::vector<float> tensor_split_vector(params.tensor_split, params.tensor_split + LLAMA_MAX_DEVICES);
    dump_vector_float_yaml(stream, "tensor_split", tensor_split_vector);

    fprintf(stream, "tfs: %f # default: 1.0\n", params.tfs_z);
    fprintf(stream, "threads: %d # default: %d\n", params.n_threads, std::thread::hardware_concurrency());
    fprintf(stream, "top_k: %d # default: 40\n", params.top_k);
    fprintf(stream, "top_p: %f # default: 0.95\n", params.top_p);
    fprintf(stream, "typical_p: %f # default: 1.0\n", params.typical_p);
    fprintf(stream, "verbose_prompt: %s # default: false\n", params.verbose_prompt ? "true" : "false");
}
