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
#include <regex>

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
            params.n_batch = std::min(512, params.n_batch);
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
        } else if (arg == "--mul-mat-q" || arg == "-mmq") {
#ifdef GGML_USE_CUBLAS
            params.mul_mat_q = true;
#else
            fprintf(stderr, "warning: llama.cpp was compiled without cuBLAS. It is not possible to use mul_mat_q kernels.\n");
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
        } else if (arg == "--perplexity") {
            params.perplexity = true;
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

    if (escape_prompt) {
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
    fprintf(stdout, "  -e                    process prompt escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\)\n");
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
    fprintf(stdout, "  -mg i, --main-gpu i   the GPU to use for scratch and small tensors\n" );
    fprintf(stdout, "  -lv, --low-vram       don't allocate VRAM scratch buffer\n" );
    fprintf(stdout, "  -mmq, --mul-mat-q     use experimental mul_mat_q CUDA kernels instead of cuBLAS. TEMP!!!\n" );
    fprintf(stdout, "                        Reduces VRAM usage by 700/970/1430 MiB for 7b/13b/33b but prompt processing speed\n" );
    fprintf(stdout, "                        is still suboptimal, especially q2_K, q3_K, q5_K, and q6_K.\n" );
#endif
    fprintf(stdout, "  --mtest               compute maximum memory usage\n");
    fprintf(stdout, "  --export              export the computation graph to 'llama.ggml'\n");
    fprintf(stdout, "  --verbose-prompt      print prompt before generation\n");
    fprintf(stderr, "  --simple-io           use basic IO for better compatibility in subprocesses and limited consoles\n");
    fprintf(stdout, "  --lora FNAME          apply LoRA adapter (implies --no-mmap)\n");
    fprintf(stdout, "  --lora-base FNAME     optional model to use as a base for the layers modified by the LoRA adapter\n");
    fprintf(stdout, "  -m FNAME, --model FNAME\n");
    fprintf(stdout, "                        model path (default: %s)\n", params.model.c_str());
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

std::string llama_token_to_str(const struct llama_context * ctx, llama_token token) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_str(ctx, token, result.data(), result.size());
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_str(ctx, token, result.data(), result.size());
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

std::vector<llama_token> llama_tokenize_bpe(
        struct llama_context * ctx,
           const std::string & text,
                        bool   add_bos) {
    int n_tokens = text.length() + add_bos;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize_bpe(ctx, text.c_str(), result.data(), result.size(), add_bos);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize_bpe(ctx, text.c_str(), result.data(), result.size(), add_bos);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string llama_token_to_str_bpe(const struct llama_context * ctx, llama_token token) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_str_bpe(ctx, token, result.data(), result.size());
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        const int check = llama_token_to_str_bpe(ctx, token, result.data(), result.size());
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

