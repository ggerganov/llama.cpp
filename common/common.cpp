#if defined(_MSC_VER)
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#endif

#include "common.h"
// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"
#include "json-schema-to-grammar.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <codecvt>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <climits>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <locale>
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#if defined(LLAMA_USE_CURL)
#include <curl/curl.h>
#include <curl/easy.h>
#include <thread>
#include <future>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#if (defined(GGML_USE_CUDA) || defined(GGML_USE_SYCL))
#define GGML_USE_CUDA_SYCL
#endif

#if (defined(GGML_USE_CUDA) || defined(GGML_USE_SYCL)) || defined(GGML_USE_VULKAN)
#define GGML_USE_CUDA_SYCL_VULKAN
#endif

#if defined(LLAMA_USE_CURL)
#ifdef __linux__
#include <linux/limits.h>
#elif defined(_WIN32)
#define PATH_MAX MAX_PATH
#else
#include <sys/syslimits.h>
#endif
#define LLAMA_CURL_MAX_URL_LENGTH 2084 // Maximum URL Length in Chrome: 2083
#endif // LLAMA_USE_CURL

using json = nlohmann::ordered_json;

//
// CPU utils
//

int32_t cpu_get_num_physical_cores() {
#ifdef __linux__
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
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
#elif defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    unsigned int n_threads_win = std::thread::hardware_concurrency();
    unsigned int default_threads = n_threads_win > 0 ? (n_threads_win <= 4 ? n_threads_win : n_threads_win / 2) : 4;

    DWORD buffer_size = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size)) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            return default_threads;
        }
    }

    std::vector<char> buffer(buffer_size);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &buffer_size)) {
        return default_threads;
    }

    int32_t num_physical_cores = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    while (buffer_size > 0) {
        if (info->Relationship == RelationProcessorCore) {
            num_physical_cores += info->Processor.GroupCount;
        }
        buffer_size -= info->Size;
        info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(reinterpret_cast<char*>(info) + info->Size);
    }

    return num_physical_cores > 0 ? num_physical_cores : default_threads;
#endif
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>

static void cpuid(unsigned leaf, unsigned subleaf,
                  unsigned *eax, unsigned *ebx, unsigned *ecx, unsigned *edx) {
    __asm__("movq\t%%rbx,%%rsi\n\t"
            "cpuid\n\t"
            "xchgq\t%%rbx,%%rsi"
            : "=a"(*eax), "=S"(*ebx), "=c"(*ecx), "=d"(*edx)
            : "0"(leaf), "2"(subleaf));
}

static int pin_cpu(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

static bool is_hybrid_cpu(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return !!(edx & (1u << 15));
}

static bool is_running_on_efficiency_core(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(0x1a, 0, &eax, &ebx, &ecx, &edx);
    int intel_atom = 0x20;
    int core_type = (eax & 0xff000000u) >> 24;
    return core_type == intel_atom;
}

static int cpu_count_math_cpus(int n_cpu) {
    int result = 0;
    for (int cpu = 0; cpu < n_cpu; ++cpu) {
        if (pin_cpu(cpu)) {
            return -1;
        }
        if (is_running_on_efficiency_core()) {
            continue; // efficiency cores harm lockstep threading
        }
        ++cpu; // hyperthreading isn't useful for linear algebra
        ++result;
    }
    return result;
}

#endif // __x86_64__ && __linux__

/**
 * Returns number of CPUs on system that are useful for math.
 */
int32_t cpu_get_num_math() {
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
    int n_cpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cpu < 1) {
        return cpu_get_num_physical_cores();
    }
    if (is_hybrid_cpu()) {
        cpu_set_t affinity;
        if (!pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity)) {
            int result = cpu_count_math_cpus(n_cpu);
            pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
            if (result > 0) {
                return result;
            }
        }
    }
#endif
    return cpu_get_num_physical_cores();
}

// Helper for setting process priority

#if defined(_WIN32)

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    DWORD p = NORMAL_PRIORITY_CLASS;
    switch (prio) {
        case GGML_SCHED_PRIO_NORMAL:   p = NORMAL_PRIORITY_CLASS;       break;
        case GGML_SCHED_PRIO_MEDIUM:   p = ABOVE_NORMAL_PRIORITY_CLASS; break;
        case GGML_SCHED_PRIO_HIGH:     p = HIGH_PRIORITY_CLASS;         break;
        case GGML_SCHED_PRIO_REALTIME: p = REALTIME_PRIORITY_CLASS;     break;
    }

    if (!SetPriorityClass(GetCurrentProcess(), p)) {
        fprintf(stderr, "warn: failed to set process priority class %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#else // MacOS and POSIX
#include <sys/types.h>
#include <sys/resource.h>

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    int p = 0;
    switch (prio) {
        case GGML_SCHED_PRIO_NORMAL:   p =  0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   p = -5;  break;
        case GGML_SCHED_PRIO_HIGH:     p = -10; break;
        case GGML_SCHED_PRIO_REALTIME: p = -20; break;
    }

    if (!setpriority(PRIO_PROCESS, 0, p)) {
        fprintf(stderr, "warn: failed to set process priority %d : %s (%d)\n", prio, strerror(errno), errno);
        return false;
    }
    return true;
}

#endif

//
// CLI argument parsing
//

#ifdef __GNUC__
#ifdef __MINGW32__
#define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define LLAMA_COMMON_ATTRIBUTE_FORMAT(...)
#endif

LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

static void gpt_params_handle_model_default(gpt_params & params) {
    if (!params.hf_repo.empty()) {
        // short-hand to avoid specifying --hf-file -> default it to --model
        if (params.hf_file.empty()) {
            if (params.model.empty()) {
                throw std::invalid_argument("error: --hf-repo requires either --hf-file or --model\n");
            }
            params.hf_file = params.model;
        } else if (params.model.empty()) {
            params.model = fs_get_cache_file(string_split(params.hf_file, '/').back());
        }
    } else if (!params.model_url.empty()) {
        if (params.model.empty()) {
            auto f = string_split(params.model_url, '#').front();
            f = string_split(f, '?').front();
            params.model = fs_get_cache_file(string_split(f, '/').back());
        }
    } else if (params.model.empty()) {
        params.model = DEFAULT_MODEL_PATH;
    }
}

void postprocess_cpu_params(cpu_params& cpuparams, const cpu_params* role_model) {
    int32_t n_set = 0;

    if (cpuparams.n_threads < 0) {
        // Assuming everything about cpuparams is invalid
        if (role_model != nullptr) {
            cpuparams = *role_model;
        } else {
            cpuparams.n_threads = cpu_get_num_math();
        }
    }

    for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (cpuparams.cpumask[i]) {
            n_set++;
        }
    }

    if (n_set && n_set < cpuparams.n_threads) {
        // Not enough set bits, may experience performance issues.
        fprintf(stderr, "warn: Not enough set bits in CPU mask (%d) to satisfy requested thread count: %d\n", n_set, cpuparams.n_threads);
    }
}

bool gpt_params_parse_ex(int argc, char ** argv, gpt_params & params, std::vector<llama_arg> & options) {
    std::string arg;
    const std::string arg_prefix = "--";
    gpt_sampler_params & sparams = params.sparams;

    std::unordered_map<std::string, llama_arg *> arg_to_options;
    for (auto & opt : options) {
        for (const auto & arg : opt.args) {
            arg_to_options[arg] = &opt;
        }
    }

    // handle environment variables
    for (auto & opt : options) {
        std::string value;
        if (opt.get_value_from_env(value)) {
            try {
                if (opt.handler_void && (value == "1" || value == "true")) {
                    opt.handler_void(params);
                }
                if (opt.handler_int) {
                    opt.handler_int(params, std::stoi(value));
                }
                if (opt.handler_string) {
                    opt.handler_string(params, value);
                    continue;
                }
            } catch (std::exception & e) {
                throw std::invalid_argument(format(
                    "error while handling environment variable \"%s\": %s\n\n", opt.env, e.what()));
            }
        }
    }

    // handle command line arguments
    auto check_arg = [&](int i) {
        if (i+1 >= argc) {
            throw std::invalid_argument("expected value for argument");
        }
    };

    for (int i = 1; i < argc; i++) {
        const std::string arg_prefix = "--";

        std::string arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }
        if (arg_to_options.find(arg) == arg_to_options.end()) {
            throw std::invalid_argument(format("error: invalid argument: %s", arg.c_str()));
        }
        auto opt = *arg_to_options[arg];
        if (opt.has_value_from_env()) {
            fprintf(stderr, "warn: %s environment variable is set, but will be overwritten by command line argument %s\n", opt.env, arg.c_str());
        }
        try {
            if (opt.handler_void) {
                opt.handler_void(params);
                continue;
            }

            // arg with single value
            check_arg(i);
            std::string val = argv[++i];
            if (opt.handler_int) {
                opt.handler_int(params, std::stoi(val));
                continue;
            }
            if (opt.handler_string) {
                opt.handler_string(params, val);
                continue;
            }

            // arg with 2 values
            check_arg(i);
            std::string val2 = argv[++i];
            if (opt.handler_str_str) {
                opt.handler_str_str(params, val, val2);
                continue;
            }
        } catch (std::exception & e) {
            throw std::invalid_argument(format(
                "error while handling argument \"%s\": %s\n\n"
                "usage:\n%s\n\nto show complete usage, run with -h",
                arg.c_str(), e.what(), arg_to_options[arg]->to_string().c_str()));
        }
    }

    postprocess_cpu_params(params.cpuparams, nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);
    postprocess_cpu_params(params.draft_cpuparams, &params.cpuparams);
    postprocess_cpu_params(params.draft_cpuparams_batch, &params.cpuparams_batch);

    if (params.prompt_cache_all && (params.interactive || params.interactive_first)) {
        throw std::invalid_argument("error: --prompt-cache-all not supported in interactive mode yet\n");
    }

    gpt_params_handle_model_default(params);

    if (params.escape) {
        string_process_escapes(params.prompt);
        string_process_escapes(params.input_prefix);
        string_process_escapes(params.input_suffix);
        for (auto & antiprompt : params.antiprompt) {
            string_process_escapes(antiprompt);
        }
    }

    if (!params.kv_overrides.empty()) {
        params.kv_overrides.emplace_back();
        params.kv_overrides.back().key[0] = 0;
    }

    if (sparams.seed == LLAMA_DEFAULT_SEED) {
        sparams.seed = time(NULL);
    }

    return true;
}

bool gpt_params_parse(int argc, char ** argv, gpt_params & params, std::vector<llama_arg> & options) {
    const auto params_org = params; // the example can modify the default params

    try {
        if (!gpt_params_parse_ex(argc, argv, params, options)) {
            params = params_org;
            return false;
        }
        if (params.usage) {
            gpt_params_print_usage(params, options);
            if (params.print_usage) {
                params.print_usage(argc, argv);
            }
            exit(0);
        }
    } catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        params = params_org;
        return false;
    }

    return true;
}

bool parse_cpu_range(const std::string & range, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    size_t dash_loc = range.find('-');
    if (dash_loc == std::string::npos) {
        fprintf(stderr, "Format of CPU range is invalid! Expected [<start>]-[<end>].\n");
        return false;
    }

    size_t start_i;
    size_t end_i;

    if (dash_loc == 0) {
        start_i = 0;
    } else {
        start_i = std::stoull(range.substr(0, dash_loc));
        if (start_i >= GGML_MAX_N_THREADS) {
            fprintf(stderr, "Start index out of bounds!\n");
            return false;
        }
    }

    if (dash_loc == range.length() - 1) {
        end_i = GGML_MAX_N_THREADS - 1;
    } else {
        end_i = std::stoull(range.substr(dash_loc + 1));
        if (end_i >= GGML_MAX_N_THREADS) {
            fprintf(stderr, "End index out of bounds!\n");
            return false;
        }
    }

    for (size_t i = start_i; i <= end_i; i++) {
        boolmask[i] = true;
    }

    return true;
}

bool parse_cpu_mask(const std::string & mask, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    // Discard potential 0x prefix
    size_t start_i = 0;
    if (mask.length() >= 2 && mask.substr(0, 2) == "0x") {
        start_i = 2;
    }

    size_t num_digits = mask.length() - start_i;
    if (num_digits > 128) num_digits = 128;

    size_t end_i = num_digits + start_i;

    for (size_t i = start_i, n = (num_digits*4 - 1); i < end_i; i++, n-=4) {
        char c = mask.at(i);
        int8_t id = c;

        if ((c >= '0' && c <= '9')) {
            id -= '0';
        } else if (c >= 'a' && c <= 'f') {
            id -= 'a' - 10;
        } else if (c >= 'A' && c <= 'F') {
            id -= 'A' - 10;
        } else {
            fprintf(stderr, "Invalid hex character '%c' at position %d\n", c, int32_t(i));
            return false;
        }

        boolmask[  n  ] = boolmask[  n  ] || ((id & 8) != 0);
        boolmask[n - 1] = boolmask[n - 1] || ((id & 4) != 0);
        boolmask[n - 2] = boolmask[n - 2] || ((id & 2) != 0);
        boolmask[n - 3] = boolmask[n - 3] || ((id & 1) != 0);
    }

    return true;
}

static std::vector<std::string> break_str_into_lines(std::string input, size_t max_char_per_line) {
    std::vector<std::string> result;
    std::istringstream iss(input);
    std::string line;
    auto add_line = [&](const std::string& l) {
        if (l.length() <= max_char_per_line) {
            result.push_back(l);
        } else {
            std::istringstream line_stream(l);
            std::string word, current_line;
            while (line_stream >> word) {
                if (current_line.length() + !current_line.empty() + word.length() > max_char_per_line) {
                    if (!current_line.empty()) result.push_back(current_line);
                    current_line = word;
                } else {
                    current_line += (!current_line.empty() ? " " : "") + word;
                }
            }
            if (!current_line.empty()) result.push_back(current_line);
        }
    };
    while (std::getline(iss, line)) {
        add_line(line);
    }
    return result;
}

std::string llama_arg::to_string() {
    // params for printing to console
    const static int n_leading_spaces = 40;
    const static int n_char_per_line_help = 70; // TODO: detect this based on current console
    std::string leading_spaces(n_leading_spaces, ' ');

    std::ostringstream ss;
    for (const auto arg : args) {
        if (arg == args.front()) {
            if (args.size() == 1) {
                ss << arg;
            } else {
                ss << format("%-7s", arg) << ", ";
            }
        } else {
            ss << arg << (arg != args.back() ? ", " : "");
        }
    }
    if (value_hint) ss << " " << value_hint;
    if (value_hint_2) ss << " " << value_hint_2;
    if (ss.tellp() > n_leading_spaces - 3) {
        // current line is too long, add new line
        ss << "\n" << leading_spaces;
    } else {
        // padding between arg and help, same line
        ss << std::string(leading_spaces.size() - ss.tellp(), ' ');
    }
    const auto help_lines = break_str_into_lines(help, n_char_per_line_help);
    for (const auto & line : help_lines) {
        ss << (&line == &help_lines.front() ? "" : leading_spaces) << line << "\n";
    }
    return ss.str();
}

void gpt_params_print_usage(gpt_params & params, std::vector<llama_arg> & options) {
    auto print_options = [](std::vector<llama_arg *> & options) {
        for (llama_arg * opt : options) {
            printf("%s", opt->to_string().c_str());
        }
    };

    std::vector<llama_arg *> common_options;
    std::vector<llama_arg *> specific_options;
    for (auto & opt : options) {
        // in case multiple LLAMA_EXAMPLE_* are set, we prioritize the LLAMA_EXAMPLE_* matching current example
        if (opt.in_example(params.curr_ex)) {
            specific_options.push_back(&opt);
        } else {
            common_options.push_back(&opt);
        }
    }
    printf("----- common options -----\n\n");
    print_options(common_options);
    // TODO: maybe convert enum llama_example to string
    printf("\n\n----- example-specific options -----\n\n");
    print_options(specific_options);
}

std::vector<llama_arg> gpt_params_parser_init(gpt_params & params, llama_example ex) {
    return gpt_params_parser_init(params, ex, nullptr);
}

std::vector<llama_arg> gpt_params_parser_init(gpt_params & params, llama_example ex, std::function<void(int, char **)> print_usage) {
    std::vector<llama_arg> options;
    params.print_usage = print_usage;
    params.curr_ex     = ex;

    std::string sampler_type_chars;
    std::string sampler_type_names;
    for (const auto & sampler : params.sparams.samplers) {
        sampler_type_chars += gpt_sampler_type_to_chr(sampler);
        sampler_type_names += gpt_sampler_type_to_str(sampler) + ";";
    }
    sampler_type_names.pop_back();


    /**
     * filter options by example
     * rules:
     * - all examples inherit options from LLAMA_EXAMPLE_COMMON
     * - if LLAMA_EXAMPLE_* is set (other than COMMON), we only show the option in the corresponding example
     * - if both {LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_*,} are set, we will prioritize the LLAMA_EXAMPLE_* matching current example
     */
    std::unordered_set<std::string> seen_args;
    auto add_opt = [&](llama_arg arg) {
        if (arg.in_example(ex) || arg.in_example(LLAMA_EXAMPLE_COMMON)) {
            // make sure there is no argument duplications
            for (const auto & a : arg.args) {
                if (seen_args.find(a) == seen_args.end()) {
                    seen_args.insert(a);
                } else {
                    throw std::runtime_error(format("found duplicated argument in source code: %s", a));
                }
            }
            options.push_back(std::move(arg));
        }
    };


    add_opt(llama_arg(
        {"-h", "--help", "--usage"},
        "print usage and exit",
        [](gpt_params & params) {
            params.usage = true;
        }
    ));
    add_opt(llama_arg(
        {"--version"},
        "show version and build info",
        [](gpt_params &) {
            fprintf(stderr, "version: %d (%s)\n", LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
            fprintf(stderr, "built with %s for %s\n", LLAMA_COMPILER, LLAMA_BUILD_TARGET);
            exit(0);
        }
    ));
    add_opt(llama_arg(
        {"-v", "--verbose"},
        "print verbose information",
        [](gpt_params & params) {
            params.verbosity = 1;
        }
    ));
    add_opt(llama_arg(
        {"--verbosity"}, "N",
        format("set specific verbosity level (default: %d)", params.verbosity),
        [](gpt_params & params, int value) {
            params.verbosity = value;
        }
    ));
    add_opt(llama_arg(
        {"--verbose-prompt"},
        format("print a verbose prompt before generation (default: %s)", params.verbose_prompt ? "true" : "false"),
        [](gpt_params & params) {
            params.verbose_prompt = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"--no-display-prompt"},
        format("don't print prompt at generation (default: %s)", !params.display_prompt ? "true" : "false"),
        [](gpt_params & params) {
            params.display_prompt = false;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"-co", "--color"},
        format("colorise output to distinguish prompt and user input from generations (default: %s)", params.use_color ? "true" : "false"),
        [](gpt_params & params) {
            params.use_color = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_INFILL}));
    add_opt(llama_arg(
        {"-s", "--seed"}, "SEED",
        format("RNG seed (default: %d, use random seed for < 0)", params.sparams.seed),
        [](gpt_params & params, const std::string & value) {
            params.sparams.seed = std::stoul(value);
        }
    ));
    add_opt(llama_arg(
        {"-t", "--threads"}, "N",
        format("number of threads to use during generation (default: %d)", params.cpuparams.n_threads),
        [](gpt_params & params, int value) {
            params.cpuparams.n_threads = value;
            if (params.cpuparams.n_threads <= 0) {
                params.cpuparams.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_env("LLAMA_ARG_THREADS"));
    add_opt(llama_arg(
        {"-tb", "--threads-batch"}, "N",
        "number of threads to use during batch and prompt processing (default: same as --threads)",
        [](gpt_params & params, int value) {
            params.cpuparams_batch.n_threads = value;
            if (params.cpuparams_batch.n_threads <= 0) {
                params.cpuparams_batch.n_threads = std::thread::hardware_concurrency();
            }
        }
    ));
    add_opt(llama_arg(
        {"-td", "--threads-draft"}, "N",
        "number of threads to use during generation (default: same as --threads)",
        [](gpt_params & params, int value) {
            params.draft_cpuparams.n_threads = value;
            if (params.draft_cpuparams.n_threads <= 0) {
                params.draft_cpuparams.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"-tbd", "--threads-batch-draft"}, "N",
        "number of threads to use during batch and prompt processing (default: same as --threads-draft)",
        [](gpt_params & params, int value) {
            params.draft_cpuparams_batch.n_threads = value;
            if (params.draft_cpuparams_batch.n_threads <= 0) {
                params.draft_cpuparams_batch.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"-C", "--cpu-mask"}, "M",
        "CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: \"\")",
        [](gpt_params & params, const std::string & value) {
            std::string mask = value;
            params.cpuparams.mask_valid = true;
            if (!parse_cpu_mask(mask, params.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ));
    add_opt(llama_arg(
        {"-Cr", "--cpu-range"}, "lo-hi",
        "range of CPUs for affinity. Complements --cpu-mask",
        [](gpt_params & params, const std::string & value) {
            std::string range = value;
            params.cpuparams.mask_valid = true;
            if (!parse_cpu_range(range, params.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ));
    add_opt(llama_arg(
        {"--cpu-strict"}, "<0|1>",
        format("use strict CPU placement (default: %u)\n", (unsigned) params.cpuparams.strict_cpu),
        [](gpt_params & params, const std::string & value) {
            params.cpuparams.strict_cpu = std::stoul(value);
        }
    ));
    add_opt(llama_arg(
        {"--poll"}, "<0...100>",
        format("use polling level to wait for work (0 - no polling, default: %u)\n", (unsigned) params.cpuparams.poll),
        [](gpt_params & params, const std::string & value) {
            params.cpuparams.poll = std::stoul(value);
        }
    ));
    add_opt(llama_arg(
        {"-Cb", "--cpu-mask-batch"}, "M",
        "CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask)",
        [](gpt_params & params, const std::string & value) {
            std::string mask = value;
            params.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_mask(mask, params.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ));
    add_opt(llama_arg(
        {"-Crb", "--cpu-range-batch"}, "lo-hi",
        "ranges of CPUs for affinity. Complements --cpu-mask-batch",
        [](gpt_params & params, const std::string & value) {
            std::string range = value;
            params.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_range(range, params.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ));
    add_opt(llama_arg(
        {"--cpu-strict-batch"}, "<0|1>",
        "use strict CPU placement (default: same as --cpu-strict)",
        [](gpt_params & params, int value) {
            params.cpuparams_batch.strict_cpu = value;
        }
    ));
    add_opt(llama_arg(
        {"--poll-batch"}, "<0|1>",
        "use polling to wait for work (default: same as --poll)",
        [](gpt_params & params, int value) {
            params.cpuparams_batch.poll = value;
        }
    ));
    add_opt(llama_arg(
        {"-Cd", "--cpu-mask-draft"}, "M",
        "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)",
        [](gpt_params & params, const std::string & value) {
            std::string mask = value;
            params.draft_cpuparams.mask_valid = true;
            if (!parse_cpu_mask(mask, params.draft_cpuparams.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"-Crd", "--cpu-range-draft"}, "lo-hi",
        "Ranges of CPUs for affinity. Complements --cpu-mask-draft",
        [](gpt_params & params, const std::string & value) {
            std::string range = value;
            params.draft_cpuparams.mask_valid = true;
            if (!parse_cpu_range(range, params.draft_cpuparams.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"--cpu-strict-draft"}, "<0|1>",
        "Use strict CPU placement for draft model (default: same as --cpu-strict)",
        [](gpt_params & params, int value) {
            params.draft_cpuparams.strict_cpu = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"--poll-draft"}, "<0|1>",
        "Use polling to wait for draft model work (default: same as --poll])",
        [](gpt_params & params, int value) {
            params.draft_cpuparams.poll = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"-Crbd", "--cpu-range-batch-draft"}, "lo-hi",
        "Ranges of CPUs for affinity. Complements --cpu-mask-draft-batch)",
        [](gpt_params & params, const std::string & value) {
            std::string range = value;
            params.draft_cpuparams_batch.mask_valid = true;
            if (!parse_cpu_range(range, params.draft_cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"--cpu-strict-batch-draft"}, "<0|1>",
        "Use strict CPU placement for draft model (default: --cpu-strict-draft)",
        [](gpt_params & params, int value) {
            params.draft_cpuparams_batch.strict_cpu = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"--poll-batch-draft"}, "<0|1>",
        "Use polling to wait for draft model work (default: --poll-draft)",
        [](gpt_params & params, int value) {
            params.draft_cpuparams_batch.poll = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"--draft"}, "N",
        format("number of tokens to draft for speculative decoding (default: %d)", params.n_draft),
        [](gpt_params & params, int value) {
            params.n_draft = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"-ps", "--p-split"}, "N",
        format("speculative decoding split probability (default: %.1f)", (double)params.p_split),
        [](gpt_params & params, const std::string & value) {
            params.p_split = std::stof(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"-lcs", "--lookup-cache-static"}, "FNAME",
        "path to static lookup cache to use for lookup decoding (not updated by generation)",
        [](gpt_params & params, const std::string & value) {
            params.lookup_cache_static = value;
        }
    ));
    add_opt(llama_arg(
        {"-lcd", "--lookup-cache-dynamic"}, "FNAME",
        "path to dynamic lookup cache to use for lookup decoding (updated by generation)",
        [](gpt_params & params, const std::string & value) {
            params.lookup_cache_dynamic = value;
        }
    ));
    add_opt(llama_arg(
        {"-c", "--ctx-size"}, "N",
        format("size of the prompt context (default: %d, 0 = loaded from model)", params.n_ctx),
        [](gpt_params & params, int value) {
            params.n_ctx = value;
        }
    ).set_env("LLAMA_ARG_CTX_SIZE"));
    add_opt(llama_arg(
        {"-n", "--predict", "--n-predict"}, "N",
        format("number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)", params.n_predict),
        [](gpt_params & params, int value) {
            params.n_predict = value;
        }
    ).set_env("LLAMA_ARG_N_PREDICT"));
    add_opt(llama_arg(
        {"-b", "--batch-size"}, "N",
        format("logical maximum batch size (default: %d)", params.n_batch),
        [](gpt_params & params, int value) {
            params.n_batch = value;
        }
    ).set_env("LLAMA_ARG_BATCH"));
    add_opt(llama_arg(
        {"-ub", "--ubatch-size"}, "N",
        format("physical maximum batch size (default: %d)", params.n_ubatch),
        [](gpt_params & params, int value) {
            params.n_ubatch = value;
        }
    ).set_env("LLAMA_ARG_UBATCH"));
    add_opt(llama_arg(
        {"--keep"}, "N",
        format("number of tokens to keep from the initial prompt (default: %d, -1 = all)", params.n_keep),
        [](gpt_params & params, int value) {
            params.n_keep = value;
        }
    ));
    add_opt(llama_arg(
        {"--chunks"}, "N",
        format("max number of chunks to process (default: %d, -1 = all)", params.n_chunks),
        [](gpt_params & params, int value) {
            params.n_chunks = value;
        }
    ));
    add_opt(llama_arg(
        {"-fa", "--flash-attn"},
        format("enable Flash Attention (default: %s)", params.flash_attn ? "enabled" : "disabled"),
        [](gpt_params & params) {
            params.flash_attn = true;
        }
    ).set_env("LLAMA_ARG_FLASH_ATTN"));
    add_opt(llama_arg(
        {"-p", "--prompt"}, "PROMPT",
        ex == LLAMA_EXAMPLE_MAIN
            ? "prompt to start generation with\nif -cnv is set, this will be used as system prompt"
            : "prompt to start generation with",
        [](gpt_params & params, const std::string & value) {
            params.prompt = value;
        }
    ));
    add_opt(llama_arg(
        {"-f", "--file"}, "FNAME",
        "a file containing the prompt (default: none)",
        [](gpt_params & params, const std::string & value) {
            std::ifstream file(value);
            if (!file) {
                throw std::runtime_error(format("error: failed to open file '%s'\n", value.c_str()));
            }
            // store the external file name in params
            params.prompt_file = value;
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
            if (!params.prompt.empty() && params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        }
    ));
    add_opt(llama_arg(
        {"--in-file"}, "FNAME",
        "an input file (repeat to specify multiple files)",
        [](gpt_params & params, const std::string & value) {
            std::ifstream file(value);
            if (!file) {
                throw std::runtime_error(format("error: failed to open file '%s'\n", value.c_str()));
            }
            params.in_files.push_back(value);
        }
    ));
    add_opt(llama_arg(
        {"-bf", "--binary-file"}, "FNAME",
        "binary file containing the prompt (default: none)",
        [](gpt_params & params, const std::string & value) {
            std::ifstream file(value, std::ios::binary);
            if (!file) {
                throw std::runtime_error(format("error: failed to open file '%s'\n", value.c_str()));
            }
            // store the external file name in params
            params.prompt_file = value;
            std::ostringstream ss;
            ss << file.rdbuf();
            params.prompt = ss.str();
            fprintf(stderr, "Read %zu bytes from binary file %s\n", params.prompt.size(), value.c_str());
        }
    ));
    add_opt(llama_arg(
        {"-e", "--escape"},
        format("process escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\) (default: %s)", params.escape ? "true" : "false"),
        [](gpt_params & params) {
            params.escape = true;
        }
    ));
    add_opt(llama_arg(
        {"--no-escape"},
        "do not process escape sequences",
        [](gpt_params & params) {
            params.escape = false;
        }
    ));
    add_opt(llama_arg(
        {"-ptc", "--print-token-count"}, "N",
        format("print token count every N tokens (default: %d)", params.n_print),
        [](gpt_params & params, int value) {
            params.n_print = value;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"--prompt-cache"}, "FNAME",
        "file to cache prompt state for faster startup (default: none)",
        [](gpt_params & params, const std::string & value) {
            params.path_prompt_cache = value;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"--prompt-cache-all"},
        "if specified, saves user input and generations to cache as well\n",
        [](gpt_params & params) {
            params.prompt_cache_all = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"--prompt-cache-ro"},
        "if specified, uses the prompt cache but does not update it",
        [](gpt_params & params) {
            params.prompt_cache_ro = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"-r", "--reverse-prompt"}, "PROMPT",
        "halt generation at PROMPT, return control in interactive mode\n",
        [](gpt_params & params, const std::string & value) {
            params.antiprompt.emplace_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"-sp", "--special"},
        format("special tokens output enabled (default: %s)", params.special ? "true" : "false"),
        [](gpt_params & params) {
            params.special = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"-cnv", "--conversation"},
        format(
            "run in conversation mode:\n"
            "- does not print special tokens and suffix/prefix\n"
            "- interactive mode is also enabled\n"
            "(default: %s)",
            params.conversation ? "true" : "false"
        ),
        [](gpt_params & params) {
            params.conversation = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"-i", "--interactive"},
        format("run in interactive mode (default: %s)", params.interactive ? "true" : "false"),
        [](gpt_params & params) {
            params.interactive = true;
        }
    ).set_examples({LLAMA_EXAMPLE_INFILL}));
    add_opt(llama_arg(
        {"-if", "--interactive-first"},
        format("run in interactive mode and wait for input right away (default: %s)", params.interactive_first ? "true" : "false"),
        [](gpt_params & params) {
            params.interactive_first = true;
        }
    ).set_examples({LLAMA_EXAMPLE_INFILL}));
    add_opt(llama_arg(
        {"-mli", "--multiline-input"},
        "allows you to write or paste multiple lines without ending each in '\\'",
        [](gpt_params & params) {
            params.multiline_input = true;
        }
    ).set_examples({LLAMA_EXAMPLE_INFILL}));
    add_opt(llama_arg(
        {"--in-prefix-bos"},
        "prefix BOS to user inputs, preceding the `--in-prefix` string",
        [](gpt_params & params) {
            params.input_prefix_bos = true;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_INFILL}));
    add_opt(llama_arg(
        {"--in-prefix"}, "STRING",
        "string to prefix user inputs with (default: empty)",
        [](gpt_params & params, const std::string & value) {
            params.input_prefix = value;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_INFILL}));
    add_opt(llama_arg(
        {"--in-suffix"}, "STRING",
        "string to suffix after user inputs with (default: empty)",
        [](gpt_params & params, const std::string & value) {
            params.input_suffix = value;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_INFILL}));
    add_opt(llama_arg(
        {"--no-warmup"},
        "skip warming up the model with an empty run",
        [](gpt_params & params) {
            params.warmup = false;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"--spm-infill"},
        format(
            "use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this. (default: %s)",
            params.spm_infill ? "enabled" : "disabled"
        ),
        [](gpt_params & params) {
            params.spm_infill = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_INFILL}));
    add_opt(llama_arg(
        {"--samplers"}, "SAMPLERS",
        format("samplers that will be used for generation in the order, separated by \';\'\n(default: %s)", sampler_type_names.c_str()),
        [](gpt_params & params, const std::string & value) {
            const auto sampler_names = string_split(value, ';');
            params.sparams.samplers = gpt_sampler_types_from_names(sampler_names, true);
        }
    ));
    add_opt(llama_arg(
        {"--sampling-seq"}, "SEQUENCE",
        format("simplified sequence for samplers that will be used (default: %s)", sampler_type_chars.c_str()),
        [](gpt_params & params, const std::string & value) {
            params.sparams.samplers = gpt_sampler_types_from_chars(value);
        }
    ));
    add_opt(llama_arg(
        {"--ignore-eos"},
        "ignore end of stream token and continue generating (implies --logit-bias EOS-inf)",
        [](gpt_params & params) {
            params.sparams.ignore_eos = true;
        }
    ));
    add_opt(llama_arg(
        {"--penalize-nl"},
        format("penalize newline tokens (default: %s)", params.sparams.penalize_nl ? "true" : "false"),
        [](gpt_params & params) {
            params.sparams.penalize_nl = true;
        }
    ));
    add_opt(llama_arg(
        {"--temp"}, "N",
        format("temperature (default: %.1f)", (double)params.sparams.temp),
        [](gpt_params & params, const std::string & value) {
            params.sparams.temp = std::stof(value);
            params.sparams.temp = std::max(params.sparams.temp, 0.0f);
        }
    ));
    add_opt(llama_arg(
        {"--top-k"}, "N",
        format("top-k sampling (default: %d, 0 = disabled)", params.sparams.top_k),
        [](gpt_params & params, int value) {
            params.sparams.top_k = value;
        }
    ));
    add_opt(llama_arg(
        {"--top-p"}, "N",
        format("top-p sampling (default: %.1f, 1.0 = disabled)", (double)params.sparams.top_p),
        [](gpt_params & params, const std::string & value) {
            params.sparams.top_p = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--min-p"}, "N",
        format("min-p sampling (default: %.1f, 0.0 = disabled)", (double)params.sparams.min_p),
        [](gpt_params & params, const std::string & value) {
            params.sparams.min_p = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--tfs"}, "N",
        format("tail free sampling, parameter z (default: %.1f, 1.0 = disabled)", (double)params.sparams.tfs_z),
        [](gpt_params & params, const std::string & value) {
            params.sparams.tfs_z = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--typical"}, "N",
        format("locally typical sampling, parameter p (default: %.1f, 1.0 = disabled)", (double)params.sparams.typ_p),
        [](gpt_params & params, const std::string & value) {
            params.sparams.typ_p = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--repeat-last-n"}, "N",
        format("last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)", params.sparams.penalty_last_n),
        [](gpt_params & params, int value) {
            params.sparams.penalty_last_n = value;
            params.sparams.n_prev = std::max(params.sparams.n_prev, params.sparams.penalty_last_n);
        }
    ));
    add_opt(llama_arg(
        {"--repeat-penalty"}, "N",
        format("penalize repeat sequence of tokens (default: %.1f, 1.0 = disabled)", (double)params.sparams.penalty_repeat),
        [](gpt_params & params, const std::string & value) {
            params.sparams.penalty_repeat = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--presence-penalty"}, "N",
        format("repeat alpha presence penalty (default: %.1f, 0.0 = disabled)", (double)params.sparams.penalty_present),
        [](gpt_params & params, const std::string & value) {
            params.sparams.penalty_present = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--frequency-penalty"}, "N",
        format("repeat alpha frequency penalty (default: %.1f, 0.0 = disabled)", (double)params.sparams.penalty_freq),
        [](gpt_params & params, const std::string & value) {
            params.sparams.penalty_freq = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--dynatemp-range"}, "N",
        format("dynamic temperature range (default: %.1f, 0.0 = disabled)", (double)params.sparams.dynatemp_range),
        [](gpt_params & params, const std::string & value) {
            params.sparams.dynatemp_range = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--dynatemp-exp"}, "N",
        format("dynamic temperature exponent (default: %.1f)", (double)params.sparams.dynatemp_exponent),
        [](gpt_params & params, const std::string & value) {
            params.sparams.dynatemp_exponent = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--mirostat"}, "N",
        format("use Mirostat sampling.\nTop K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.\n"
        "(default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)", params.sparams.mirostat),
        [](gpt_params & params, int value) {
            params.sparams.mirostat = value;
        }
    ));
    add_opt(llama_arg(
        {"--mirostat-lr"}, "N",
        format("Mirostat learning rate, parameter eta (default: %.1f)", (double)params.sparams.mirostat_eta),
        [](gpt_params & params, const std::string & value) {
            params.sparams.mirostat_eta = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--mirostat-ent"}, "N",
        format("Mirostat target entropy, parameter tau (default: %.1f)", (double)params.sparams.mirostat_tau),
        [](gpt_params & params, const std::string & value) {
            params.sparams.mirostat_tau = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"-l", "--logit-bias"}, "TOKEN_ID(+/-)BIAS",
        "modifies the likelihood of token appearing in the completion,\n"
        "i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n"
        "or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'",
        [](gpt_params & params, const std::string & value) {
            std::stringstream ss(value);
            llama_token key;
            char sign;
            std::string value_str;
            try {
                if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
                    const float bias = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
                    params.sparams.logit_bias.push_back({key, bias});
                } else {
                    throw std::invalid_argument("invalid input format");
                }
            } catch (const std::exception&) {
                throw std::invalid_argument("invalid input format");
            }
        }
    ));
    add_opt(llama_arg(
        {"--grammar"}, "GRAMMAR",
        format("BNF-like grammar to constrain generations (see samples in grammars/ dir) (default: '%s')", params.sparams.grammar.c_str()),
        [](gpt_params & params, const std::string & value) {
            params.sparams.grammar = value;
        }
    ));
    add_opt(llama_arg(
        {"--grammar-file"}, "FNAME",
        "file to read grammar from",
        [](gpt_params & params, const std::string & value) {
            std::ifstream file(value);
            if (!file) {
                throw std::runtime_error(format("error: failed to open file '%s'\n", value.c_str()));
            }
            std::copy(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(),
                std::back_inserter(params.sparams.grammar)
            );
        }
    ));
    add_opt(llama_arg(
        {"-j", "--json-schema"}, "SCHEMA",
        "JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object\nFor schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead",
        [](gpt_params & params, const std::string & value) {
            params.sparams.grammar = json_schema_to_grammar(json::parse(value));
        }
    ));
    add_opt(llama_arg(
        {"--pooling"}, "{none,mean,cls,last}",
        "pooling type for embeddings, use model default if unspecified",
        [](gpt_params & params, const std::string & value) {
            /**/ if (value == "none") { params.pooling_type = LLAMA_POOLING_TYPE_NONE; }
            else if (value == "mean") { params.pooling_type = LLAMA_POOLING_TYPE_MEAN; }
            else if (value == "cls") { params.pooling_type = LLAMA_POOLING_TYPE_CLS; }
            else if (value == "last") { params.pooling_type = LLAMA_POOLING_TYPE_LAST; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(llama_arg(
        {"--attention"}, "{causal,non,causal}",
        "attention type for embeddings, use model default if unspecified",
        [](gpt_params & params, const std::string & value) {
            /**/ if (value == "causal") { params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL; }
            else if (value == "non-causal") { params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(llama_arg(
        {"--rope-scaling"}, "{none,linear,yarn}",
        "RoPE frequency scaling method, defaults to linear unless specified by the model",
        [](gpt_params & params, const std::string & value) {
            /**/ if (value == "none") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE; }
            else if (value == "linear") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR; }
            else if (value == "yarn") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ));
    add_opt(llama_arg(
        {"--rope-scale"}, "N",
        "RoPE context scaling factor, expands context by a factor of N",
        [](gpt_params & params, const std::string & value) {
            params.rope_freq_scale = 1.0f / std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--rope-freq-base"}, "N",
        "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)",
        [](gpt_params & params, const std::string & value) {
            params.rope_freq_base = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--rope-freq-scale"}, "N",
        "RoPE frequency scaling factor, expands context by a factor of 1/N",
        [](gpt_params & params, const std::string & value) {
            params.rope_freq_scale = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--yarn-orig-ctx"}, "N",
        format("YaRN: original context size of model (default: %d = model training context size)", params.yarn_orig_ctx),
        [](gpt_params & params, int value) {
            params.yarn_orig_ctx = value;
        }
    ));
    add_opt(llama_arg(
        {"--yarn-ext-factor"}, "N",
        format("YaRN: extrapolation mix factor (default: %.1f, 0.0 = full interpolation)", (double)params.yarn_ext_factor),
        [](gpt_params & params, const std::string & value) {
            params.yarn_ext_factor = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--yarn-attn-factor"}, "N",
        format("YaRN: scale sqrt(t) or attention magnitude (default: %.1f)", (double)params.yarn_attn_factor),
        [](gpt_params & params, const std::string & value) {
            params.yarn_attn_factor = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--yarn-beta-slow"}, "N",
        format("YaRN: high correction dim or alpha (default: %.1f)", (double)params.yarn_beta_slow),
        [](gpt_params & params, const std::string & value) {
            params.yarn_beta_slow = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--yarn-beta-fast"}, "N",
        format("YaRN: low correction dim or beta (default: %.1f)", (double)params.yarn_beta_fast),
        [](gpt_params & params, const std::string & value) {
            params.yarn_beta_fast = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"-gan", "--grp-attn-n"}, "N",
        format("group-attention factor (default: %d)", params.grp_attn_n),
        [](gpt_params & params, int value) {
            params.grp_attn_n = value;
        }
    ));
    add_opt(llama_arg(
        {"-gaw", "--grp-attn-w"}, "N",
        format("group-attention width (default: %.1f)", (double)params.grp_attn_w),
        [](gpt_params & params, int value) {
            params.grp_attn_w = value;
        }
    ));
    add_opt(llama_arg(
        {"-dkvc", "--dump-kv-cache"},
        "verbose print of the KV cache",
        [](gpt_params & params) {
            params.dump_kv_cache = true;
        }
    ));
    add_opt(llama_arg(
        {"-nkvo", "--no-kv-offload"},
        "disable KV offload",
        [](gpt_params & params) {
            params.no_kv_offload = true;
        }
    ));
    add_opt(llama_arg(
        {"-ctk", "--cache-type-k"}, "TYPE",
        format("KV cache data type for K (default: %s)", params.cache_type_k.c_str()),
        [](gpt_params & params, const std::string & value) {
            // TODO: get the type right here
            params.cache_type_k = value;
        }
    ));
    add_opt(llama_arg(
        {"-ctv", "--cache-type-v"}, "TYPE",
        format("KV cache data type for V (default: %s)", params.cache_type_v.c_str()),
        [](gpt_params & params, const std::string & value) {
            // TODO: get the type right here
            params.cache_type_v = value;
        }
    ));
    add_opt(llama_arg(
        {"--all-logits"},
        format("return logits for all tokens in the batch (default: %s)", params.logits_all ? "true" : "false"),
        [](gpt_params & params) {
            params.logits_all = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--hellaswag"},
        "compute HellaSwag score over random tasks from datafile supplied with -f",
        [](gpt_params & params) {
            params.hellaswag = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--hellaswag-tasks"}, "N",
        format("number of tasks to use when computing the HellaSwag score (default: %zu)", params.hellaswag_tasks),
        [](gpt_params & params, int value) {
            params.hellaswag_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--winogrande"},
        "compute Winogrande score over random tasks from datafile supplied with -f",
        [](gpt_params & params) {
            params.winogrande = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--winogrande-tasks"}, "N",
        format("number of tasks to use when computing the Winogrande score (default: %zu)", params.winogrande_tasks),
        [](gpt_params & params, int value) {
            params.winogrande_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--multiple-choice"},
        "compute multiple choice score over random tasks from datafile supplied with -f",
        [](gpt_params & params) {
            params.multiple_choice = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--multiple-choice-tasks"}, "N",
        format("number of tasks to use when computing the multiple choice score (default: %zu)", params.multiple_choice_tasks),
        [](gpt_params & params, int value) {
            params.multiple_choice_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--kl-divergence"},
        "computes KL-divergence to logits provided via --kl-divergence-base",
        [](gpt_params & params) {
            params.kl_divergence = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--ppl-stride"}, "N",
        format("stride for perplexity calculation (default: %d)", params.ppl_stride),
        [](gpt_params & params, int value) {
            params.ppl_stride = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--ppl-output-type"}, "<0|1>",
        format("output type for perplexity calculation (default: %d)", params.ppl_output_type),
        [](gpt_params & params, int value) {
            params.ppl_output_type = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"-dt", "--defrag-thold"}, "N",
        format("KV cache defragmentation threshold (default: %.1f, < 0 - disabled)", (double)params.defrag_thold),
        [](gpt_params & params, const std::string & value) {
            params.defrag_thold = std::stof(value);
        }
    ).set_env("LLAMA_ARG_DEFRAG_THOLD"));
    add_opt(llama_arg(
        {"-np", "--parallel"}, "N",
        format("number of parallel sequences to decode (default: %d)", params.n_parallel),
        [](gpt_params & params, int value) {
            params.n_parallel = value;
        }
    ));
    add_opt(llama_arg(
        {"-ns", "--sequences"}, "N",
        format("number of sequences to decode (default: %d)", params.n_sequences),
        [](gpt_params & params, int value) {
            params.n_sequences = value;
        }
    ));
    add_opt(llama_arg(
        {"-cb", "--cont-batching"},
        format("enable continuous batching (a.k.a dynamic batching) (default: %s)", params.cont_batching ? "enabled" : "disabled"),
        [](gpt_params & params) {
            params.cont_batching = true;
        }
    ).set_env("LLAMA_ARG_CONT_BATCHING"));
    add_opt(llama_arg(
        {"-nocb", "--no-cont-batching"},
        "disable continuous batching",
        [](gpt_params & params) {
            params.cont_batching = false;
        }
    ).set_env("LLAMA_ARG_NO_CONT_BATCHING"));
    add_opt(llama_arg(
        {"--mmproj"}, "FILE",
        "path to a multimodal projector file for LLaVA. see examples/llava/README.md",
        [](gpt_params & params, const std::string & value) {
            params.mmproj = value;
        }
    ).set_examples({LLAMA_EXAMPLE_LLAVA}));
    add_opt(llama_arg(
        {"--image"}, "FILE",
        "path to an image file. use with multimodal models. Specify multiple times for batching",
        [](gpt_params & params, const std::string & value) {
            params.image.emplace_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_LLAVA}));
#ifdef GGML_USE_RPC
    add_opt(llama_arg(
        {"--rpc"}, "SERVERS",
        "comma separated list of RPC servers",
        [](gpt_params & params, const std::string & value) {
            params.rpc_servers = value;
        }
    ));
#endif
    add_opt(llama_arg(
        {"--mlock"},
        "force system to keep model in RAM rather than swapping or compressing",
        [](gpt_params & params) {
            params.use_mlock = true;
        }
    ));
    add_opt(llama_arg(
        {"--no-mmap"},
        "do not memory-map model (slower load but may reduce pageouts if not using mlock)",
        [](gpt_params & params) {
            params.use_mmap = false;
        }
    ));
    add_opt(llama_arg(
        {"--numa"}, "TYPE",
        "attempt optimizations that help on some NUMA systems\n"
        "- distribute: spread execution evenly over all nodes\n"
        "- isolate: only spawn threads on CPUs on the node that execution started on\n"
        "- numactl: use the CPU map provided by numactl\n"
        "if run without this previously, it is recommended to drop the system page cache before using this\n"
        "see https://github.com/ggerganov/llama.cpp/issues/1437",
        [](gpt_params & params, const std::string & value) {
            /**/ if (value == "distribute" || value == "") { params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE; }
            else if (value == "isolate") { params.numa = GGML_NUMA_STRATEGY_ISOLATE; }
            else if (value == "numactl") { params.numa = GGML_NUMA_STRATEGY_NUMACTL; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ));
    add_opt(llama_arg(
        {"-ngl", "--gpu-layers"}, "N",
        "number of layers to store in VRAM",
        [](gpt_params & params, int value) {
            params.n_gpu_layers = value;
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: not compiled with GPU offload support, --gpu-layers option will be ignored\n");
                fprintf(stderr, "warning: see main README.md for information on enabling GPU BLAS support\n");
            }
        }
    ).set_env("LLAMA_ARG_N_GPU_LAYERS"));
    add_opt(llama_arg(
        {"-ngld", "--gpu-layers-draft"}, "N",
        "number of layers to store in VRAM for the draft model",
        [](gpt_params & params, int value) {
            params.n_gpu_layers_draft = value;
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: not compiled with GPU offload support, --gpu-layers-draft option will be ignored\n");
                fprintf(stderr, "warning: see main README.md for information on enabling GPU BLAS support\n");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"-sm", "--split-mode"}, "{none,layer,row}",
        "how to split the model across multiple GPUs, one of:\n"
        "- none: use one GPU only\n"
        "- layer (default): split layers and KV across GPUs\n"
        "- row: split rows across GPUs",
        [](gpt_params & params, const std::string & value) {
            std::string arg_next = value;
            if (arg_next == "none") {
                params.split_mode = LLAMA_SPLIT_MODE_NONE;
            } else if (arg_next == "layer") {
                params.split_mode = LLAMA_SPLIT_MODE_LAYER;
            }
            else if (arg_next == "row") {
#ifdef GGML_USE_SYCL
                fprintf(stderr, "warning: The split mode value:[row] is not supported by llama.cpp with SYCL. It's developing.\nExit!\n");
                exit(1);
#endif // GGML_USE_SYCL
                params.split_mode = LLAMA_SPLIT_MODE_ROW;
            }
            else {
                throw std::invalid_argument("invalid value");
            }
#ifndef GGML_USE_CUDA_SYCL_VULKAN
            fprintf(stderr, "warning: llama.cpp was compiled without CUDA/SYCL/Vulkan. Setting the split mode has no effect.\n");
#endif // GGML_USE_CUDA_SYCL_VULKAN
        }
    ));
    add_opt(llama_arg(
        {"-ts", "--tensor-split"}, "N0,N1,N2,...",
        "fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1",
        [](gpt_params & params, const std::string & value) {
            std::string arg_next = value;

            // split string by , and /
            const std::regex regex{ R"([,/]+)" };
            std::sregex_token_iterator it{ arg_next.begin(), arg_next.end(), regex, -1 };
            std::vector<std::string> split_arg{ it, {} };
            if (split_arg.size() >= llama_max_devices()) {
                throw std::invalid_argument(
                    format("got %d input configs, but system only has %d devices", (int)split_arg.size(), (int)llama_max_devices())
                );
            }
            for (size_t i = 0; i < llama_max_devices(); ++i) {
                if (i < split_arg.size()) {
                        params.tensor_split[i] = std::stof(split_arg[i]);
                } else {
                        params.tensor_split[i] = 0.0f;
                }
            }
#ifndef GGML_USE_CUDA_SYCL_VULKAN
            fprintf(stderr, "warning: llama.cpp was compiled without CUDA/SYCL/Vulkan. Setting a tensor split has no effect.\n");
#endif // GGML_USE_CUDA_SYCL_VULKAN
        }
    ));
    add_opt(llama_arg(
        {"-mg", "--main-gpu"}, "INDEX",
        format("the GPU to use for the model (with split-mode = none), or for intermediate results and KV (with split-mode = row) (default: %d)", params.main_gpu),
        [](gpt_params & params, int value) {
            params.main_gpu = value;
#ifndef GGML_USE_CUDA_SYCL_VULKAN
            fprintf(stderr, "warning: llama.cpp was compiled without CUDA/SYCL/Vulkan. Setting the main GPU has no effect.\n");
#endif // GGML_USE_CUDA_SYCL_VULKAN
        }
    ));
    add_opt(llama_arg(
        {"--check-tensors"},
        format("check model tensor data for invalid values (default: %s)", params.check_tensors ? "true" : "false"),
        [](gpt_params & params) {
            params.check_tensors = true;
        }
    ));
    add_opt(llama_arg(
        {"--override-kv"}, "KEY=TYPE:VALUE",
        "advanced option to override model metadata by key. may be specified multiple times.\n"
        "types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false",
        [](gpt_params & params, const std::string & value) {
            if (!string_parse_kv_override(value.c_str(), params.kv_overrides)) {
                throw std::runtime_error(format("error: Invalid type for KV override: %s\n", value.c_str()));
            }
        }
    ));
    add_opt(llama_arg(
        {"--lora"}, "FNAME",
        "path to LoRA adapter (can be repeated to use multiple adapters)",
        [](gpt_params & params, const std::string & value) {
            params.lora_adapters.push_back({ std::string(value), 1.0 });
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(llama_arg(
        {"--lora-scaled"}, "FNAME", "SCALE",
        "path to LoRA adapter with user defined scaling (can be repeated to use multiple adapters)",
        [](gpt_params & params, const std::string & fname, const std::string & scale) {
            params.lora_adapters.push_back({ fname, std::stof(scale) });
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(llama_arg(
        {"--control-vector"}, "FNAME",
        "add a control vector\nnote: this argument can be repeated to add multiple control vectors",
        [](gpt_params & params, const std::string & value) {
            params.control_vectors.push_back({ 1.0f, value, });
        }
    ));
    add_opt(llama_arg(
        {"--control-vector-scaled"}, "FNAME", "SCALE",
        "add a control vector with user defined scaling SCALE\n"
        "note: this argument can be repeated to add multiple scaled control vectors",
        [](gpt_params & params, const std::string & fname, const std::string & scale) {
            params.control_vectors.push_back({ std::stof(scale), fname });
        }
    ));
    add_opt(llama_arg(
        {"--control-vector-layer-range"}, "START", "END",
        "layer range to apply the control vector(s) to, start and end inclusive",
        [](gpt_params & params, const std::string & start, const std::string & end) {
            params.control_vector_layer_start = std::stoi(start);
            params.control_vector_layer_end = std::stoi(end);
        }
    ));
    add_opt(llama_arg(
        {"-a", "--alias"}, "STRING",
        "set alias for model name (to be used by REST API)",
        [](gpt_params & params, const std::string & value) {
            params.model_alias = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODEL"));
    add_opt(llama_arg(
        {"-m", "--model"}, "FNAME",
        ex == LLAMA_EXAMPLE_EXPORT_LORA
            ? std::string("model path from which to load base model")
            : format(
                "model path (default: `models/$filename` with filename from `--hf-file` "
                "or `--model-url` if set, otherwise %s)", DEFAULT_MODEL_PATH
            ),
        [](gpt_params & params, const std::string & value) {
            params.model = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}).set_env("LLAMA_ARG_MODEL"));
    add_opt(llama_arg(
        {"-md", "--model-draft"}, "FNAME",
        "draft model for speculative decoding (default: unused)",
        [](gpt_params & params, const std::string & value) {
            params.model_draft = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"-mu", "--model-url"}, "MODEL_URL",
        "model download url (default: unused)",
        [](gpt_params & params, const std::string & value) {
            params.model_url = value;
        }
    ).set_env("LLAMA_ARG_MODEL_URL"));
    add_opt(llama_arg(
        {"-hfr", "--hf-repo"}, "REPO",
        "Hugging Face model repository (default: unused)",
        [](gpt_params & params, const std::string & value) {
            params.hf_repo = value;
        }
    ).set_env("LLAMA_ARG_HF_REPO"));
    add_opt(llama_arg(
        {"-hff", "--hf-file"}, "FILE",
        "Hugging Face model file (default: unused)",
        [](gpt_params & params, const std::string & value) {
            params.hf_file = value;
        }
    ).set_env("LLAMA_ARG_HF_FILE"));
    add_opt(llama_arg(
        {"-hft", "--hf-token"}, "TOKEN",
        "Hugging Face access token (default: value from HF_TOKEN environment variable)",
        [](gpt_params & params, const std::string & value) {
            params.hf_token = value;
        }
    ).set_env("HF_TOKEN"));
    add_opt(llama_arg(
        {"--context-file"}, "FNAME",
        "file to load context from (repeat to specify multiple files)",
        [](gpt_params & params, const std::string & value) {
            std::ifstream file(value, std::ios::binary);
            if (!file) {
                throw std::runtime_error(format("error: failed to open file '%s'\n", value.c_str()));
            }
            params.context_files.push_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(llama_arg(
        {"--chunk-size"}, "N",
        format("minimum length of embedded text chunks (default: %d)", params.chunk_size),
        [](gpt_params & params, int value) {
            params.chunk_size = value;
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(llama_arg(
        {"--chunk-separator"}, "STRING",
        format("separator between chunks (default: '%s')", params.chunk_separator.c_str()),
        [](gpt_params & params, const std::string & value) {
            params.chunk_separator = value;
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(llama_arg(
        {"--junk"}, "N",
        format("number of times to repeat the junk text (default: %d)", params.n_junk),
        [](gpt_params & params, int value) {
            params.n_junk = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PASSKEY}));
    add_opt(llama_arg(
        {"--pos"}, "N",
        format("position of the passkey in the junk text (default: %d)", params.i_pos),
        [](gpt_params & params, int value) {
            params.i_pos = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PASSKEY}));
    add_opt(llama_arg(
        {"-o", "--output"}, "FNAME",
        format("output file (default: '%s')",
            ex == LLAMA_EXAMPLE_EXPORT_LORA
                ? params.lora_outfile.c_str()
                : ex == LLAMA_EXAMPLE_CVECTOR_GENERATOR
                    ? params.cvector_outfile.c_str()
                    : params.out_file.c_str()),
        [](gpt_params & params, const std::string & value) {
            params.out_file = value;
            params.cvector_outfile = value;
            params.lora_outfile = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_CVECTOR_GENERATOR, LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(llama_arg(
        {"-ofreq", "--output-frequency"}, "N",
        format("output the imatrix every N iterations (default: %d)", params.n_out_freq),
        [](gpt_params & params, int value) {
            params.n_out_freq = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(llama_arg(
        {"--save-frequency"}, "N",
        format("save an imatrix copy every N iterations (default: %d)", params.n_save_freq),
        [](gpt_params & params, int value) {
            params.n_save_freq = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(llama_arg(
        {"--process-output"},
        format("collect data for the output tensor (default: %s)", params.process_output ? "true" : "false"),
        [](gpt_params & params) {
            params.process_output = true;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(llama_arg(
        {"--no-ppl"},
        format("do not compute perplexity (default: %s)", params.compute_ppl ? "true" : "false"),
        [](gpt_params & params) {
            params.compute_ppl = false;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(llama_arg(
        {"--chunk"}, "N",
        format("start processing the input from chunk N (default: %d)", params.i_chunk),
        [](gpt_params & params, int value) {
            params.i_chunk = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(llama_arg(
        {"-pps"},
        format("is the prompt shared across parallel sequences (default: %s)", params.is_pp_shared ? "true" : "false"),
        [](gpt_params & params) {
            params.is_pp_shared = true;
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(llama_arg(
        {"-npp"}, "n0,n1,...",
        "number of prompt tokens",
        [](gpt_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_pp.insert(params.n_pp.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(llama_arg(
        {"-ntg"}, "n0,n1,...",
        "number of text generation tokens",
        [](gpt_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_tg.insert(params.n_tg.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(llama_arg(
        {"-npl"}, "n0,n1,...",
        "number of parallel prompts",
        [](gpt_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_pl.insert(params.n_pl.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(llama_arg(
        {"--embd-normalize"}, "N",
        format("normalisation for embendings (default: %d) (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)", params.embd_normalize),
        [](gpt_params & params, int value) {
            params.embd_normalize = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(llama_arg(
        {"--embd-output-format"}, "FORMAT",
        "empty = default, \"array\" = [[],[]...], \"json\" = openai style, \"json+\" = same \"json\" + cosine similarity matrix",
        [](gpt_params & params, const std::string & value) {
            params.embd_out = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(llama_arg(
        {"--embd-separator"}, "STRING",
        "separator of embendings (default \\n) for example \"<#sep#>\"",
        [](gpt_params & params, const std::string & value) {
            params.embd_sep = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(llama_arg(
        {"--host"}, "HOST",
        format("ip address to listen (default: %s)", params.hostname.c_str()),
        [](gpt_params & params, const std::string & value) {
            params.hostname = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_HOST"));
    add_opt(llama_arg(
        {"--port"}, "PORT",
        format("port to listen (default: %d)", params.port),
        [](gpt_params & params, int value) {
            params.port = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_PORT"));
    add_opt(llama_arg(
        {"--path"}, "PATH",
        format("path to serve static files from (default: %s)", params.public_path.c_str()),
        [](gpt_params & params, const std::string & value) {
            params.public_path = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"--embedding", "--embeddings"},
        format("restrict to only support embedding use case; use only with dedicated embedding models (default: %s)", params.embedding ? "enabled" : "disabled"),
        [](gpt_params & params) {
            params.embedding = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_EMBEDDINGS"));
    add_opt(llama_arg(
        {"--api-key"}, "KEY",
        "API key to use for authentication (default: none)",
        [](gpt_params & params, const std::string & value) {
            params.api_keys.push_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_API_KEY"));
    add_opt(llama_arg(
        {"--api-key-file"}, "FNAME",
        "path to file containing API keys (default: none)",
        [](gpt_params & params, const std::string & value) {
            std::ifstream key_file(value);
            if (!key_file) {
                throw std::runtime_error(format("error: failed to open file '%s'\n", value.c_str()));
            }
            std::string key;
            while (std::getline(key_file, key)) {
                if (!key.empty()) {
                        params.api_keys.push_back(key);
                }
            }
            key_file.close();
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"--ssl-key-file"}, "FNAME",
        "path to file a PEM-encoded SSL private key",
        [](gpt_params & params, const std::string & value) {
            params.ssl_file_key = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"--ssl-cert-file"}, "FNAME",
        "path to file a PEM-encoded SSL certificate",
        [](gpt_params & params, const std::string & value) {
            params.ssl_file_cert = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"--timeout"}, "N",
        format("server read/write timeout in seconds (default: %d)", params.timeout_read),
        [](gpt_params & params, int value) {
            params.timeout_read  = value;
            params.timeout_write = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"--threads-http"}, "N",
        format("number of threads used to process HTTP requests (default: %d)", params.n_threads_http),
        [](gpt_params & params, int value) {
            params.n_threads_http = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_THREADS_HTTP"));
    add_opt(llama_arg(
        {"-spf", "--system-prompt-file"}, "FNAME",
        "set a file to load a system prompt (initial prompt of all slots), this is useful for chat applications",
        [](gpt_params & params, const std::string & value) {
            std::ifstream file(value);
            if (!file) {
                throw std::runtime_error(format("error: failed to open file '%s'\n", value.c_str()));
            }
            std::string system_prompt;
            std::copy(
                        std::istreambuf_iterator<char>(file),
                        std::istreambuf_iterator<char>(),
                        std::back_inserter(system_prompt)
                        );
            params.system_prompt = system_prompt;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"--log-format"}, "{text, json}",
        "log output format: json or text (default: json)",
        [](gpt_params & params, const std::string & value) {
            if (value == "json") {
                params.log_json = true;
            } else if (value == "text") {
                params.log_json = false;
            } else {
                throw std::invalid_argument("invalid value");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"--metrics"},
        format("enable prometheus compatible metrics endpoint (default: %s)", params.endpoint_metrics ? "enabled" : "disabled"),
        [](gpt_params & params) {
            params.endpoint_metrics = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ENDPOINT_METRICS"));
    add_opt(llama_arg(
        {"--no-slots"},
        format("disables slots monitoring endpoint (default: %s)", params.endpoint_slots ? "enabled" : "disabled"),
        [](gpt_params & params) {
            params.endpoint_slots = false;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_NO_ENDPOINT_SLOTS"));
    add_opt(llama_arg(
        {"--slot-save-path"}, "PATH",
        "path to save slot kv cache (default: disabled)",
        [](gpt_params & params, const std::string & value) {
            params.slot_save_path = value;
            // if doesn't end with DIRECTORY_SEPARATOR, add it
            if (!params.slot_save_path.empty() && params.slot_save_path[params.slot_save_path.size() - 1] != DIRECTORY_SEPARATOR) {
                params.slot_save_path += DIRECTORY_SEPARATOR;
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"--chat-template"}, "JINJA_TEMPLATE",
        "set custom jinja chat template (default: template taken from model's metadata)\n"
        "if suffix/prefix are specified, template will be disabled\n"
        "only commonly used templates are accepted:\nhttps://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template",
        [](gpt_params & params, const std::string & value) {
            if (!llama_chat_verify_template(value)) {
                throw std::runtime_error(format(
                    "error: the supplied chat template is not supported: %s\n"
                    "note: llama.cpp does not use jinja parser, we only support commonly used templates\n",
                    value.c_str()
                ));
            }
            params.chat_template = value;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CHAT_TEMPLATE"));
    add_opt(llama_arg(
        {"-sps", "--slot-prompt-similarity"}, "SIMILARITY",
        format("how much the prompt of a request must match the prompt of a slot in order to use that slot (default: %.2f, 0.0 = disabled)\n", params.slot_prompt_similarity),
        [](gpt_params & params, const std::string & value) {
            params.slot_prompt_similarity = std::stof(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"--lora-init-without-apply"},
        format("load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: %s)", params.lora_init_without_apply ? "enabled" : "disabled"),
        [](gpt_params & params) {
            params.lora_init_without_apply = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"--simple-io"},
        "use basic IO for better compatibility in subprocesses and limited consoles",
        [](gpt_params & params) {
            params.simple_io = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_INFILL}));
    add_opt(llama_arg(
        {"-ld", "--logdir"}, "LOGDIR",
        "path under which to save YAML logs (no logging if unset)",
        [](gpt_params & params, const std::string & value) {
            params.logdir = value;

            if (params.logdir.back() != DIRECTORY_SEPARATOR) {
                params.logdir += DIRECTORY_SEPARATOR;
            }
        }
    ));
    add_opt(llama_arg(
        {"--positive-file"}, "FNAME",
        format("positive prompts file, one prompt per line (default: '%s')", params.cvector_positive_file.c_str()),
        [](gpt_params & params, const std::string & value) {
            params.cvector_positive_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(llama_arg(
        {"--negative-file"}, "FNAME",
        format("negative prompts file, one prompt per line (default: '%s')", params.cvector_negative_file.c_str()),
        [](gpt_params & params, const std::string & value) {
            params.cvector_negative_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(llama_arg(
        {"--pca-batch"}, "N",
        format("batch size used for PCA. Larger batch runs faster, but uses more memory (default: %d)", params.n_pca_batch),
        [](gpt_params & params, int value) {
            params.n_pca_batch = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(llama_arg(
        {"--pca-iter"}, "N",
        format("number of iterations used for PCA (default: %d)", params.n_pca_iterations),
        [](gpt_params & params, int value) {
            params.n_pca_iterations = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(llama_arg(
        {"--method"}, "{pca, mean}",
        "dimensionality reduction method to be used (default: pca)",
        [](gpt_params & params, const std::string & value) {
            /**/ if (value == "pca") { params.cvector_dimre_method = DIMRE_METHOD_PCA; }
            else if (value == "mean") { params.cvector_dimre_method = DIMRE_METHOD_MEAN; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(llama_arg(
        {"--output-format"}, "{md,jsonl}",
        "output format for batched-bench results (default: md)",
        [](gpt_params & params, const std::string & value) {
            /**/ if (value == "jsonl") { params.batched_bench_output_jsonl = true; }
            else if (value == "md") { params.batched_bench_output_jsonl = false; }
            else { std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
#ifndef LOG_DISABLE_LOGS
    // TODO: make this looks less weird
    add_opt(llama_arg(
        {"--log-test"},
        "Log test",
        [](gpt_params &) { log_param_single_parse("--log-test"); }
    ));
    add_opt(llama_arg(
        {"--log-disable"},
        "Log disable",
        [](gpt_params &) { log_param_single_parse("--log-disable"); }
    ));
    add_opt(llama_arg(
        {"--log-enable"},
        "Log enable",
        [](gpt_params &) { log_param_single_parse("--log-enable"); }
    ));
    add_opt(llama_arg(
        {"--log-new"},
        "Log new",
        [](gpt_params &) { log_param_single_parse("--log-new"); }
    ));
    add_opt(llama_arg(
        {"--log-append"},
        "Log append",
        [](gpt_params &) { log_param_single_parse("--log-append"); }
    ));
    add_opt(llama_arg(
        {"--log-file"}, "FNAME",
        "Log file",
        [](gpt_params &, const std::string & value) { log_param_pair_parse(false, "--log-file", value); }
    ));
#endif // LOG_DISABLE_LOGS

    return options;
}

std::string gpt_params_get_system_info(const gpt_params & params) {
    std::ostringstream os;

    os << "system_info: n_threads = " << params.cpuparams.n_threads;
    if (params.cpuparams_batch.n_threads != -1) {
        os << " (n_threads_batch = " << params.cpuparams_batch.n_threads << ")";
    }
#if defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    DWORD logicalProcessorCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    os << " / " << logicalProcessorCount << " | " << llama_print_system_info();
#else
    os << " / " << std::thread::hardware_concurrency() << " | " << llama_print_system_info();
#endif

    return os.str();
}

//
// String utils
//

std::vector<std::string> string_split(std::string input, char separator) {
    std::vector<std::string> parts;
    size_t separator_pos = input.find(separator);
    while (separator_pos != std::string::npos) {
        std::string part = input.substr(0, separator_pos);
        parts.emplace_back(part);
        input = input.substr(separator_pos + 1);
        separator_pos = input.find(separator);
    }
    parts.emplace_back(input);
    return parts;
}

std::string string_strip(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();
    while (start < end && std::isspace(str[start])) {
        start++;
    }
    while (end > start && std::isspace(str[end - 1])) {
        end--;
    }
    return str.substr(start, end - start);
}

std::string string_get_sortable_timestamp() {
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

void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

void string_process_escapes(std::string & input) {
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
                case 'x':
                    // Handle \x12, etc
                    if (input_idx + 2 < input_len) {
                        const char x[3] = { input[input_idx + 1], input[input_idx + 2], 0 };
                        char *err_p = nullptr;
                        const long val = std::strtol(x, &err_p, 16);
                        if (err_p == x + 2) {
                            input_idx += 2;
                            input[output_idx++] = char(val);
                            break;
                        }
                    }
                    // fall through
                default:   input[output_idx++] = '\\';
                           input[output_idx++] = input[input_idx]; break;
            }
        } else {
            input[output_idx++] = input[input_idx];
        }
    }

    input.resize(output_idx);
}

bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides) {
    const char * sep = strchr(data, '=');
    if (sep == nullptr || sep - data >= 128) {
        fprintf(stderr, "%s: malformed KV override '%s'\n", __func__, data);
        return false;
    }
    llama_model_kv_override kvo;
    std::strncpy(kvo.key, data, sep - data);
    kvo.key[sep - data] = 0;
    sep++;
    if (strncmp(sep, "int:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        kvo.val_i64 = std::atol(sep);
    } else if (strncmp(sep, "float:", 6) == 0) {
        sep += 6;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
        kvo.val_f64 = std::atof(sep);
    } else if (strncmp(sep, "bool:", 5) == 0) {
        sep += 5;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
        if (std::strcmp(sep, "true") == 0) {
            kvo.val_bool = true;
        } else if (std::strcmp(sep, "false") == 0) {
            kvo.val_bool = false;
        } else {
            fprintf(stderr, "%s: invalid boolean value for KV override '%s'\n", __func__, data);
            return false;
        }
    } else if (strncmp(sep, "str:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        if (strlen(sep) > 127) {
            fprintf(stderr, "%s: malformed KV override '%s', value cannot exceed 127 chars\n", __func__, data);
            return false;
        }
        strncpy(kvo.val_str, sep, 127);
        kvo.val_str[127] = '\0';
    } else {
        fprintf(stderr, "%s: invalid type for KV override '%s'\n", __func__, data);
        return false;
    }
    overrides.emplace_back(std::move(kvo));
    return true;
}

//
// Filesystem utils
//

// Validate if a filename is safe to use
// To validate a full path, split the path by the OS-specific path separator, and validate each part with this function
bool fs_validate_filename(const std::string & filename) {
    if (!filename.length()) {
        // Empty filename invalid
        return false;
    }
    if (filename.length() > 255) {
        // Limit at common largest possible filename on Linux filesystems
        // to avoid unnecessary further validation
        // (On systems with smaller limits it will be caught by the OS)
        return false;
    }

    std::u32string filename_utf32;
    try {
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
        filename_utf32 = converter.from_bytes(filename);

        // If the reverse conversion mismatches, it means overlong UTF-8 sequences were used,
        // or invalid encodings were encountered. Reject such attempts
        std::string filename_reencoded = converter.to_bytes(filename_utf32);
        if (filename_reencoded != filename) {
            return false;
        }
    } catch (const std::exception &) {
        return false;
    }

    // Check for forbidden codepoints:
    // - Control characters
    // - Unicode equivalents of illegal characters
    // - UTF-16 surrogate pairs
    // - UTF-8 replacement character
    // - Byte order mark (BOM)
    // - Illegal characters: / \ : * ? " < > |
    for (char32_t c : filename_utf32) {
        if (c <= 0x1F // Control characters (C0)
            || c == 0x7F // Control characters (DEL)
            || (c >= 0x80 && c <= 0x9F) // Control characters (C1)
            || c == 0xFF0E // Fullwidth Full Stop (period equivalent)
            || c == 0x2215 // Division Slash (forward slash equivalent)
            || c == 0x2216 // Set Minus (backslash equivalent)
            || (c >= 0xD800 && c <= 0xDFFF) // UTF-16 surrogate pairs
            || c == 0xFFFD // Replacement Character (UTF-8)
            || c == 0xFEFF // Byte Order Mark (BOM)
            || c == '/' || c == '\\' || c == ':' || c == '*' // Illegal characters
            || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
            return false;
        }
    }

    // Reject any leading or trailing ' ', or any trailing '.', these are stripped on Windows and will cause a different filename
    // Unicode and other whitespace is not affected, only 0x20 space
    if (filename.front() == ' ' || filename.back() == ' ' || filename.back() == '.') {
        return false;
    }

    // Reject any ".." (currently stricter than necessary, it should be fine to just check for == ".." instead)
    if (filename.find("..") != std::string::npos) {
        return false;
    }

    // Reject "."
    if (filename == ".") {
        return false;
    }

    return true;
}

// returns true if successful, false otherwise
bool fs_create_directory_with_parents(const std::string & path) {
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

std::string fs_get_cache_directory() {
    std::string cache_directory = "";
    auto ensure_trailing_slash = [](std::string p) {
        // Make sure to add trailing slash
        if (p.back() != DIRECTORY_SEPARATOR) {
            p += DIRECTORY_SEPARATOR;
        }
        return p;
    };
    if (getenv("LLAMA_CACHE")) {
        cache_directory = std::getenv("LLAMA_CACHE");
    } else {
#ifdef __linux__
        if (std::getenv("XDG_CACHE_HOME")) {
            cache_directory = std::getenv("XDG_CACHE_HOME");
        } else {
            cache_directory = std::getenv("HOME") + std::string("/.cache/");
        }
#elif defined(__APPLE__)
        cache_directory = std::getenv("HOME") + std::string("/Library/Caches/");
#elif defined(_WIN32)
        cache_directory = std::getenv("LOCALAPPDATA");
#endif // __linux__
        cache_directory = ensure_trailing_slash(cache_directory);
        cache_directory += "llama.cpp";
    }
    return ensure_trailing_slash(cache_directory);
}

std::string fs_get_cache_file(const std::string & filename) {
    GGML_ASSERT(filename.find(DIRECTORY_SEPARATOR) == std::string::npos);
    std::string cache_directory = fs_get_cache_directory();
    const bool success = fs_create_directory_with_parents(cache_directory);
    if (!success) {
        throw std::runtime_error("failed to create cache directory: " + cache_directory);
    }
    return cache_directory + filename;
}


//
// Model utils
//
struct llama_init_result llama_init_from_gpt_params(gpt_params & params) {
    llama_init_result iparams;
    auto mparams = llama_model_params_from_gpt_params(params);

    llama_model * model = nullptr;

    if (!params.hf_repo.empty() && !params.hf_file.empty()) {
        model = llama_load_model_from_hf(params.hf_repo.c_str(), params.hf_file.c_str(), params.model.c_str(), params.hf_token.c_str(), mparams);
    } else if (!params.model_url.empty()) {
        model = llama_load_model_from_url(params.model_url.c_str(), params.model.c_str(), params.hf_token.c_str(), mparams);
    } else {
        model = llama_load_model_from_file(params.model.c_str(), mparams);
    }

    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
        return iparams;
    }

    auto cparams = llama_context_params_from_gpt_params(params);

    llama_context * lctx = llama_new_context_with_model(model, cparams);
    if (lctx == NULL) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params.model.c_str());
        llama_free_model(model);
        return iparams;
    }

    if (!params.control_vectors.empty()) {
        if (params.control_vector_layer_start <= 0) params.control_vector_layer_start = 1;
        if (params.control_vector_layer_end   <= 0) params.control_vector_layer_end   = llama_n_layer(model);

        const auto cvec = llama_control_vector_load(params.control_vectors);
        if (cvec.n_embd == -1) {
            llama_free(lctx);
            llama_free_model(model);
            return iparams;
        }

        int err = llama_control_vector_apply(lctx,
                                             cvec.data.data(),
                                             cvec.data.size(),
                                             cvec.n_embd,
                                             params.control_vector_layer_start,
                                             params.control_vector_layer_end);
        if (err) {
            llama_free(lctx);
            llama_free_model(model);
            return iparams;
        }
    }

    // load and optionally apply lora adapters
    for (auto & la : params.lora_adapters) {
        llama_lora_adapter_container loaded_la;
        loaded_la.path = la.path;
        loaded_la.scale = la.scale;
        loaded_la.adapter = llama_lora_adapter_init(model, la.path.c_str());
        if (loaded_la.adapter == nullptr) {
            fprintf(stderr, "%s: error: failed to apply lora adapter '%s'\n", __func__, la.path.c_str());
            llama_free(lctx);
            llama_free_model(model);
            return iparams;
        }
        iparams.lora_adapters.push_back(loaded_la); // copy to list of loaded adapters
    }
    if (!params.lora_init_without_apply) {
        llama_lora_adapters_apply(lctx, iparams.lora_adapters);
    }

    if (params.sparams.ignore_eos && llama_token_eos(model) == -1) {
        fprintf(stderr, "%s: warning: model does not have an EOS token, ignoring --ignore-eos\n", __func__);
        params.sparams.ignore_eos = false;
    }

    if (params.warmup) {
        LOG("warming up the model with an empty run\n");

        std::vector<llama_token> tmp;
        llama_token bos = llama_token_bos(model);
        llama_token eos = llama_token_eos(model);
        // some models (e.g. T5) don't have a BOS token
        if (bos != LLAMA_TOKEN_NULL) {
            tmp.push_back(bos);
        }
        if (eos != LLAMA_TOKEN_NULL) {
            tmp.push_back(eos);
        }
        if (tmp.empty()) {
            tmp.push_back(0);
        }

        if (llama_model_has_encoder(model)) {
            llama_encode(lctx, llama_batch_get_one(tmp.data(), tmp.size(), 0, 0));
            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == -1) {
                decoder_start_token_id = bos;
            }
            tmp.clear();
            tmp.push_back(decoder_start_token_id);
        }
        if (llama_model_has_decoder(model)) {
            llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) params.n_batch), 0, 0));
        }
        llama_kv_cache_clear(lctx);
        llama_synchronize(lctx);
        llama_perf_reset(lctx, LLAMA_PERF_TYPE_CONTEXT);
    }

    iparams.model   = model;
    iparams.context = lctx;
    return iparams;
}

void llama_lora_adapters_apply(struct llama_context * ctx, std::vector<llama_lora_adapter_container> & lora_adapters) {
    llama_lora_adapter_clear(ctx);
    for (auto & la : lora_adapters) {
        if (la.scale != 0.0f) {
            llama_lora_adapter_set(ctx, la.adapter, la.scale);
        }
    }
}

struct llama_model_params llama_model_params_from_gpt_params(const gpt_params & params) {
    auto mparams = llama_model_default_params();

    if (params.n_gpu_layers != -1) {
        mparams.n_gpu_layers = params.n_gpu_layers;
    }
    mparams.rpc_servers     = params.rpc_servers.c_str();
    mparams.main_gpu        = params.main_gpu;
    mparams.split_mode      = params.split_mode;
    mparams.tensor_split    = params.tensor_split;
    mparams.use_mmap        = params.use_mmap;
    mparams.use_mlock       = params.use_mlock;
    mparams.check_tensors   = params.check_tensors;
    if (params.kv_overrides.empty()) {
        mparams.kv_overrides = NULL;
    } else {
        GGML_ASSERT(params.kv_overrides.back().key[0] == 0 && "KV overrides not terminated with empty key");
        mparams.kv_overrides = params.kv_overrides.data();
    }

    return mparams;
}

static ggml_type kv_cache_type_from_str(const std::string & s) {
    if (s == "f32") {
        return GGML_TYPE_F32;
    }
    if (s == "f16") {
        return GGML_TYPE_F16;
    }
    if (s == "q8_0") {
        return GGML_TYPE_Q8_0;
    }
    if (s == "q4_0") {
        return GGML_TYPE_Q4_0;
    }
    if (s == "q4_1") {
        return GGML_TYPE_Q4_1;
    }
    if (s == "iq4_nl") {
        return GGML_TYPE_IQ4_NL;
    }
    if (s == "q5_0") {
        return GGML_TYPE_Q5_0;
    }
    if (s == "q5_1") {
        return GGML_TYPE_Q5_1;
    }

    throw std::runtime_error("Invalid cache type: " + s);
}

struct llama_context_params llama_context_params_from_gpt_params(const gpt_params & params) {
    auto cparams = llama_context_default_params();

    cparams.n_ctx             = params.n_ctx;
    cparams.n_seq_max         = params.n_parallel;
    cparams.n_batch           = params.n_batch;
    cparams.n_ubatch          = params.n_ubatch;
    cparams.n_threads         = params.cpuparams.n_threads;
    cparams.n_threads_batch   = params.cpuparams_batch.n_threads == -1 ?
                                    params.cpuparams.n_threads : params.cpuparams_batch.n_threads;
    cparams.logits_all        = params.logits_all;
    cparams.embeddings        = params.embedding;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.rope_freq_base    = params.rope_freq_base;
    cparams.rope_freq_scale   = params.rope_freq_scale;
    cparams.yarn_ext_factor   = params.yarn_ext_factor;
    cparams.yarn_attn_factor  = params.yarn_attn_factor;
    cparams.yarn_beta_fast    = params.yarn_beta_fast;
    cparams.yarn_beta_slow    = params.yarn_beta_slow;
    cparams.yarn_orig_ctx     = params.yarn_orig_ctx;
    cparams.pooling_type      = params.pooling_type;
    cparams.attention_type    = params.attention_type;
    cparams.defrag_thold      = params.defrag_thold;
    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;
    cparams.offload_kqv       = !params.no_kv_offload;
    cparams.flash_attn        = params.flash_attn;

    cparams.type_k = kv_cache_type_from_str(params.cache_type_k);
    cparams.type_v = kv_cache_type_from_str(params.cache_type_v);

    return cparams;
}

struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const cpu_params & params) {
    struct ggml_threadpool_params tpp;

    ggml_threadpool_params_init(&tpp, params.n_threads); // setup the defaults

    if (params.mask_valid) {
        std::memcpy(&tpp.cpumask, &params.cpumask, GGML_MAX_N_THREADS);
    }

    tpp.prio       = params.priority;
    tpp.poll       = params.poll;
    tpp.strict_cpu = params.strict_cpu;

    return tpp;
}

#ifdef LLAMA_USE_CURL

static bool starts_with(const std::string & str, const std::string & prefix) {
    // While we wait for C++20's std::string::starts_with...
    return str.rfind(prefix, 0) == 0;
}

static bool llama_download_file(const std::string & url, const std::string & path, const std::string & hf_token) {

    // Initialize libcurl
    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), &curl_easy_cleanup);
    if (!curl) {
        fprintf(stderr, "%s: error initializing libcurl\n", __func__);
        return false;
    }

    bool force_download = false;

    // Set the URL, allow to follow http redirection
    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);

    // Check if hf-token or bearer-token was specified
    if (!hf_token.empty()) {
      std::string auth_header = "Authorization: Bearer ";
      auth_header += hf_token.c_str();
      struct curl_slist *http_headers = NULL;
      http_headers = curl_slist_append(http_headers, auth_header.c_str());
      curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, http_headers);
    }

#if defined(_WIN32)
    // CURLSSLOPT_NATIVE_CA tells libcurl to use standard certificate store of
    //   operating system. Currently implemented under MS-Windows.
    curl_easy_setopt(curl.get(), CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA);
#endif

    // Check if the file already exists locally
    struct stat model_file_info;
    auto file_exists = (stat(path.c_str(), &model_file_info) == 0);

    // If the file exists, check its JSON metadata companion file.
    std::string metadata_path = path + ".json";
    nlohmann::json metadata;
    std::string etag;
    std::string last_modified;

    if (file_exists) {
        // Try and read the JSON metadata file (note: stream autoclosed upon exiting this block).
        std::ifstream metadata_in(metadata_path);
        if (metadata_in.good()) {
            try {
                metadata_in >> metadata;
                fprintf(stderr, "%s: previous metadata file found %s: %s\n", __func__, metadata_path.c_str(), metadata.dump().c_str());
                if (metadata.contains("url") && metadata.at("url").is_string()) {
                    auto previous_url = metadata.at("url").get<std::string>();
                    if (previous_url != url) {
                        fprintf(stderr, "%s: Model URL mismatch: %s != %s\n", __func__, url.c_str(), previous_url.c_str());
                        return false;
                    }
                }
                if (metadata.contains("etag") && metadata.at("etag").is_string()) {
                    etag = metadata.at("etag");
                }
                if (metadata.contains("lastModified") && metadata.at("lastModified").is_string()) {
                    last_modified = metadata.at("lastModified");
                }
            } catch (const nlohmann::json::exception & e) {
                fprintf(stderr, "%s: error reading metadata file %s: %s\n", __func__, metadata_path.c_str(), e.what());
                return false;
            }
        }
    } else {
        fprintf(stderr, "%s: no previous model file found %s\n", __func__, path.c_str());
    }

    // Send a HEAD request to retrieve the etag and last-modified headers
    struct llama_load_model_from_url_headers {
        std::string etag;
        std::string last_modified;
    };
    llama_load_model_from_url_headers headers;
    {
        typedef size_t(*CURLOPT_HEADERFUNCTION_PTR)(char *, size_t, size_t, void *);
        auto header_callback = [](char * buffer, size_t /*size*/, size_t n_items, void * userdata) -> size_t {
            llama_load_model_from_url_headers *headers = (llama_load_model_from_url_headers *) userdata;

            static std::regex header_regex("([^:]+): (.*)\r\n");
            static std::regex etag_regex("ETag", std::regex_constants::icase);
            static std::regex last_modified_regex("Last-Modified", std::regex_constants::icase);

            std::string header(buffer, n_items);
            std::smatch match;
            if (std::regex_match(header, match, header_regex)) {
                const std::string & key = match[1];
                const std::string & value = match[2];
                if (std::regex_match(key, match, etag_regex)) {
                    headers->etag = value;
                } else if (std::regex_match(key, match, last_modified_regex)) {
                    headers->last_modified = value;
                }
            }
            return n_items;
        };

        curl_easy_setopt(curl.get(), CURLOPT_NOBODY, 1L); // will trigger the HEAD verb
        curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 1L); // hide head request progress
        curl_easy_setopt(curl.get(), CURLOPT_HEADERFUNCTION, static_cast<CURLOPT_HEADERFUNCTION_PTR>(header_callback));
        curl_easy_setopt(curl.get(), CURLOPT_HEADERDATA, &headers);

        CURLcode res = curl_easy_perform(curl.get());
        if (res != CURLE_OK) {
            fprintf(stderr, "%s: curl_easy_perform() failed: %s\n", __func__, curl_easy_strerror(res));
            return false;
        }

        long http_code = 0;
        curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code != 200) {
            // HEAD not supported, we don't know if the file has changed
            // force trigger downloading
            force_download = true;
            fprintf(stderr, "%s: HEAD invalid http status code received: %ld\n", __func__, http_code);
        }
    }

    bool should_download = !file_exists || force_download;
    if (!should_download) {
        if (!etag.empty() && etag != headers.etag) {
            fprintf(stderr, "%s: ETag header is different (%s != %s): triggering a new download\n", __func__, etag.c_str(), headers.etag.c_str());
            should_download = true;
        } else if (!last_modified.empty() && last_modified != headers.last_modified) {
            fprintf(stderr, "%s: Last-Modified header is different (%s != %s): triggering a new download\n", __func__, last_modified.c_str(), headers.last_modified.c_str());
            should_download = true;
        }
    }
    if (should_download) {
        std::string path_temporary = path + ".downloadInProgress";
        if (file_exists) {
            fprintf(stderr, "%s: deleting previous downloaded file: %s\n", __func__, path.c_str());
            if (remove(path.c_str()) != 0) {
                fprintf(stderr, "%s: unable to delete file: %s\n", __func__, path.c_str());
                return false;
            }
        }

        // Set the output file

        struct FILE_deleter {
            void operator()(FILE * f) const {
                fclose(f);
            }
        };

        std::unique_ptr<FILE, FILE_deleter> outfile(fopen(path_temporary.c_str(), "wb"));
        if (!outfile) {
            fprintf(stderr, "%s: error opening local file for writing: %s\n", __func__, path.c_str());
            return false;
        }

        typedef size_t(*CURLOPT_WRITEFUNCTION_PTR)(void * data, size_t size, size_t nmemb, void * fd);
        auto write_callback = [](void * data, size_t size, size_t nmemb, void * fd) -> size_t {
            return fwrite(data, size, nmemb, (FILE *)fd);
        };
        curl_easy_setopt(curl.get(), CURLOPT_NOBODY, 0L);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, static_cast<CURLOPT_WRITEFUNCTION_PTR>(write_callback));
        curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, outfile.get());

        //  display download progress
        curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 0L);

        // helper function to hide password in URL
        auto llama_download_hide_password_in_url = [](const std::string & url) -> std::string {
            std::size_t protocol_pos = url.find("://");
            if (protocol_pos == std::string::npos) {
                return url;  // Malformed URL
            }

            std::size_t at_pos = url.find('@', protocol_pos + 3);
            if (at_pos == std::string::npos) {
                return url;  // No password in URL
            }

            return url.substr(0, protocol_pos + 3) + "********" + url.substr(at_pos);
        };

        // start the download
        fprintf(stderr, "%s: downloading from %s to %s (server_etag:%s, server_last_modified:%s)...\n", __func__,
                llama_download_hide_password_in_url(url).c_str(), path.c_str(), headers.etag.c_str(), headers.last_modified.c_str());
        auto res = curl_easy_perform(curl.get());
        if (res != CURLE_OK) {
            fprintf(stderr, "%s: curl_easy_perform() failed: %s\n", __func__, curl_easy_strerror(res));
            return false;
        }

        long http_code = 0;
        curl_easy_getinfo (curl.get(), CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code < 200 || http_code >= 400) {
            fprintf(stderr, "%s: invalid http status code received: %ld\n", __func__, http_code);
            return false;
        }

        // Causes file to be closed explicitly here before we rename it.
        outfile.reset();

        // Write the updated JSON metadata file.
        metadata.update({
            {"url", url},
            {"etag", headers.etag},
            {"lastModified", headers.last_modified}
        });
        std::ofstream(metadata_path) << metadata.dump(4);
        fprintf(stderr, "%s: file metadata saved: %s\n", __func__, metadata_path.c_str());

        if (rename(path_temporary.c_str(), path.c_str()) != 0) {
            fprintf(stderr, "%s: unable to rename file: %s to %s\n", __func__, path_temporary.c_str(), path.c_str());
            return false;
        }
    }

    return true;
}

struct llama_model * llama_load_model_from_url(
        const char * model_url,
        const char * path_model,
        const char * hf_token,
        const struct llama_model_params & params) {
    // Basic validation of the model_url
    if (!model_url || strlen(model_url) == 0) {
        fprintf(stderr, "%s: invalid model_url\n", __func__);
        return NULL;
    }

    if (!llama_download_file(model_url, path_model, hf_token)) {
        return NULL;
    }

    // check for additional GGUFs split to download
    int n_split = 0;
    {
        struct gguf_init_params gguf_params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ NULL,
        };
        auto * ctx_gguf = gguf_init_from_file(path_model, gguf_params);
        if (!ctx_gguf) {
            fprintf(stderr, "\n%s:  failed to load input GGUF from %s\n", __func__, path_model);
            return NULL;
        }

        auto key_n_split = gguf_find_key(ctx_gguf, LLM_KV_SPLIT_COUNT);
        if (key_n_split >= 0) {
            n_split = gguf_get_val_u16(ctx_gguf, key_n_split);
        }

        gguf_free(ctx_gguf);
    }

    if (n_split > 1) {
        char split_prefix[PATH_MAX] = {0};
        char split_url_prefix[LLAMA_CURL_MAX_URL_LENGTH] = {0};

        // Verify the first split file format
        // and extract split URL and PATH prefixes
        {
            if (!llama_split_prefix(split_prefix, sizeof(split_prefix), path_model, 0, n_split)) {
                fprintf(stderr, "\n%s: unexpected model file name: %s"
                                " n_split=%d\n", __func__, path_model, n_split);
                return NULL;
            }

            if (!llama_split_prefix(split_url_prefix, sizeof(split_url_prefix), model_url, 0, n_split)) {
                fprintf(stderr, "\n%s: unexpected model url: %s"
                                " n_split=%d\n", __func__, model_url, n_split);
                return NULL;
            }
        }

        // Prepare download in parallel
        std::vector<std::future<bool>> futures_download;
        for (int idx = 1; idx < n_split; idx++) {
            futures_download.push_back(std::async(std::launch::async, [&split_prefix, &split_url_prefix, &n_split, hf_token](int download_idx) -> bool {
                char split_path[PATH_MAX] = {0};
                llama_split_path(split_path, sizeof(split_path), split_prefix, download_idx, n_split);

                char split_url[LLAMA_CURL_MAX_URL_LENGTH] = {0};
                llama_split_path(split_url, sizeof(split_url), split_url_prefix, download_idx, n_split);

                return llama_download_file(split_url, split_path, hf_token);
            }, idx));
        }

        // Wait for all downloads to complete
        for (auto & f : futures_download) {
            if (!f.get()) {
                return NULL;
            }
        }
    }

    return llama_load_model_from_file(path_model, params);
}

struct llama_model * llama_load_model_from_hf(
        const char * repo,
        const char * model,
        const char * path_model,
        const char * hf_token,
        const struct llama_model_params & params) {
    // construct hugging face model url:
    //
    //  --repo ggml-org/models --file tinyllama-1.1b/ggml-model-f16.gguf
    //    https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b/ggml-model-f16.gguf
    //
    //  --repo TheBloke/Mixtral-8x7B-v0.1-GGUF --file mixtral-8x7b-v0.1.Q4_K_M.gguf
    //    https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF/resolve/main/mixtral-8x7b-v0.1.Q4_K_M.gguf
    //

    std::string model_url = "https://huggingface.co/";
    model_url += repo;
    model_url += "/resolve/main/";
    model_url += model;

    return llama_load_model_from_url(model_url.c_str(), path_model, hf_token, params);
}

#else

struct llama_model * llama_load_model_from_url(
        const char * /*model_url*/,
        const char * /*path_model*/,
        const char * /*hf_token*/,
        const struct llama_model_params & /*params*/) {
    fprintf(stderr, "%s: llama.cpp built without libcurl, downloading from an url not supported.\n", __func__);
    return nullptr;
}

struct llama_model * llama_load_model_from_hf(
        const char * /*repo*/,
        const char * /*model*/,
        const char * /*path_model*/,
        const char * /*hf_token*/,
        const struct llama_model_params & /*params*/) {
    fprintf(stderr, "%s: llama.cpp built without libcurl, downloading from Hugging Face not supported.\n", __func__);
    return nullptr;
}

#endif // LLAMA_USE_CURL

//
// Batch utils
//

void llama_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

void llama_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}

//
// Vocab utils
//

std::vector<llama_token> llama_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    return llama_tokenize(llama_get_model(ctx), text, add_special, parse_special);
}

std::vector<llama_token> llama_tokenize(
    const struct llama_model * model,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(llama_get_model(ctx), token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(llama_get_model(ctx), token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

std::string llama_detokenize(llama_context * ctx, const std::vector<llama_token> & tokens, bool special) {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = llama_detokenize(llama_get_model(ctx), tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = llama_detokenize(llama_get_model(ctx), tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
        GGML_ASSERT(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

//
// Chat template utils
//

bool llama_chat_verify_template(const std::string & tmpl) {
    llama_chat_message chat[] = {{"user", "test"}};
    int res = llama_chat_apply_template(nullptr, tmpl.c_str(), chat, 1, true, nullptr, 0);
    return res >= 0;
}

std::string llama_chat_apply_template(const struct llama_model * model,
        const std::string & tmpl,
        const std::vector<llama_chat_msg> & msgs,
        bool add_ass) {
    int alloc_size = 0;
    bool fallback = false; // indicate if we must fallback to default chatml
    std::vector<llama_chat_message> chat;
    for (auto & msg : msgs) {
        chat.push_back({msg.role.c_str(), msg.content.c_str()});
        alloc_size += (msg.role.size() + msg.content.size()) * 1.25;
    }

    const char * ptr_tmpl = tmpl.empty() ? nullptr : tmpl.c_str();
    std::vector<char> buf(alloc_size);

    // run the first time to get the total output length
    int32_t res = llama_chat_apply_template(model, ptr_tmpl, chat.data(), chat.size(), add_ass, buf.data(), buf.size());

    // error: chat template is not supported
    if (res < 0) {
        if (ptr_tmpl != nullptr) {
            // if the custom "tmpl" is not supported, we throw an error
            // this is a bit redundant (for good), since we're not sure if user validated the custom template with llama_chat_verify_template()
            throw std::runtime_error("this custom template is not supported");
        } else {
            // If the built-in template is not supported, we default to chatml
            res = llama_chat_apply_template(nullptr, "chatml", chat.data(), chat.size(), add_ass, buf.data(), buf.size());
            fallback = true;
        }
    }

    // if it turns out that our buffer is too small, we resize it
    if ((size_t) res > buf.size()) {
        buf.resize(res);
        res = llama_chat_apply_template(
            fallback ? nullptr : model,
            fallback ? "chatml" : ptr_tmpl,
            chat.data(), chat.size(), add_ass, buf.data(), buf.size());
    }

    std::string formatted_chat(buf.data(), res);
    return formatted_chat;
}

std::string llama_chat_format_single(const struct llama_model * model,
        const std::string & tmpl,
        const std::vector<llama_chat_msg> & past_msg,
        const llama_chat_msg & new_msg,
        bool add_ass) {
    std::ostringstream ss;
    auto fmt_past_msg = past_msg.empty() ? "" : llama_chat_apply_template(model, tmpl, past_msg, false);
    std::vector<llama_chat_msg> chat_new(past_msg);
    // if the past_msg ends with a newline, we must preserve it in the formatted version
    if (add_ass && !fmt_past_msg.empty() && fmt_past_msg.back() == '\n') {
        ss << "\n";
    };
    // format chat with new_msg
    chat_new.push_back(new_msg);
    auto fmt_new_msg = llama_chat_apply_template(model, tmpl, chat_new, add_ass);
    // get the diff part
    ss << fmt_new_msg.substr(fmt_past_msg.size(), fmt_new_msg.size() - fmt_past_msg.size());
    return ss.str();
}

std::string llama_chat_format_example(const struct llama_model * model,
        const std::string & tmpl) {
    std::vector<llama_chat_msg> msgs = {
        {"system",    "You are a helpful assistant"},
        {"user",      "Hello"},
        {"assistant", "Hi there"},
        {"user",      "How are you?"},
    };
    return llama_chat_apply_template(model, tmpl, msgs, true);
}

//
// KV cache utils
//

void llama_kv_cache_dump_view(const llama_kv_cache_view & view, int row_size) {
    static const char slot_chars[] = ".123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+";

    printf("=== Dumping KV cache. total cells %d, max sequences per cell %d, populated cells %d, total tokens in cache %d, largest empty slot=%d @ %d",
        view.n_cells, view.n_seq_max, view.used_cells, view.token_count, view.max_contiguous, view.max_contiguous_idx);

    llama_kv_cache_view_cell * c_curr = view.cells;
    llama_seq_id * cs_curr = view.cells_sequences;

    for (int i = 0; i < view.n_cells; i++, c_curr++, cs_curr += view.n_seq_max) {
        if (i % row_size == 0) {
            printf("\n%5d: ", i);
        }
        int seq_count = 0;
        for (int j = 0; j < view.n_seq_max; j++) {
            if (cs_curr[j] >= 0) { seq_count++; }
        }
        putchar(slot_chars[std::min(sizeof(slot_chars) - 2, size_t(seq_count))]);
    }

    printf("\n=== Done dumping\n");
}

void llama_kv_cache_dump_view_seqs(const llama_kv_cache_view & view, int row_size) {
    static const char slot_chars[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    printf("=== Dumping KV cache. total cells %d, max sequences per cell %d, populated cells %d, total tokens in cache %d, largest empty slot=%d @ %d\n",
        view.n_cells, view.n_seq_max, view.used_cells, view.token_count, view.max_contiguous, view.max_contiguous_idx);

    std::unordered_map<llama_seq_id, size_t> seqs;
    llama_kv_cache_view_cell * c_curr = view.cells;
    llama_seq_id * cs_curr = view.cells_sequences;

    for (int i = 0; i < view.n_cells; i++, c_curr++, cs_curr += view.n_seq_max) {
        for (int j = 0; j < view.n_seq_max; j++) {
            if (cs_curr[j] < 0) { continue; }
            if (seqs.find(cs_curr[j]) == seqs.end()) {
                if (seqs.size() + 1 >= sizeof(slot_chars)) { break; }
                const size_t sz = seqs.size();
                seqs[cs_curr[j]] = sz;
            }
        }
        if (seqs.size() + 1 >= sizeof(slot_chars)) { break; }
    }

    printf("=== Sequence legend: ");
    for (const auto & it : seqs) {
        printf("%zu=%d, ", it.second, it.first);
    }
    printf("'+'=other sequence ids");

    c_curr = view.cells;
    cs_curr = view.cells_sequences;
    for (int i = 0; i < view.n_cells; i++, c_curr++, cs_curr += view.n_seq_max) {
        if (i % row_size == 0) {
            printf("\n%5d: ", i);
        }
        for (int j = 0; j < view.n_seq_max; j++) {
            if (cs_curr[j] >= 0) {
                const auto & it = seqs.find(cs_curr[j]);
                putchar(it != seqs.end() ? int(slot_chars[it->second]) : '+');
            } else {
                putchar('.');
            }
        }
        putchar(' ');
    }

    printf("\n=== Done dumping\n");
}

//
// Embedding utils
//

void llama_embd_normalize(const float * inp, float * out, int n, int embd_norm) {
    double sum = 0.0;

    switch (embd_norm) {
        case -1: // no normalisation
            sum = 1.0;
            break;
        case 0: // max absolute
            for (int i = 0; i < n; i++) {
                if (sum < std::abs(inp[i])) sum = std::abs(inp[i]);
            }
            sum /= 32760.0; // make an int16 range
            break;
        case 2: // euclidean
            for (int i = 0; i < n; i++) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default: // p-norm (euclidean is p-norm p=2)
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(inp[i]), embd_norm);
            }
            sum = std::pow(sum, 1.0 / embd_norm);
            break;
    }

    const float norm = sum > 0.0 ? 1.0 / sum : 0.0f;

    for (int i = 0; i < n; i++) {
        out[i] = inp[i] * norm;
    }
}

float llama_embd_similarity_cos(const float * embd1, const float * embd2, int n){
    double sum  = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 0; i < n; i++) {
        sum  += embd1[i] * embd2[i];
        sum1 += embd1[i] * embd1[i];
        sum2 += embd2[i] * embd2[i];
    }

    // Handle the case where one or both vectors are zero vectors
    if (sum1 == 0.0 || sum2 == 0.0) {
        if (sum1 == 0.0 && sum2 == 0.0) {
            return 1.0f; // two zero vectors are similar
        }
        return 0.0f;
    }

    return sum / (sqrt(sum1) * sqrt(sum2));
}

//
// Control vector utils
//

static llama_control_vector_data llama_control_vector_load_one(const llama_control_vector_load_info & load_info) {
    llama_control_vector_data result = { -1, {} };

    ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(load_info.fname.c_str(), meta_gguf_params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s: failed to load control vector file from %s\n", __func__, load_info.fname.c_str());
        return result;
    }

    int32_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    if (n_tensors == 0) {
        fprintf(stderr, "%s: no direction tensors found in %s\n", __func__, load_info.fname.c_str());
    }

    for (int i = 0; i < n_tensors; i++) {
        std::string name = gguf_get_tensor_name(ctx_gguf, i);

        int layer_idx = -1;

        // split on '.'
        size_t dotpos = name.find('.');
        if (dotpos != std::string::npos && name.substr(0, dotpos) == "direction") {
            try {
                layer_idx = std::stoi(name.substr(dotpos + 1));
            } catch (...) {
                layer_idx = -1;
            }
        }
        if (layer_idx < 0) {
            fprintf(stderr, "%s: invalid/unparsable direction tensor layer index in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        } else if (layer_idx == 0) {
            fprintf(stderr, "%s: invalid (zero) direction tensor layer index in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
        if (tensor->type != GGML_TYPE_F32) {
            fprintf(stderr, "%s: invalid (non-F32) direction tensor type in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }
        if (ggml_n_dims(tensor) != 1) {
            fprintf(stderr, "%s: invalid (non-1D) direction tensor shape in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result.n_embd = ggml_nelements(tensor);
        } else if (ggml_nelements(tensor) != result.n_embd) {
            fprintf(stderr, "%s: direction tensor in %s does not match previous dimensions\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        // extend if necessary - do not store data for layer 0 (it's not used)
        result.data.resize(std::max(result.data.size(), static_cast<size_t>(result.n_embd * layer_idx)), 0.0f);

        const float * src = (const float *) tensor->data;
        float * dst = result.data.data() + result.n_embd * (layer_idx - 1);  // layer 1 at [0]
        for (int j = 0; j < result.n_embd; j++) {
            dst[j] += src[j] * load_info.strength;  // allows multiple directions for same layer in same file
        }

    }

    if (result.n_embd == -1) {
        fprintf(stderr, "%s: skipping %s due to invalid direction tensors\n", __func__, load_info.fname.c_str());
        result.data.clear();
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    return result;
}

llama_control_vector_data llama_control_vector_load(const std::vector<llama_control_vector_load_info> & load_infos) {
    llama_control_vector_data result = { -1, {} };

    for (const auto & info : load_infos) {
        auto cur = llama_control_vector_load_one(info);

        if (cur.n_embd == -1) {
            result.n_embd = -1;
            break;
        }
        if (result.n_embd != -1 && result.n_embd != cur.n_embd) {
            fprintf(stderr, "%s: control vectors in %s does not match previous dimensions\n", __func__, info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result = std::move(cur);
        } else {
            result.data.resize(std::max(result.data.size(), cur.data.size()), 0.0f);  // extend if necessary
            for (size_t i = 0; i < cur.data.size(); i++) {
                result.data[i] += cur.data[i];
            }
        }
    }

    if (result.n_embd == -1) {
        fprintf(stderr, "%s: no valid control vector files passed\n", __func__);
        result.data.clear();
    }

    return result;
}

//
// YAML utils
//

void yaml_dump_vector_float(FILE * stream, const char * prop_name, const std::vector<float> & data) {
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

void yaml_dump_vector_int(FILE * stream, const char * prop_name, const std::vector<int> & data) {
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

void yaml_dump_string_multiline(FILE * stream, const char * prop_name, const char * data) {
    std::string data_str(data == NULL ? "" : data);

    if (data_str.empty()) {
        fprintf(stream, "%s:\n", prop_name);
        return;
    }

    size_t pos_start = 0;
    size_t pos_found = 0;

    if (std::isspace(data_str[0]) || std::isspace(data_str.back())) {
        data_str = std::regex_replace(data_str, std::regex("\n"), "\\n");
        data_str = std::regex_replace(data_str, std::regex("\""), "\\\"");
        data_str = std::regex_replace(data_str, std::regex(R"(\\[^n"])"), R"(\$&)");
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

void yaml_dump_non_result_info(FILE * stream, const gpt_params & params, const llama_context * lctx,
                               const std::string & timestamp, const std::vector<int> & prompt_tokens, const char * model_desc) {
    const auto & sparams = params.sparams;

    fprintf(stream, "build_commit: %s\n",        LLAMA_COMMIT);
    fprintf(stream, "build_number: %d\n",        LLAMA_BUILD_NUMBER);
    fprintf(stream, "cpu_has_arm_fma: %s\n",     ggml_cpu_has_arm_fma()     ? "true" : "false");
    fprintf(stream, "cpu_has_avx: %s\n",         ggml_cpu_has_avx()         ? "true" : "false");
    fprintf(stream, "cpu_has_avx_vnni: %s\n",    ggml_cpu_has_avx_vnni()    ? "true" : "false");
    fprintf(stream, "cpu_has_avx2: %s\n",        ggml_cpu_has_avx2()        ? "true" : "false");
    fprintf(stream, "cpu_has_avx512: %s\n",      ggml_cpu_has_avx512()      ? "true" : "false");
    fprintf(stream, "cpu_has_avx512_vbmi: %s\n", ggml_cpu_has_avx512_vbmi() ? "true" : "false");
    fprintf(stream, "cpu_has_avx512_vnni: %s\n", ggml_cpu_has_avx512_vnni() ? "true" : "false");
    fprintf(stream, "cpu_has_cuda: %s\n",        ggml_cpu_has_cuda()        ? "true" : "false");
    fprintf(stream, "cpu_has_vulkan: %s\n",      ggml_cpu_has_vulkan()      ? "true" : "false");
    fprintf(stream, "cpu_has_kompute: %s\n",     ggml_cpu_has_kompute()     ? "true" : "false");
    fprintf(stream, "cpu_has_fma: %s\n",         ggml_cpu_has_fma()         ? "true" : "false");
    fprintf(stream, "cpu_has_gpublas: %s\n",     ggml_cpu_has_gpublas()     ? "true" : "false");
    fprintf(stream, "cpu_has_neon: %s\n",        ggml_cpu_has_neon()        ? "true" : "false");
    fprintf(stream, "cpu_has_sve: %s\n",         ggml_cpu_has_sve()         ? "true" : "false");
    fprintf(stream, "cpu_has_f16c: %s\n",        ggml_cpu_has_f16c()        ? "true" : "false");
    fprintf(stream, "cpu_has_fp16_va: %s\n",     ggml_cpu_has_fp16_va()     ? "true" : "false");
    fprintf(stream, "cpu_has_wasm_simd: %s\n",   ggml_cpu_has_wasm_simd()   ? "true" : "false");
    fprintf(stream, "cpu_has_blas: %s\n",        ggml_cpu_has_blas()        ? "true" : "false");
    fprintf(stream, "cpu_has_sse3: %s\n",        ggml_cpu_has_sse3()        ? "true" : "false");
    fprintf(stream, "cpu_has_vsx: %s\n",         ggml_cpu_has_vsx()         ? "true" : "false");
    fprintf(stream, "cpu_has_matmul_int8: %s\n", ggml_cpu_has_matmul_int8() ? "true" : "false");

#ifdef NDEBUG
    fprintf(stream, "debug: false\n");
#else
    fprintf(stream, "debug: true\n");
#endif // NDEBUG

    fprintf(stream, "model_desc: %s\n", model_desc);
    fprintf(stream, "n_vocab: %d  # output size of the final layer, 32001 for some models\n", llama_n_vocab(llama_get_model(lctx)));

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
    fprintf(stream, "chunks: %d # default: -1 (unlimited)\n", params.n_chunks);
    fprintf(stream, "color: %s # default: false\n", params.use_color ? "true" : "false");
    fprintf(stream, "ctx_size: %d # default: 512\n", params.n_ctx);
    fprintf(stream, "escape: %s # default: false\n", params.escape ? "true" : "false");
    fprintf(stream, "file: # never logged, see prompt instead. Can still be specified for input.\n");
    fprintf(stream, "frequency_penalty: %f # default: 0.0 \n", sparams.penalty_freq);
    yaml_dump_string_multiline(stream, "grammar", sparams.grammar.c_str());
    fprintf(stream, "grammar-file: # never logged, see grammar instead. Can still be specified for input.\n");
    fprintf(stream, "hellaswag: %s # default: false\n", params.hellaswag ? "true" : "false");
    fprintf(stream, "hellaswag_tasks: %zu # default: 400\n", params.hellaswag_tasks);
    fprintf(stream, "ignore_eos: %s # default: false\n", sparams.ignore_eos ? "true" : "false");

    yaml_dump_string_multiline(stream, "in_prefix", params.input_prefix.c_str());
    fprintf(stream, "in_prefix_bos: %s # default: false\n", params.input_prefix_bos ? "true" : "false");
    yaml_dump_string_multiline(stream, "in_suffix", params.input_prefix.c_str());
    fprintf(stream, "interactive: %s # default: false\n", params.interactive ? "true" : "false");
    fprintf(stream, "interactive_first: %s # default: false\n", params.interactive_first ? "true" : "false");
    fprintf(stream, "keep: %d # default: 0\n", params.n_keep);
    fprintf(stream, "logdir: %s # default: unset (no logging)\n", params.logdir.c_str());

    fprintf(stream, "logit_bias:\n");
    for (const auto & logit_bias : sparams.logit_bias) {
        fprintf(stream, "  %d: %f", logit_bias.token, logit_bias.bias);
    }

    fprintf(stream, "lora:\n");
    for (auto & la : params.lora_adapters) {
        if (la.scale == 1.0f) {
            fprintf(stream, "  - %s\n", la.path.c_str());
        }
    }
    fprintf(stream, "lora_scaled:\n");
    for (auto & la : params.lora_adapters) {
        if (la.scale != 1.0f) {
            fprintf(stream, "  - %s: %f\n", la.path.c_str(), la.scale);
        }
    }
    fprintf(stream, "lora_init_without_apply: %s # default: false\n", params.lora_init_without_apply ? "true" : "false");
    fprintf(stream, "main_gpu: %d # default: 0\n", params.main_gpu);
    fprintf(stream, "min_keep: %d # default: 0 (disabled)\n", sparams.min_keep);
    fprintf(stream, "mirostat: %d # default: 0 (disabled)\n", sparams.mirostat);
    fprintf(stream, "mirostat_ent: %f # default: 5.0\n", sparams.mirostat_tau);
    fprintf(stream, "mirostat_lr: %f # default: 0.1\n", sparams.mirostat_eta);
    fprintf(stream, "mlock: %s # default: false\n", params.use_mlock ? "true" : "false");
    fprintf(stream, "model: %s # default: %s\n", params.model.c_str(), DEFAULT_MODEL_PATH);
    fprintf(stream, "model_draft: %s # default:\n", params.model_draft.c_str());
    fprintf(stream, "multiline_input: %s # default: false\n", params.multiline_input ? "true" : "false");
    fprintf(stream, "n_gpu_layers: %d # default: -1\n", params.n_gpu_layers);
    fprintf(stream, "n_predict: %d # default: -1 (unlimited)\n", params.n_predict);
    fprintf(stream, "n_probs: %d # only used by server binary, default: 0\n", sparams.n_probs);
    fprintf(stream, "no_mmap: %s # default: false\n", !params.use_mmap ? "true" : "false");
    fprintf(stream, "penalize_nl: %s # default: false\n", sparams.penalize_nl ? "true" : "false");
    fprintf(stream, "ppl_output_type: %d # default: 0\n", params.ppl_output_type);
    fprintf(stream, "ppl_stride: %d # default: 0\n", params.ppl_stride);
    fprintf(stream, "presence_penalty: %f # default: 0.0\n", sparams.penalty_present);
    yaml_dump_string_multiline(stream, "prompt", params.prompt.c_str());
    fprintf(stream, "prompt_cache: %s\n", params.path_prompt_cache.c_str());
    fprintf(stream, "prompt_cache_all: %s # default: false\n", params.prompt_cache_all ? "true" : "false");
    fprintf(stream, "prompt_cache_ro: %s # default: false\n", params.prompt_cache_ro ? "true" : "false");
    yaml_dump_vector_int(stream, "prompt_tokens", prompt_tokens);
    fprintf(stream, "repeat_penalty: %f # default: 1.1\n", sparams.penalty_repeat);

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
    fprintf(stream, "simple_io: %s # default: false\n", params.simple_io ? "true" : "false");
    fprintf(stream, "cont_batching: %s # default: false\n", params.cont_batching ? "true" : "false");
    fprintf(stream, "flash_attn: %s # default: false\n", params.flash_attn ? "true" : "false");
    fprintf(stream, "temp: %f # default: 0.8\n", sparams.temp);

    const std::vector<float> tensor_split_vector(params.tensor_split, params.tensor_split + llama_max_devices());
    yaml_dump_vector_float(stream, "tensor_split", tensor_split_vector);

    fprintf(stream, "tfs: %f # default: 1.0\n", sparams.tfs_z);
    fprintf(stream, "threads: %d # default: %u\n", params.cpuparams.n_threads, std::thread::hardware_concurrency());
    fprintf(stream, "top_k: %d # default: 40\n", sparams.top_k);
    fprintf(stream, "top_p: %f # default: 0.95\n", sparams.top_p);
    fprintf(stream, "min_p: %f # default: 0.0\n", sparams.min_p);
    fprintf(stream, "typ_p: %f # default: 1.0\n", sparams.typ_p);
    fprintf(stream, "verbose_prompt: %s # default: false\n", params.verbose_prompt ? "true" : "false");
    fprintf(stream, "display_prompt: %s # default: true\n", params.display_prompt ? "true" : "false");
}
