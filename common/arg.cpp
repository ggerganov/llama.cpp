#include "arg.h"

#include "log.h"
#include "sampling.h"

#include <algorithm>
#include <climits>
#include <cstdarg>
#include <fstream>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "json-schema-to-grammar.h"

using json = nlohmann::ordered_json;

llama_arg & llama_arg::set_examples(std::initializer_list<enum llama_example> examples) {
    this->examples = std::move(examples);
    return *this;
}

llama_arg & llama_arg::set_env(const char * env) {
    help = help + "\n(env: " + env + ")";
    this->env = env;
    return *this;
}

llama_arg & llama_arg::set_sparam() {
    is_sparam = true;
    return *this;
}

bool llama_arg::in_example(enum llama_example ex) {
    return examples.find(ex) != examples.end();
}

bool llama_arg::get_value_from_env(std::string & output) {
    if (env == nullptr) return false;
    char * value = std::getenv(env);
    if (value) {
        output = value;
        return true;
    }
    return false;
}

bool llama_arg::has_value_from_env() {
    return env != nullptr && std::getenv(env);
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
                // first arg is usually abbreviation, we need padding to make it more beautiful
                auto tmp = std::string(arg) + ", ";
                auto spaces = std::string(std::max(0, 7 - (int)tmp.size()), ' ');
                ss << tmp << spaces;
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

//
// utils
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

//
// CLI argument parsing functions
//

static bool gpt_params_parse_ex(int argc, char ** argv, gpt_params_context & ctx_arg) {
    std::string arg;
    const std::string arg_prefix = "--";
    gpt_params & params = ctx_arg.params;

    std::unordered_map<std::string, llama_arg *> arg_to_options;
    for (auto & opt : ctx_arg.options) {
        for (const auto & arg : opt.args) {
            arg_to_options[arg] = &opt;
        }
    }

    // handle environment variables
    for (auto & opt : ctx_arg.options) {
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

    if (params.reranking && params.embedding) {
        throw std::invalid_argument("error: either --embedding or --reranking can be specified, but not both");
    }

    return true;
}

static void gpt_params_print_usage(gpt_params_context & ctx_arg) {
    auto print_options = [](std::vector<llama_arg *> & options) {
        for (llama_arg * opt : options) {
            printf("%s", opt->to_string().c_str());
        }
    };

    std::vector<llama_arg *> common_options;
    std::vector<llama_arg *> sparam_options;
    std::vector<llama_arg *> specific_options;
    for (auto & opt : ctx_arg.options) {
        // in case multiple LLAMA_EXAMPLE_* are set, we prioritize the LLAMA_EXAMPLE_* matching current example
        if (opt.is_sparam) {
            sparam_options.push_back(&opt);
        } else if (opt.in_example(ctx_arg.ex)) {
            specific_options.push_back(&opt);
        } else {
            common_options.push_back(&opt);
        }
    }
    printf("----- common params -----\n\n");
    print_options(common_options);
    printf("\n\n----- sampling params -----\n\n");
    print_options(sparam_options);
    // TODO: maybe convert enum llama_example to string
    printf("\n\n----- example-specific params -----\n\n");
    print_options(specific_options);
}

bool gpt_params_parse(int argc, char ** argv, gpt_params & params, llama_example ex, void(*print_usage)(int, char **)) {
    auto ctx_arg = gpt_params_parser_init(params, ex, print_usage);
    const gpt_params params_org = ctx_arg.params; // the example can modify the default params

    try {
        if (!gpt_params_parse_ex(argc, argv, ctx_arg)) {
            ctx_arg.params = params_org;
            return false;
        }
        if (ctx_arg.params.usage) {
            gpt_params_print_usage(ctx_arg);
            if (ctx_arg.print_usage) {
                ctx_arg.print_usage(argc, argv);
            }
            exit(0);
        }
    } catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        ctx_arg.params = params_org;
        return false;
    }

    return true;
}

gpt_params_context gpt_params_parser_init(gpt_params & params, llama_example ex, void(*print_usage)(int, char **)) {
    gpt_params_context ctx_arg(params);
    ctx_arg.print_usage = print_usage;
    ctx_arg.ex          = ex;

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
    auto add_opt = [&](llama_arg arg) {
        if (arg.in_example(ex) || arg.in_example(LLAMA_EXAMPLE_COMMON)) {
            ctx_arg.options.push_back(std::move(arg));
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
        {"--verbose-prompt"},
        format("print a verbose prompt before generation (default: %s)", params.verbose_prompt ? "true" : "false"),
        [](gpt_params & params) {
            params.verbose_prompt = true;
        }
    ));
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
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_INFILL, LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP}));
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
        [](gpt_params & params, const std::string & mask) {
            params.cpuparams.mask_valid = true;
            if (!parse_cpu_mask(mask, params.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ));
    add_opt(llama_arg(
        {"-Cr", "--cpu-range"}, "lo-hi",
        "range of CPUs for affinity. Complements --cpu-mask",
        [](gpt_params & params, const std::string & range) {
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
        {"--prio"}, "N",
        format("set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.cpuparams.priority),
        [](gpt_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.cpuparams.priority = (enum ggml_sched_priority) prio;
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
        [](gpt_params & params, const std::string & mask) {
            params.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_mask(mask, params.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ));
    add_opt(llama_arg(
        {"-Crb", "--cpu-range-batch"}, "lo-hi",
        "ranges of CPUs for affinity. Complements --cpu-mask-batch",
        [](gpt_params & params, const std::string & range) {
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
        {"--prio-batch"}, "N",
        format("set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.cpuparams_batch.priority),
        [](gpt_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.cpuparams_batch.priority = (enum ggml_sched_priority) prio;
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
        [](gpt_params & params, const std::string & mask) {
            params.draft_cpuparams.mask_valid = true;
            if (!parse_cpu_mask(mask, params.draft_cpuparams.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"-Crd", "--cpu-range-draft"}, "lo-hi",
        "Ranges of CPUs for affinity. Complements --cpu-mask-draft",
        [](gpt_params & params, const std::string & range) {
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
        {"--prio-draft"}, "N",
        format("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.draft_cpuparams.priority),
        [](gpt_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.draft_cpuparams.priority = (enum ggml_sched_priority) prio;
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
        {"-Cbd", "--cpu-mask-batch-draft"}, "M",
        "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)",
        [](gpt_params & params, const std::string & mask) {
            params.draft_cpuparams_batch.mask_valid = true;
            if (!parse_cpu_mask(mask, params.draft_cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"-Crbd", "--cpu-range-batch-draft"}, "lo-hi",
        "Ranges of CPUs for affinity. Complements --cpu-mask-draft-batch)",
        [](gpt_params & params, const std::string & range) {
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
        {"--prio-batch-draft"}, "N",
        format("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.draft_cpuparams_batch.priority),
        [](gpt_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.draft_cpuparams_batch.priority = (enum ggml_sched_priority) prio;
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
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP}));
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
    ).set_examples({LLAMA_EXAMPLE_LOOKUP}));
    add_opt(llama_arg(
        {"-lcd", "--lookup-cache-dynamic"}, "FNAME",
        "path to dynamic lookup cache to use for lookup decoding (updated by generation)",
        [](gpt_params & params, const std::string & value) {
            params.lookup_cache_dynamic = value;
        }
    ).set_examples({LLAMA_EXAMPLE_LOOKUP}));
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
        {"--no-context-shift"},
        format("disables context shift on inifinite text generation (default: %s)", params.ctx_shift ? "disabled" : "enabled"),
        [](gpt_params & params) {
            params.ctx_shift = false;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_NO_CONTEXT_SHIFT"));
    add_opt(llama_arg(
        {"--chunks"}, "N",
        format("max number of chunks to process (default: %d, -1 = all)", params.n_chunks),
        [](gpt_params & params, int value) {
            params.n_chunks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_RETRIEVAL}));
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
        {"--no-perf"},
        format("disable internal libllama performance timings (default: %s)", params.no_perf ? "true" : "false"),
        [](gpt_params & params) {
            params.no_perf = true;
            params.sparams.no_perf = true;
        }
    ).set_env("LLAMA_ARG_NO_PERF"));
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
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
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
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_SERVER}));
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
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"-if", "--interactive-first"},
        format("run in interactive mode and wait for input right away (default: %s)", params.interactive_first ? "true" : "false"),
        [](gpt_params & params) {
            params.interactive_first = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"-mli", "--multiline-input"},
        "allows you to write or paste multiple lines without ending each in '\\'",
        [](gpt_params & params) {
            params.multiline_input = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"--in-prefix-bos"},
        "prefix BOS to user inputs, preceding the `--in-prefix` string",
        [](gpt_params & params) {
            params.input_prefix_bos = true;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"--in-prefix"}, "STRING",
        "string to prefix user inputs with (default: empty)",
        [](gpt_params & params, const std::string & value) {
            params.input_prefix = value;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_INFILL}));
    add_opt(llama_arg(
        {"--in-suffix"}, "STRING",
        "string to suffix after user inputs with (default: empty)",
        [](gpt_params & params, const std::string & value) {
            params.input_suffix = value;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_INFILL}));
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
    ).set_sparam());
    add_opt(llama_arg(
        {"-s", "--seed"}, "SEED",
        format("RNG seed (default: %u, use random seed for %u)", params.sparams.seed, LLAMA_DEFAULT_SEED),
        [](gpt_params & params, const std::string & value) {
            params.sparams.seed = std::stoul(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--sampling-seq"}, "SEQUENCE",
        format("simplified sequence for samplers that will be used (default: %s)", sampler_type_chars.c_str()),
        [](gpt_params & params, const std::string & value) {
            params.sparams.samplers = gpt_sampler_types_from_chars(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--ignore-eos"},
        "ignore end of stream token and continue generating (implies --logit-bias EOS-inf)",
        [](gpt_params & params) {
            params.sparams.ignore_eos = true;
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--penalize-nl"},
        format("penalize newline tokens (default: %s)", params.sparams.penalize_nl ? "true" : "false"),
        [](gpt_params & params) {
            params.sparams.penalize_nl = true;
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--temp"}, "N",
        format("temperature (default: %.1f)", (double)params.sparams.temp),
        [](gpt_params & params, const std::string & value) {
            params.sparams.temp = std::stof(value);
            params.sparams.temp = std::max(params.sparams.temp, 0.0f);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--top-k"}, "N",
        format("top-k sampling (default: %d, 0 = disabled)", params.sparams.top_k),
        [](gpt_params & params, int value) {
            params.sparams.top_k = value;
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--top-p"}, "N",
        format("top-p sampling (default: %.1f, 1.0 = disabled)", (double)params.sparams.top_p),
        [](gpt_params & params, const std::string & value) {
            params.sparams.top_p = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--min-p"}, "N",
        format("min-p sampling (default: %.1f, 0.0 = disabled)", (double)params.sparams.min_p),
        [](gpt_params & params, const std::string & value) {
            params.sparams.min_p = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--tfs"}, "N",
        format("tail free sampling, parameter z (default: %.1f, 1.0 = disabled)", (double)params.sparams.tfs_z),
        [](gpt_params & params, const std::string & value) {
            params.sparams.tfs_z = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--typical"}, "N",
        format("locally typical sampling, parameter p (default: %.1f, 1.0 = disabled)", (double)params.sparams.typ_p),
        [](gpt_params & params, const std::string & value) {
            params.sparams.typ_p = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--repeat-last-n"}, "N",
        format("last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)", params.sparams.penalty_last_n),
        [](gpt_params & params, int value) {
            params.sparams.penalty_last_n = value;
            params.sparams.n_prev = std::max(params.sparams.n_prev, params.sparams.penalty_last_n);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--repeat-penalty"}, "N",
        format("penalize repeat sequence of tokens (default: %.1f, 1.0 = disabled)", (double)params.sparams.penalty_repeat),
        [](gpt_params & params, const std::string & value) {
            params.sparams.penalty_repeat = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--presence-penalty"}, "N",
        format("repeat alpha presence penalty (default: %.1f, 0.0 = disabled)", (double)params.sparams.penalty_present),
        [](gpt_params & params, const std::string & value) {
            params.sparams.penalty_present = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--frequency-penalty"}, "N",
        format("repeat alpha frequency penalty (default: %.1f, 0.0 = disabled)", (double)params.sparams.penalty_freq),
        [](gpt_params & params, const std::string & value) {
            params.sparams.penalty_freq = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--dynatemp-range"}, "N",
        format("dynamic temperature range (default: %.1f, 0.0 = disabled)", (double)params.sparams.dynatemp_range),
        [](gpt_params & params, const std::string & value) {
            params.sparams.dynatemp_range = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--dynatemp-exp"}, "N",
        format("dynamic temperature exponent (default: %.1f)", (double)params.sparams.dynatemp_exponent),
        [](gpt_params & params, const std::string & value) {
            params.sparams.dynatemp_exponent = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--mirostat"}, "N",
        format("use Mirostat sampling.\nTop K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.\n"
        "(default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)", params.sparams.mirostat),
        [](gpt_params & params, int value) {
            params.sparams.mirostat = value;
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--mirostat-lr"}, "N",
        format("Mirostat learning rate, parameter eta (default: %.1f)", (double)params.sparams.mirostat_eta),
        [](gpt_params & params, const std::string & value) {
            params.sparams.mirostat_eta = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--mirostat-ent"}, "N",
        format("Mirostat target entropy, parameter tau (default: %.1f)", (double)params.sparams.mirostat_tau),
        [](gpt_params & params, const std::string & value) {
            params.sparams.mirostat_tau = std::stof(value);
        }
    ).set_sparam());
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
    ).set_sparam());
    add_opt(llama_arg(
        {"--grammar"}, "GRAMMAR",
        format("BNF-like grammar to constrain generations (see samples in grammars/ dir) (default: '%s')", params.sparams.grammar.c_str()),
        [](gpt_params & params, const std::string & value) {
            params.sparams.grammar = value;
        }
    ).set_sparam());
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
    ).set_sparam());
    add_opt(llama_arg(
        {"-j", "--json-schema"}, "SCHEMA",
        "JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object\nFor schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead",
        [](gpt_params & params, const std::string & value) {
            params.sparams.grammar = json_schema_to_grammar(json::parse(value));
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--pooling"}, "{none,mean,cls,last,rank}",
        "pooling type for embeddings, use model default if unspecified",
        [](gpt_params & params, const std::string & value) {
            /**/ if (value == "none") { params.pooling_type = LLAMA_POOLING_TYPE_NONE; }
            else if (value == "mean") { params.pooling_type = LLAMA_POOLING_TYPE_MEAN; }
            else if (value == "cls")  { params.pooling_type = LLAMA_POOLING_TYPE_CLS;  }
            else if (value == "last") { params.pooling_type = LLAMA_POOLING_TYPE_LAST; }
            else if (value == "rank") { params.pooling_type = LLAMA_POOLING_TYPE_RANK; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_RETRIEVAL, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_POOLING"));
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
    ).set_env("LLAMA_ARG_ROPE_SCALING_TYPE"));
    add_opt(llama_arg(
        {"--rope-scale"}, "N",
        "RoPE context scaling factor, expands context by a factor of N",
        [](gpt_params & params, const std::string & value) {
            params.rope_freq_scale = 1.0f / std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_SCALE"));
    add_opt(llama_arg(
        {"--rope-freq-base"}, "N",
        "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)",
        [](gpt_params & params, const std::string & value) {
            params.rope_freq_base = std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_FREQ_BASE"));
    add_opt(llama_arg(
        {"--rope-freq-scale"}, "N",
        "RoPE frequency scaling factor, expands context by a factor of 1/N",
        [](gpt_params & params, const std::string & value) {
            params.rope_freq_scale = std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_FREQ_SCALE"));
    add_opt(llama_arg(
        {"--yarn-orig-ctx"}, "N",
        format("YaRN: original context size of model (default: %d = model training context size)", params.yarn_orig_ctx),
        [](gpt_params & params, int value) {
            params.yarn_orig_ctx = value;
        }
    ).set_env("LLAMA_ARG_YARN_ORIG_CTX"));
    add_opt(llama_arg(
        {"--yarn-ext-factor"}, "N",
        format("YaRN: extrapolation mix factor (default: %.1f, 0.0 = full interpolation)", (double)params.yarn_ext_factor),
        [](gpt_params & params, const std::string & value) {
            params.yarn_ext_factor = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_EXT_FACTOR"));
    add_opt(llama_arg(
        {"--yarn-attn-factor"}, "N",
        format("YaRN: scale sqrt(t) or attention magnitude (default: %.1f)", (double)params.yarn_attn_factor),
        [](gpt_params & params, const std::string & value) {
            params.yarn_attn_factor = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_ATTN_FACTOR"));
    add_opt(llama_arg(
        {"--yarn-beta-slow"}, "N",
        format("YaRN: high correction dim or alpha (default: %.1f)", (double)params.yarn_beta_slow),
        [](gpt_params & params, const std::string & value) {
            params.yarn_beta_slow = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_BETA_SLOW"));
    add_opt(llama_arg(
        {"--yarn-beta-fast"}, "N",
        format("YaRN: low correction dim or beta (default: %.1f)", (double)params.yarn_beta_fast),
        [](gpt_params & params, const std::string & value) {
            params.yarn_beta_fast = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_BETA_FAST"));
    add_opt(llama_arg(
        {"-gan", "--grp-attn-n"}, "N",
        format("group-attention factor (default: %d)", params.grp_attn_n),
        [](gpt_params & params, int value) {
            params.grp_attn_n = value;
        }
    ).set_env("LLAMA_ARG_GRP_ATTN_N"));
    add_opt(llama_arg(
        {"-gaw", "--grp-attn-w"}, "N",
        format("group-attention width (default: %.1f)", (double)params.grp_attn_w),
        [](gpt_params & params, int value) {
            params.grp_attn_w = value;
        }
    ).set_env("LLAMA_ARG_GRP_ATTN_W"));
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
    ).set_env("LLAMA_ARG_NO_KV_OFFLOAD"));
    add_opt(llama_arg(
        {"-ctk", "--cache-type-k"}, "TYPE",
        format("KV cache data type for K (default: %s)", params.cache_type_k.c_str()),
        [](gpt_params & params, const std::string & value) {
            // TODO: get the type right here
            params.cache_type_k = value;
        }
    ).set_env("LLAMA_ARG_CACHE_TYPE_K"));
    add_opt(llama_arg(
        {"-ctv", "--cache-type-v"}, "TYPE",
        format("KV cache data type for V (default: %s)", params.cache_type_v.c_str()),
        [](gpt_params & params, const std::string & value) {
            // TODO: get the type right here
            params.cache_type_v = value;
        }
    ).set_env("LLAMA_ARG_CACHE_TYPE_V"));
    add_opt(llama_arg(
        {"--perplexity", "--all-logits"},
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
        {"--save-all-logits", "--kl-divergence-base"}, "FNAME",
        "set logits file",
        [](gpt_params & params, const std::string & value) {
            params.logits_file = value;
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
    ).set_env("LLAMA_ARG_N_PARALLEL"));
    add_opt(llama_arg(
        {"-ns", "--sequences"}, "N",
        format("number of sequences to decode (default: %d)", params.n_sequences),
        [](gpt_params & params, int value) {
            params.n_sequences = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PARALLEL}));
    add_opt(llama_arg(
        {"-cb", "--cont-batching"},
        format("enable continuous batching (a.k.a dynamic batching) (default: %s)", params.cont_batching ? "enabled" : "disabled"),
        [](gpt_params & params) {
            params.cont_batching = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CONT_BATCHING"));
    add_opt(llama_arg(
        {"-nocb", "--no-cont-batching"},
        "disable continuous batching",
        [](gpt_params & params) {
            params.cont_batching = false;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_NO_CONT_BATCHING"));
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
    ).set_env("LLAMA_ARG_RPC"));
#endif
    add_opt(llama_arg(
        {"--mlock"},
        "force system to keep model in RAM rather than swapping or compressing",
        [](gpt_params & params) {
            params.use_mlock = true;
        }
    ).set_env("LLAMA_ARG_MLOCK"));
    add_opt(llama_arg(
        {"--no-mmap"},
        "do not memory-map model (slower load but may reduce pageouts if not using mlock)",
        [](gpt_params & params) {
            params.use_mmap = false;
        }
    ).set_env("LLAMA_ARG_NO_MMAP"));
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
    ).set_env("LLAMA_ARG_NUMA"));
    add_opt(llama_arg(
        {"-ngl", "--gpu-layers", "--n-gpu-layers"}, "N",
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
        {"-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"}, "N",
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
            } else if (arg_next == "row") {
#ifdef GGML_USE_SYCL
                fprintf(stderr, "warning: The split mode value:[row] is not supported by llama.cpp with SYCL. It's developing.\nExit!\n");
                exit(1);
#endif // GGML_USE_SYCL
                params.split_mode = LLAMA_SPLIT_MODE_ROW;
            } else {
                throw std::invalid_argument("invalid value");
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting the split mode has no effect.\n");
            }
        }
    ).set_env("LLAMA_ARG_SPLIT_MODE"));
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
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting a tensor split has no effect.\n");
            }
        }
    ).set_env("LLAMA_ARG_TENSOR_SPLIT"));
    add_opt(llama_arg(
        {"-mg", "--main-gpu"}, "INDEX",
        format("the GPU to use for the model (with split-mode = none), or for intermediate results and KV (with split-mode = row) (default: %d)", params.main_gpu),
        [](gpt_params & params, int value) {
            params.main_gpu = value;
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting the main GPU has no effect.\n");
            }
        }
    ).set_env("LLAMA_ARG_MAIN_GPU"));
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
        // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora, it will be categorized as "example-specific" arg
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(llama_arg(
        {"--lora-scaled"}, "FNAME", "SCALE",
        "path to LoRA adapter with user defined scaling (can be repeated to use multiple adapters)",
        [](gpt_params & params, const std::string & fname, const std::string & scale) {
            params.lora_adapters.push_back({ fname, std::stof(scale) });
        }
        // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora, it will be categorized as "example-specific" arg
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
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ALIAS"));
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
        {"-o", "--output", "--output-file"}, "FNAME",
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
        {"--chunk", "--from-chunk"}, "N",
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
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_STATIC_PATH"));
    add_opt(llama_arg(
        {"--embedding", "--embeddings"},
        format("restrict to only support embedding use case; use only with dedicated embedding models (default: %s)", params.embedding ? "enabled" : "disabled"),
        [](gpt_params & params) {
            params.embedding = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_EMBEDDINGS"));
    add_opt(llama_arg(
        {"--reranking", "--rerank"},
        format("enable reranking endpoint on server (default: %s)", params.reranking ? "enabled" : "disabled"),
        [](gpt_params & params) {
            params.reranking = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_RERANKING"));
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
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SSL_KEY_FILE"));
    add_opt(llama_arg(
        {"--ssl-cert-file"}, "FNAME",
        "path to file a PEM-encoded SSL certificate",
        [](gpt_params & params, const std::string & value) {
            params.ssl_file_cert = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SSL_CERT_FILE"));
    add_opt(llama_arg(
        {"-to", "--timeout"}, "N",
        format("server read/write timeout in seconds (default: %d)", params.timeout_read),
        [](gpt_params & params, int value) {
            params.timeout_read  = value;
            params.timeout_write = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_TIMEOUT"));
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
    add_opt(llama_arg(
        {"--log-disable"},
        "Log disable",
        [](gpt_params &) {
            gpt_log_pause(gpt_log_main());
        }
    ));
    add_opt(llama_arg(
        {"--log-file"}, "FNAME",
        "Log to file",
        [](gpt_params &, const std::string & value) {
            gpt_log_set_file(gpt_log_main(), value.c_str());
        }
    ));
    add_opt(llama_arg(
        {"--log-colors"},
        "Enable colored logging",
        [](gpt_params &) {
            gpt_log_set_colors(gpt_log_main(), true);
        }
    ).set_env("LLAMA_LOG_COLORS"));
    add_opt(llama_arg(
        {"-v", "--verbose", "--log-verbose"},
        "Set verbosity level to infinity (i.e. log all messages, useful for debugging)",
        [](gpt_params & params) {
            params.verbosity = INT_MAX;
            gpt_log_set_verbosity_thold(INT_MAX);
        }
    ));
    add_opt(llama_arg(
        {"-lv", "--verbosity", "--log-verbosity"}, "N",
        "Set the verbosity threshold. Messages with a higher verbosity will be ignored.",
        [](gpt_params & params, int value) {
            params.verbosity = value;
            gpt_log_set_verbosity_thold(value);
        }
    ).set_env("LLAMA_LOG_VERBOSITY"));
    add_opt(llama_arg(
        {"--log-prefix"},
        "Enable prefx in log messages",
        [](gpt_params &) {
            gpt_log_set_prefix(gpt_log_main(), true);
        }
    ).set_env("LLAMA_LOG_PREFIX"));
    add_opt(llama_arg(
        {"--log-timestamps"},
        "Enable timestamps in log messages",
        [](gpt_params &) {
            gpt_log_set_timestamps(gpt_log_main(), true);
        }
    ).set_env("LLAMA_LOG_TIMESTAMPS"));

    return ctx_arg;
}
