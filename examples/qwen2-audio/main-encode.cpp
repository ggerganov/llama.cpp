#include "common.h"
#include "common-nexa.h"

#include "whisper.h"
#include "grammar-parser.h"

#include <cmath>
#include <fstream>
#include <cstdio>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#include <cstring>

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

// helper function to replace substrings
static void replace_all(std::string &s, const std::string &search, const std::string &replace)
{
    for (size_t pos = 0;; pos += replace.length())
    {
        pos = s.find(search, pos);
        if (pos == std::string::npos)
            break;
        s.erase(pos, search.length());
        s.insert(pos, replace);
    }
}

// command-line parameters
struct whisper_params
{
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t n_processors = 1;
    int32_t offset_t_ms = 0;
    int32_t offset_n = 0;
    int32_t duration_ms = 0;
    int32_t progress_step = 5;
    int32_t max_context = -1;
    int32_t max_len = 0;
    int32_t best_of = whisper_full_default_params(WHISPER_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
    int32_t audio_ctx = 0;

    float word_thold = 0.01f;
    float entropy_thold = 2.40f;
    float logprob_thold = -1.00f;
    float grammar_penalty = 100.0f;
    float temperature = 0.0f;
    float temperature_inc = 0.2f;

    bool debug_mode = false;
    bool translate = false;
    bool detect_language = false;
    bool diarize = false;
    bool tinydiarize = false;
    bool split_on_word = false;
    bool no_fallback = false;
    bool no_prints = false;
    bool print_special = false;
    bool print_colors = false;
    bool print_progress = false;
    bool no_timestamps = false;
    bool log_score = false;
    bool use_gpu = true;
    bool flash_attn = false;

    std::string language = "en";
    std::string prompt;
    std::string font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf";
    std::string model = "models/ggml-base.en.bin";
    std::string grammar;
    std::string grammar_rule;

    // [TDRZ] speaker turn string
    std::string tdrz_speaker_turn = " [SPEAKER_TURN]"; // TODO: set from command line

    // A regular expression that matches tokens to suppress
    std::string suppress_regex;

    std::string openvino_encode_device = "CPU";

    std::string dtw = "";

    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};

    grammar_parser::parse_state grammar_parsed;
};

static void whisper_print_usage(int argc, char **argv, const whisper_params &params);

static char *whisper_param_turn_lowercase(char *in)
{
    int string_len = strlen(in);
    for (int i = 0; i < string_len; i++)
    {
        *(in + i) = tolower((unsigned char)*(in + i));
    }
    return in;
}

static bool whisper_params_parse(int argc, char **argv, whisper_params &params)
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-")
        {
            params.fname_inp.push_back(arg);
            continue;
        }

        if (arg[0] != '-')
        {
            params.fname_inp.push_back(arg);
            continue;
        }

        if (arg == "-h" || arg == "--help")
        {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t" || arg == "--threads")
        {
            params.n_threads = std::stoi(argv[++i]);
        }
        else if (arg == "-p" || arg == "--processors")
        {
            params.n_processors = std::stoi(argv[++i]);
        }
        else if (arg == "-ot" || arg == "--offset-t")
        {
            params.offset_t_ms = std::stoi(argv[++i]);
        }
        else if (arg == "-on" || arg == "--offset-n")
        {
            params.offset_n = std::stoi(argv[++i]);
        }
        else if (arg == "-d" || arg == "--duration")
        {
            params.duration_ms = std::stoi(argv[++i]);
        }
        else if (arg == "-mc" || arg == "--max-context")
        {
            params.max_context = std::stoi(argv[++i]);
        }
        else if (arg == "-ml" || arg == "--max-len")
        {
            params.max_len = std::stoi(argv[++i]);
        }
        else if (arg == "-bo" || arg == "--best-of")
        {
            params.best_of = std::stoi(argv[++i]);
        }
        else if (arg == "-bs" || arg == "--beam-size")
        {
            params.beam_size = std::stoi(argv[++i]);
        }
        else if (arg == "-ac" || arg == "--audio-ctx")
        {
            params.audio_ctx = std::stoi(argv[++i]);
        }
        else if (arg == "-wt" || arg == "--word-thold")
        {
            params.word_thold = std::stof(argv[++i]);
        }
        else if (arg == "-et" || arg == "--entropy-thold")
        {
            params.entropy_thold = std::stof(argv[++i]);
        }
        else if (arg == "-lpt" || arg == "--logprob-thold")
        {
            params.logprob_thold = std::stof(argv[++i]);
        }
        else if (arg == "-tp" || arg == "--temperature")
        {
            params.temperature = std::stof(argv[++i]);
        }
        else if (arg == "-tpi" || arg == "--temperature-inc")
        {
            params.temperature_inc = std::stof(argv[++i]);
        }
        else if (arg == "-debug" || arg == "--debug-mode")
        {
            params.debug_mode = true;
        }
        else if (arg == "-tr" || arg == "--translate")
        {
            params.translate = true;
        }
        else if (arg == "-di" || arg == "--diarize")
        {
            params.diarize = true;
        }
        else if (arg == "-tdrz" || arg == "--tinydiarize")
        {
            params.tinydiarize = true;
        }
        else if (arg == "-sow" || arg == "--split-on-word")
        {
            params.split_on_word = true;
        }
        else if (arg == "-nf" || arg == "--no-fallback")
        {
            params.no_fallback = true;
        }
        else if (arg == "-fp" || arg == "--font-path")
        {
            params.font_path = argv[++i];
        }
        else if (arg == "-np" || arg == "--no-prints")
        {
            params.no_prints = true;
        }
        else if (arg == "-ps" || arg == "--print-special")
        {
            params.print_special = true;
        }
        else if (arg == "-pc" || arg == "--print-colors")
        {
            params.print_colors = true;
        }
        else if (arg == "-pp" || arg == "--print-progress")
        {
            params.print_progress = true;
        }
        else if (arg == "-nt" || arg == "--no-timestamps")
        {
            params.no_timestamps = true;
        }
        else if (arg == "-l" || arg == "--language")
        {
            params.language = whisper_param_turn_lowercase(argv[++i]);
        }
        else if (arg == "-dl" || arg == "--detect-language")
        {
            params.detect_language = true;
        }
        else if (arg == "--prompt")
        {
            params.prompt = argv[++i];
        }
        else if (arg == "-m" || arg == "--model")
        {
            params.model = argv[++i];
        }
        else if (arg == "-f" || arg == "--file")
        {
            params.fname_inp.emplace_back(argv[++i]);
        }
        else if (arg == "-oved" || arg == "--ov-e-device")
        {
            params.openvino_encode_device = argv[++i];
        }
        else if (arg == "-dtw" || arg == "--dtw")
        {
            params.dtw = argv[++i];
        }
        else if (arg == "-ls" || arg == "--log-score")
        {
            params.log_score = true;
        }
        else if (arg == "-ng" || arg == "--no-gpu")
        {
            params.use_gpu = false;
        }
        else if (arg == "-fa" || arg == "--flash-attn")
        {
            params.flash_attn = true;
        }
        else if (arg == "--suppress-regex")
        {
            params.suppress_regex = argv[++i];
        }
        else if (arg == "--grammar")
        {
            params.grammar = argv[++i];
        }
        else if (arg == "--grammar-rule")
        {
            params.grammar_rule = argv[++i];
        }
        else if (arg == "--grammar-penalty")
        {
            params.grammar_penalty = std::stof(argv[++i]);
        }
        else
        {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

static void whisper_print_usage(int /*argc*/, char **argv, const whisper_params &params)
{
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options] file0.wav file1.wav ...\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,        --help              [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,      --threads N         [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -p N,      --processors N      [%-7d] number of processors to use during computation\n", params.n_processors);
    fprintf(stderr, "  -ot N,     --offset-t N        [%-7d] time offset in milliseconds\n", params.offset_t_ms);
    fprintf(stderr, "  -on N,     --offset-n N        [%-7d] segment index offset\n", params.offset_n);
    fprintf(stderr, "  -d  N,     --duration N        [%-7d] duration of audio to process in milliseconds\n", params.duration_ms);
    fprintf(stderr, "  -mc N,     --max-context N     [%-7d] maximum number of text context tokens to store\n", params.max_context);
    fprintf(stderr, "  -ml N,     --max-len N         [%-7d] maximum segment length in characters\n", params.max_len);
    fprintf(stderr, "  -sow,      --split-on-word     [%-7s] split on word rather than on token\n", params.split_on_word ? "true" : "false");
    fprintf(stderr, "  -bo N,     --best-of N         [%-7d] number of best candidates to keep\n", params.best_of);
    fprintf(stderr, "  -bs N,     --beam-size N       [%-7d] beam size for beam search\n", params.beam_size);
    fprintf(stderr, "  -ac N,     --audio-ctx N       [%-7d] audio context size (0 - all)\n", params.audio_ctx);
    fprintf(stderr, "  -wt N,     --word-thold N      [%-7.2f] word timestamp probability threshold\n", params.word_thold);
    fprintf(stderr, "  -et N,     --entropy-thold N   [%-7.2f] entropy threshold for decoder fail\n", params.entropy_thold);
    fprintf(stderr, "  -lpt N,    --logprob-thold N   [%-7.2f] log probability threshold for decoder fail\n", params.logprob_thold);
    fprintf(stderr, "  -tp,       --temperature N     [%-7.2f] The sampling temperature, between 0 and 1\n", params.temperature);
    fprintf(stderr, "  -tpi,      --temperature-inc N [%-7.2f] The increment of temperature, between 0 and 1\n", params.temperature_inc);
    fprintf(stderr, "  -debug,    --debug-mode        [%-7s] enable debug mode (eg. dump log_mel)\n", params.debug_mode ? "true" : "false");
    fprintf(stderr, "  -tr,       --translate         [%-7s] translate from source language to english\n", params.translate ? "true" : "false");
    fprintf(stderr, "  -di,       --diarize           [%-7s] stereo audio diarization\n", params.diarize ? "true" : "false");
    fprintf(stderr, "  -tdrz,     --tinydiarize       [%-7s] enable tinydiarize (requires a tdrz model)\n", params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -nf,       --no-fallback       [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -fp,       --font-path         [%-7s] path to a monospace font for karaoke video\n", params.font_path.c_str());
    fprintf(stderr, "  -np,       --no-prints         [%-7s] do not print anything other than the results\n", params.no_prints ? "true" : "false");
    fprintf(stderr, "  -ps,       --print-special     [%-7s] print special tokens\n", params.print_special ? "true" : "false");
    fprintf(stderr, "  -pc,       --print-colors      [%-7s] print colors\n", params.print_colors ? "true" : "false");
    fprintf(stderr, "  -pp,       --print-progress    [%-7s] print progress\n", params.print_progress ? "true" : "false");
    fprintf(stderr, "  -nt,       --no-timestamps     [%-7s] do not print timestamps\n", params.no_timestamps ? "true" : "false");
    fprintf(stderr, "  -l LANG,   --language LANG     [%-7s] spoken language ('auto' for auto-detect)\n", params.language.c_str());
    fprintf(stderr, "  -dl,       --detect-language   [%-7s] exit after automatically detecting language\n", params.detect_language ? "true" : "false");
    fprintf(stderr, "             --prompt PROMPT     [%-7s] initial prompt (max n_text_ctx/2 tokens)\n", params.prompt.c_str());
    fprintf(stderr, "  -m FNAME,  --model FNAME       [%-7s] model path\n", params.model.c_str());
    fprintf(stderr, "  -f FNAME,  --file FNAME        [%-7s] input WAV file path\n", "");
    fprintf(stderr, "  -oved D,   --ov-e-device DNAME [%-7s] the OpenVINO device used for encode inference\n", params.openvino_encode_device.c_str());
    fprintf(stderr, "  -dtw MODEL --dtw MODEL         [%-7s] compute token-level timestamps\n", params.dtw.c_str());
    fprintf(stderr, "  -ls,       --log-score         [%-7s] log best decoder scores of tokens\n", params.log_score ? "true" : "false");
    fprintf(stderr, "  -ng,       --no-gpu            [%-7s] disable GPU\n", params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,       --flash-attn        [%-7s] flash attention\n", params.flash_attn ? "true" : "false");
    fprintf(stderr, "  --suppress-regex REGEX         [%-7s] regular expression matching tokens to suppress\n", params.suppress_regex.c_str());
    fprintf(stderr, "  --grammar GRAMMAR              [%-7s] GBNF grammar to guide decoding\n", params.grammar.c_str());
    fprintf(stderr, "  --grammar-rule RULE            [%-7s] top-level GBNF grammar rule name\n", params.grammar_rule.c_str());
    fprintf(stderr, "  --grammar-penalty N            [%-7.1f] scales down logits of nongrammar tokens\n", params.grammar_penalty);
    fprintf(stderr, "\n");
}

struct whisper_print_user_data
{
    const whisper_params *params;

    const std::vector<std::vector<float>> *pcmf32s;
    int progress_prev;
};

static std::string estimate_diarization_speaker(std::vector<std::vector<float>> pcmf32s, int64_t t0, int64_t t1, bool id_only = false)
{
    std::string speaker = "";
    const int64_t n_samples = pcmf32s[0].size();

    const int64_t is0 = timestamp_to_sample(t0, n_samples, WHISPER_SAMPLE_RATE);
    const int64_t is1 = timestamp_to_sample(t1, n_samples, WHISPER_SAMPLE_RATE);

    double energy0 = 0.0f;
    double energy1 = 0.0f;

    for (int64_t j = is0; j < is1; j++)
    {
        energy0 += fabs(pcmf32s[0][j]);
        energy1 += fabs(pcmf32s[1][j]);
    }

    if (energy0 > 1.1 * energy1)
    {
        speaker = "0";
    }
    else if (energy1 > 1.1 * energy0)
    {
        speaker = "1";
    }
    else
    {
        speaker = "?";
    }

    // printf("is0 = %lld, is1 = %lld, energy0 = %f, energy1 = %f, speaker = %s\n", is0, is1, energy0, energy1, speaker.c_str());

    if (!id_only)
    {
        speaker.insert(0, "(speaker ");
        speaker.append(")");
    }

    return speaker;
}

static void whisper_print_progress_callback(struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, int progress, void *user_data)
{
    int progress_step = ((whisper_print_user_data *)user_data)->params->progress_step;
    int *progress_prev = &(((whisper_print_user_data *)user_data)->progress_prev);
    if (progress >= *progress_prev + progress_step)
    {
        *progress_prev += progress_step;
        fprintf(stderr, "%s: progress = %3d%%\n", __func__, progress);
    }
}

static void whisper_print_segment_callback(struct whisper_context *ctx, struct whisper_state * /*state*/, int n_new, void *user_data)
{
    const auto &params = *((whisper_print_user_data *)user_data)->params;
    const auto &pcmf32s = *((whisper_print_user_data *)user_data)->pcmf32s;

    const int n_segments = whisper_full_n_segments(ctx);

    std::string speaker = "";

    int64_t t0 = 0;
    int64_t t1 = 0;

    // print the last n_new segments
    const int s0 = n_segments - n_new;

    if (s0 == 0)
    {
        printf("\n");
    }

    for (int i = s0; i < n_segments; i++)
    {
        if (!params.no_timestamps || params.diarize)
        {
            t0 = whisper_full_get_segment_t0(ctx, i);
            t1 = whisper_full_get_segment_t1(ctx, i);
        }

        if (!params.no_timestamps)
        {
            printf("[%s --> %s]  ", to_timestamp(t0).c_str(), to_timestamp(t1).c_str());
        }

        if (params.diarize && pcmf32s.size() == 2)
        {
            speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
        }

        if (params.print_colors)
        {
            for (int j = 0; j < whisper_full_n_tokens(ctx, i); ++j)
            {
                if (params.print_special == false)
                {
                    const whisper_token id = whisper_full_get_token_id(ctx, i, j);
                    if (id >= whisper_token_eot(ctx))
                    {
                        continue;
                    }
                }

                const char *text = whisper_full_get_token_text(ctx, i, j);
                const float p = whisper_full_get_token_p(ctx, i, j);

                const int col = std::max(0, std::min((int)k_colors.size() - 1, (int)(std::pow(p, 3) * float(k_colors.size()))));

                printf("%s%s%s%s", speaker.c_str(), k_colors[col].c_str(), text, "\033[0m");
            }
        }
        else
        {
            const char *text = whisper_full_get_segment_text(ctx, i);

            printf("%s%s", speaker.c_str(), text);
        }

        if (params.tinydiarize)
        {
            if (whisper_full_get_segment_speaker_turn_next(ctx, i))
            {
                printf("%s", params.tdrz_speaker_turn.c_str());
            }
        }

        // with timestamps or speakers: each segment on new line
        if (!params.no_timestamps || params.diarize)
        {
            printf("\n");
        }

        fflush(stdout);
    }
}

static char *escape_double_quotes_and_backslashes(const char *str)
{
    if (str == NULL)
    {
        return NULL;
    }

    size_t escaped_length = strlen(str) + 1;

    for (size_t i = 0; str[i] != '\0'; i++)
    {
        if (str[i] == '"' || str[i] == '\\')
        {
            escaped_length++;
        }
    }

    char *escaped = (char *)calloc(escaped_length, 1); // pre-zeroed
    if (escaped == NULL)
    {
        return NULL;
    }

    size_t pos = 0;
    for (size_t i = 0; str[i] != '\0'; i++)
    {
        if (str[i] == '"' || str[i] == '\\')
        {
            escaped[pos++] = '\\';
        }
        escaped[pos++] = str[i];
    }

    // no need to set zero due to calloc() being used prior

    return escaped;
}

// double quote should be escaped by another double quote. (rfc4180)
static char *escape_double_quotes_in_csv(const char *str)
{
    if (str == NULL)
    {
        return NULL;
    }

    size_t escaped_length = strlen(str) + 1;

    for (size_t i = 0; str[i] != '\0'; i++)
    {
        if (str[i] == '"')
        {
            escaped_length++;
        }
    }

    char *escaped = (char *)calloc(escaped_length, 1); // pre-zeroed
    if (escaped == NULL)
    {
        return NULL;
    }

    size_t pos = 0;
    for (size_t i = 0; str[i] != '\0'; i++)
    {
        if (str[i] == '"')
        {
            escaped[pos++] = '"';
        }
        escaped[pos++] = str[i];
    }

    // no need to set zero due to calloc() being used prior

    return escaped;
}

static void cb_log_disable(enum ggml_log_level, const char *, void *) {}

int main(int argc, char **argv)
{
    whisper_params params;

    // If the only argument starts with "@", read arguments line-by-line
    // from the given file.
    std::vector<std::string> vec_args;
    if (argc == 2 && argv != nullptr && argv[1] != nullptr && argv[1][0] == '@')
    {
        // Save the name of the executable.
        vec_args.push_back(argv[0]);

        // Open the response file.
        char const *rspfile = argv[1] + sizeof(char);
        std::ifstream fin(rspfile);
        if (fin.is_open() == false)
        {
            fprintf(stderr, "error: response file '%s' not found\n", rspfile);
            return 1;
        }

        // Read the entire response file.
        std::string line;
        while (std::getline(fin, line))
        {
            vec_args.push_back(line);
        }

        // Use the contents of the response file as the command-line arguments.
        argc = static_cast<int>(vec_args.size());
        argv = static_cast<char **>(alloca(argc * sizeof(char *)));
        for (int i = 0; i < argc; ++i)
        {
            argv[i] = const_cast<char *>(vec_args[i].c_str());
        }
    }

    if (whisper_params_parse(argc, argv, params) == false)
    {
        whisper_print_usage(argc, argv, params);
        return 1;
    }

    // remove non-existent files
    for (auto it = params.fname_inp.begin(); it != params.fname_inp.end();)
    {
        const auto fname_inp = it->c_str();

        if (*it != "-" && !is_file_exist(fname_inp))
        {
            fprintf(stderr, "error: input file not found '%s'\n", fname_inp);
            it = params.fname_inp.erase(it);
            continue;
        }

        it++;
    }

    if (params.fname_inp.empty())
    {
        fprintf(stderr, "error: no input files specified\n");
        whisper_print_usage(argc, argv, params);
        return 2;
    }

    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1)
    {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    if (params.diarize && params.tinydiarize)
    {
        fprintf(stderr, "error: cannot use both --diarize and --tinydiarize\n");
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    if (params.no_prints)
    {
        whisper_log_set(cb_log_disable, NULL);
    }

    // whisper init

    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    if (!params.dtw.empty())
    {
        cparams.dtw_token_timestamps = true;
        cparams.dtw_aheads_preset = WHISPER_AHEADS_NONE;

        if (params.dtw == "tiny")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY;
        if (params.dtw == "tiny.en")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY_EN;
        if (params.dtw == "base")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE;
        if (params.dtw == "base.en")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE_EN;
        if (params.dtw == "small")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL;
        if (params.dtw == "small.en")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL_EN;
        if (params.dtw == "medium")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM;
        if (params.dtw == "medium.en")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM_EN;
        if (params.dtw == "large.v1")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V1;
        if (params.dtw == "large.v2")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V2;
        if (params.dtw == "large.v3")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V3;

        if (cparams.dtw_aheads_preset == WHISPER_AHEADS_NONE)
        {
            fprintf(stderr, "error: unknown DTW preset '%s'\n", params.dtw.c_str());
            return 3;
        }
    }

    struct whisper_context *ctx = whisper_encoder_init_from_file_with_params(params.model.c_str(), cparams);

    if (ctx == nullptr)
    {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 3;
    }

    // initialize openvino encoder. this has no effect on whisper.cpp builds that don't have OpenVINO configured
    whisper_ctx_init_openvino_encoder(ctx, nullptr, params.openvino_encode_device.c_str(), nullptr);

    if (!params.grammar.empty())
    {
        auto &grammar = params.grammar_parsed;
        if (is_file_exist(params.grammar.c_str()))
        {
            // read grammar from file
            std::ifstream ifs(params.grammar.c_str());
            const std::string txt = std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            grammar = grammar_parser::parse(txt.c_str());
        }
        else
        {
            // read grammar from string
            grammar = grammar_parser::parse(params.grammar.c_str());
        }

        // will be empty (default) if there are parse errors
        if (grammar.rules.empty())
        {
            fprintf(stderr, "error: failed to parse grammar \"%s\"\n", params.grammar.c_str());
            return 4;
        }
        else
        {
            fprintf(stderr, "%s: grammar:\n", __func__);
            grammar_parser::print_grammar(stderr, grammar);
            fprintf(stderr, "\n");
        }
    }

    for (int f = 0; f < (int)params.fname_inp.size(); ++f)
    {
        const auto fname_inp = params.fname_inp[f];
        const auto fname_out = f < (int)params.fname_out.size() && !params.fname_out[f].empty() ? params.fname_out[f] : params.fname_inp[f];

        std::vector<float> pcmf32;               // mono-channel F32 PCM
        std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

        if (!::read_wav(fname_inp, pcmf32, pcmf32s, params.diarize))
        {
            fprintf(stderr, "error: failed to read WAV file '%s'\n", fname_inp.c_str());
            continue;
        }

        if (!whisper_is_multilingual(ctx))  // TODO: something off here
        {
            if (params.language != "en" || params.translate)
            {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        if (params.detect_language)
        {
            params.language = "auto";
        }

        if (!params.no_prints)
        {
            // print system information
            fprintf(stderr, "\n");
            fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                    params.n_threads * params.n_processors, std::thread::hardware_concurrency(), whisper_print_system_info());

            // print some info about the processing
            fprintf(stderr, "\n");
            fprintf(stderr, "%s: processing '%s' (%d samples, %.1f sec), %d threads, %d processors, %d beams + best of %d, lang = %s, task = %s, %stimestamps = %d ...\n",
                    __func__, fname_inp.c_str(), int(pcmf32.size()), float(pcmf32.size()) / WHISPER_SAMPLE_RATE,
                    params.n_threads, params.n_processors, params.beam_size, params.best_of,
                    params.language.c_str(),
                    params.translate ? "translate" : "transcribe",
                    params.tinydiarize ? "tdrz = 1, " : "",
                    params.no_timestamps ? 0 : 1);

            fprintf(stderr, "\n");
        }

        // run the inference
        {
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

            const bool use_grammar = (!params.grammar_parsed.rules.empty() && !params.grammar_rule.empty());
            wparams.strategy = (params.beam_size > 1 || use_grammar) ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

            wparams.print_realtime = false;
            wparams.print_progress = params.print_progress;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.print_special = params.print_special;
            wparams.translate = params.translate;
            wparams.language = params.language.c_str();
            wparams.detect_language = params.detect_language;
            wparams.n_threads = params.n_threads;
            wparams.n_max_text_ctx = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
            wparams.offset_ms = params.offset_t_ms;
            wparams.duration_ms = params.duration_ms;

            wparams.token_timestamps = params.max_len > 0;
            wparams.thold_pt = params.word_thold;
            wparams.max_len = params.max_len == 0 ? 60 : params.max_len;
            wparams.split_on_word = params.split_on_word;
            wparams.audio_ctx = params.audio_ctx;

            wparams.debug_mode = params.debug_mode;

            wparams.tdrz_enable = params.tinydiarize; // [TDRZ]

            wparams.suppress_regex = params.suppress_regex.empty() ? nullptr : params.suppress_regex.c_str();

            wparams.initial_prompt = params.prompt.c_str();

            wparams.greedy.best_of = params.best_of;
            wparams.beam_search.beam_size = params.beam_size;

            wparams.temperature_inc = params.no_fallback ? 0.0f : params.temperature_inc;
            wparams.temperature = params.temperature;

            wparams.entropy_thold = params.entropy_thold;
            wparams.logprob_thold = params.logprob_thold;

            wparams.no_timestamps = params.no_timestamps;

            whisper_print_user_data user_data = {&params, &pcmf32s, 0};

            const auto &grammar_parsed = params.grammar_parsed;
            auto grammar_rules = grammar_parsed.c_rules();

            if (use_grammar)
            {
                if (grammar_parsed.symbol_ids.find(params.grammar_rule) == grammar_parsed.symbol_ids.end())
                {
                    fprintf(stderr, "%s: warning: grammar rule '%s' not found - skipping grammar sampling\n", __func__, params.grammar_rule.c_str());
                }
                else
                {
                    wparams.grammar_rules = grammar_rules.data();
                    wparams.n_grammar_rules = grammar_rules.size();
                    wparams.i_start_rule = grammar_parsed.symbol_ids.at(params.grammar_rule);
                    wparams.grammar_penalty = params.grammar_penalty;
                }
            }

            // this callback is called on each new segment
            if (!wparams.print_realtime)
            {
                wparams.new_segment_callback = whisper_print_segment_callback;
                wparams.new_segment_callback_user_data = &user_data;
            }

            if (wparams.print_progress)
            {
                wparams.progress_callback = whisper_print_progress_callback;
                wparams.progress_callback_user_data = &user_data;
            }

            // examples for abort mechanism
            // in examples below, we do not abort the processing, but we could if the flag is set to true

            // the callback is called before every encoder run - if it returns false, the processing is aborted
            {
                static bool is_aborted = false; // NOTE: this should be atomic to avoid data race

                wparams.encoder_begin_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, void *user_data)
                {
                    bool is_aborted = *(bool *)user_data;
                    return !is_aborted;
                };
                wparams.encoder_begin_callback_user_data = &is_aborted;
            }

            // the callback is called before every computation - if it returns true, the computation is aborted
            {
                static bool is_aborted = false; // NOTE: this should be atomic to avoid data race

                wparams.abort_callback = [](void *user_data)
                {
                    bool is_aborted = *(bool *)user_data;
                    return is_aborted;
                };
                wparams.abort_callback_user_data = &is_aborted;
            }

            if (whisper_encode_wo_cross_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                return 10;
            }

            ggml_tensor *embd_enc = whisper_full_get_embd_enc(ctx);
            // print_ggml_tensor("embd_enc", embd_enc, true);
            print_ggml_tensor_shape("embd_enc", embd_enc);
        }

    }

    if (!params.no_prints)
    {
        whisper_print_timings(ctx);
    }
    whisper_free(ctx);

    return 0;
}
