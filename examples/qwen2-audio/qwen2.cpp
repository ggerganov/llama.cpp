#include "qwen2.h"
#include "audio-projector.h"
#include "common-nexa.h"

#include "whisper.h"
#include "llama.h"
#include "common.h"
#include "log.h"
// #include "arg.h"
#include "sampling.h"
#include "llama-impl.h"

#include <cmath>
#include <fstream>
#include <cstdio>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#include <cstring>
#include <iostream>

//
// Constants
//
void* internal_chars = nullptr;

static const char *AUDIO_TOKEN = "<|AUDIO|>";

//
// Whisper
//

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
            // params.fname_inp.push_back(arg);
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
        else if (arg == "--mmproj")
        {
            continue;
            // params.mmproj = argv[++i];
        }
        else if (arg == "-ngl" || arg == "--gpu-layers" || arg == "--n-gpu-layers")
        {
            continue;
            // params.n_gpu_layers = std::stoi(argv[++i]);
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

struct whisper_context *whisper_init_context(whisper_params &params)
{
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
            return NULL;
        }
    }

    struct whisper_context *ctx = whisper_encoder_init_from_file_with_params(params.model.c_str(), cparams);
    if (ctx == nullptr)
    {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return NULL;
    }

    // initialize openvino encoder. this has no effect on whisper.cpp builds that don't have OpenVINO configured
    whisper_ctx_init_openvino_encoder(ctx, nullptr, params.openvino_encode_device.c_str(), nullptr);

    return ctx;
}

static struct whisper_full_params get_whisper_inference_params_from_whisper_params(whisper_params &params)
{

    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

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

    return wparams;
}

//
// Omni
//

static void omni_print_usage(int, char **argv)
{
    LOG("\n example usage:\n");
    LOG("\n     %s --model <omni/ggml-model.gguf> --mmproj <whisper/model-f16.gguf> --file <path/to/an/audio.wav> [-p \"describe the audio in detail.\"]\n", argv[0]);
    LOG("\n note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

bool omni_context_params_parse(int argc, char **argv, omni_context_params &params)
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg[0] != '-')
        {
            continue;
        }

        if (arg == "-h" || arg == "--help")
        {
            omni_print_usage(argc, argv);
            exit(0);
        }
        if (arg == "--prompt")
        {
            params.prompt = argv[++i];
        }
        else if (arg == "-m" || arg == "--model")
        {
            params.model = argv[++i];
        }
        else if (arg == "-f" || arg == "--file")
        {
            params.file = argv[++i];
        }
        else if (arg == "--mmproj")
        {
            params.mmproj = argv[++i];
        }
        else if (arg == "-ngl" || arg == "--gpu-layers" || arg == "--n-gpu-layers")
        {
            params.n_gpu_layers = std::stoi(argv[++i]);
        }
        else
        {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            omni_print_usage(argc, argv);
            exit(0);
        }
    }

    return true;
}

omni_context_params omni_context_default_params()
{
    omni_context_params params;
    params.model = "";
    params.mmproj = "";
    params.file = "";
    params.prompt = "this conversation talks about";
    params.n_gpu_layers = -1;
    return params;
}

struct omni_params
{
    gpt_params gpt;
    whisper_params whisper;
};

bool omni_params_parse(int argc, char **argv, omni_params &params)
{
    if (!gpt_params_parse(argc, argv, params.gpt))
    {
        return false;
    }

    if (!whisper_params_parse(argc, argv, params.whisper))
    {
        whisper_print_usage(argc, argv, params.whisper);
        return false;
    }

    if (params.gpt.model.empty() || params.gpt.mmproj.empty() || params.whisper.fname_inp.empty())
    {
        omni_print_usage(argc, argv);
        return false;
    }

    params.whisper.model = params.gpt.mmproj;

    return true;
}

static omni_params get_omni_params_from_context_params(omni_context_params &params)
{
    omni_params all_params;

    // Initialize gpt params
    all_params.gpt.n_gpu_layers = params.n_gpu_layers;
    all_params.gpt.model = params.model;
    all_params.gpt.prompt = params.prompt;

    // Initialize whisper params
    all_params.whisper.model = params.mmproj;
    all_params.whisper.fname_inp = {params.file};

    if (all_params.gpt.n_threads <= 0)
    {
        all_params.gpt.n_threads = std::thread::hardware_concurrency();
    }
    return all_params;
}

static bool eval_tokens(struct llama_context *ctx_llama, std::vector<llama_token> tokens, int n_batch, int *n_past)
{
    int N = (int)tokens.size();
    for (int i = 0; i < N; i += n_batch)
    {
        int n_eval = (int)tokens.size() - i;
        if (n_eval > n_batch)
        {
            n_eval = n_batch;
        }
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval, *n_past, 0)))
        {
            LLAMA_LOG_ERROR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context *ctx_llama, int id, int *n_past)
{
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

static bool eval_string(struct llama_context *ctx_llama, const char *str, int n_batch, int *n_past, bool add_bos)
{
    std::string str2 = str;
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx_llama, str2, add_bos, true);
    eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
    return true;
}

static const char * sample(struct llama_sampling_context * ctx_sampling,
                           struct llama_context * ctx_llama,
                           int * n_past) {
    const llama_token id = llama_sampling_sample(ctx_sampling, ctx_llama, NULL);
    llama_sampling_accept(ctx_sampling, ctx_llama, id, true);
    static std::string ret;
    if (llama_token_is_eog(llama_get_model(ctx_llama), id)) {
        ret = "</s>";
    } else {
        ret = llama_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past);
    return ret.c_str();
}

static size_t find_audio_token(const std::string &prompt)
{
    return prompt.find(AUDIO_TOKEN);
}

struct omni_context *omni_init_context(omni_context_params &params)
{

    omni_params all_params = get_omni_params_from_context_params(params);

    // llama
    LLAMA_LOG_INFO("------- llama --------\n");

    std::string prompt = all_params.gpt.prompt;
    if (prompt.empty())
    {
        prompt = "this conversation talks about";
    }

    llama_backend_init();
    llama_numa_init(all_params.gpt.numa);

    llama_model_params model_params = llama_model_params_from_gpt_params(all_params.gpt);

    llama_model *model = llama_load_model_from_file(all_params.gpt.model.c_str(), model_params);
    if (model == NULL)
    {
        LLAMA_LOG_ERROR("%s: unable to load model\n", __func__);
        return NULL;
    }

    llama_context_params ctx_params = llama_context_params_from_gpt_params(all_params.gpt);
    ctx_params.n_ctx = all_params.gpt.n_ctx < 2048 ? 2048 : all_params.gpt.n_ctx; // we need a longer context size to process image embeddings

    llama_context *ctx_llama = llama_new_context_with_model(model, ctx_params);

    if (ctx_llama == NULL)
    {
        LLAMA_LOG_ERROR("%s: failed to create the llama_context\n", __func__);
        return NULL;
    }

    // whisper
    LLAMA_LOG_INFO("------- whisper --------\n");

    whisper_context *ctx_whisper = whisper_init_context(all_params.whisper);

    // projector
    LLAMA_LOG_INFO("------- projector --------\n");

    audio_projector *projector = new audio_projector();
    if (!projector->load_from_gguf(all_params.whisper.model))
    {
        fprintf(stderr, "Failed to load model.\n");
        return NULL;
    }

    auto *ctx_omni = (struct omni_context *)malloc(sizeof(omni_context));

    ctx_omni->ctx_llama = ctx_llama;
    ctx_omni->ctx_whisper = ctx_whisper;
    ctx_omni->model = model;
    ctx_omni->projector = projector;

    LLAMA_LOG_INFO("------- qwen2 context initialized --------\n");

    return ctx_omni;
}

void omni_free(struct omni_context *ctx_omni)
{

    if(internal_chars != nullptr)
    {
        free(internal_chars);
        internal_chars = nullptr;
    }
    if (ctx_omni->ctx_whisper)
    {
        whisper_free(ctx_omni->ctx_whisper);
        ctx_omni->ctx_whisper = NULL;
    }
    if (ctx_omni->projector)
    {
        delete ctx_omni->projector;
    }

    llama_free(ctx_omni->ctx_llama);
    llama_free_model(ctx_omni->model);
    llama_backend_free();
    free(ctx_omni);
}

static bool omni_eval_audio_embed(llama_context *ctx_llama, ggml_tensor *audio_embed, int n_batch, int *n_past)
{
    int n_embd = llama_n_embd(llama_get_model(ctx_llama));

    int n_audio_embed = audio_embed->ne[1];
    GGML_ASSERT(audio_embed->ne[0] == n_embd);

    size_t audio_embed_size = ggml_nbytes(audio_embed);
    float *audio_embed_data = (float *)malloc(audio_embed_size);
    ggml_backend_tensor_get(audio_embed, audio_embed_data, 0, audio_embed_size);

    for (int i = 0; i < n_audio_embed; i += n_batch)
    {
        int n_eval = n_audio_embed - i;
        if (n_eval > n_batch)
        {
            n_eval = n_batch;
        }
        llama_batch batch = {
            /* n_tokens */  int32_t(n_eval),
            /* token */     nullptr,
            /* embd */      (audio_embed_data + i * n_embd),
            /* pos */       nullptr,
            /* n_seq_id */  nullptr,
            /* seq_id */    nullptr,
            /* logits */    nullptr,
            /* all_pos_0 */ *n_past,
            /* all_pos_1 */  1,
            /* all_seq_id */ 0,
        };
        if (llama_decode(ctx_llama, batch))
        {
            LLAMA_LOG_ERROR("%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
    }
    free(audio_embed_data);
    return true;
}

ggml_tensor *omni_process_audio(struct omni_context *ctx_omni, omni_params &params)
{
    auto fname_inp = params.whisper.fname_inp[0];

    std::vector<float> pcmf32;               // mono-channel F32 PCM
    std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

    if (!::read_wav(fname_inp, pcmf32, pcmf32s, params.whisper.diarize))
    {
        LLAMA_LOG_ERROR("error: failed to read WAV file '%s'\n", fname_inp.c_str());
        return NULL;
    }

    whisper_full_params wparams = get_whisper_inference_params_from_whisper_params(params.whisper);

    if (whisper_encode_wo_cross_parallel(ctx_omni->ctx_whisper, wparams, pcmf32.data(), pcmf32.size(), params.whisper.n_processors) != 0)
    {
        LLAMA_LOG_ERROR("%s: failed to process audio\n", __func__);
        return NULL;
    }

    ggml_tensor *embd_enc = whisper_full_get_embd_enc(ctx_omni->ctx_whisper);
#ifdef NEXA_DEBUG
    print_ggml_tensor_shape("embd_enc", embd_enc);
#endif

    ggml_tensor *embed_proj = audio_projector_inference(*ctx_omni->projector, embd_enc);
#ifdef NEXA_DEBUG
    print_ggml_tensor_shape("embed_proj", embed_proj);
#endif

    return embed_proj;
}

const char* omni_process_prompt(struct omni_context *ctx_omni, ggml_tensor *audio_embed, omni_params &params, const std::string &prompt)
{
    int n_past = 0;

    int n_audio_embed = audio_embed->ne[1];
    GGML_ASSERT(params.gpt.n_predict < 0 || params.gpt.n_predict > n_audio_embed);

    const int max_tgt_len = params.gpt.n_predict < 0 ? 256 + n_audio_embed : params.gpt.n_predict;
    std::string system_prompt, user_prompt;
    size_t audio_pos = find_audio_token(prompt);
    if (audio_pos != std::string::npos)
    {
        system_prompt = prompt.substr(0, audio_pos);
        user_prompt = prompt.substr(audio_pos + std::string(AUDIO_TOKEN).length());
        // LLAMA_LOG_INFO("system_prompt: %s\n", system_prompt.c_str());
        // LLAMA_LOG_INFO("user_prompt: %s\n", user_prompt.c_str());
    }
    // NEXA AI : major difference with nano-omni is in the prompt handling
    else
    // template from : https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
    // <|im_start|>system
    // You are a helpful assistant.<|im_end|>
    // <|im_start|>user
    // Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>
    // What's that sound?<|im_end|>
    // <|im_start|>assistant
    {
        system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAudio 1: <|audio_bos|>";
        user_prompt = "<|audio_eos|>\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    }

    eval_string(ctx_omni->ctx_llama, system_prompt.c_str(), params.gpt.n_batch, &n_past, true);
    omni_eval_audio_embed(ctx_omni->ctx_llama, audio_embed, params.gpt.n_batch, &n_past);
    eval_string(ctx_omni->ctx_llama, user_prompt.c_str(), params.gpt.n_batch, &n_past, false);

    // generate the response

    LOG("\n");

    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params.gpt.sparams);
    if (!ctx_sampling) {
        fprintf(stderr, "%s: failed to initialize sampling subsystem\n", __func__);
        exit(1);
    }

    std::string response = "";
    for (int i = 0; i < max_tgt_len; i++)
    {
        const char * tmp = sample(ctx_sampling, ctx_omni->ctx_llama, &n_past);
        if (strcmp(tmp, "</s>") == 0)
            break;
        if (strstr(tmp, "###"))
            break; // Yi-VL behavior
        // printf("%s", tmp);
        if (strstr(response.c_str(), "<|im_end|>"))
            break; // Yi-34B llava-1.6 - for some reason those decode not as the correct token (tokenizer works)
        if (strstr(response.c_str(), "<|im_start|>"))
            break; // Yi-34B llava-1.6
        if (strstr(response.c_str(), "USER:"))
            break; // mistral llava-1.6

        fflush(stdout);
        response += tmp;
    }

    llama_sampling_free(ctx_sampling);
    printf("\n");

    if(internal_chars != nullptr) { free(internal_chars); }
    internal_chars = malloc(sizeof(char)*(response.size()+1));
    strncpy((char*)(internal_chars), response.c_str(), response.size());
    ((char*)(internal_chars))[response.size()] = '\0';
    return (const char*)(internal_chars);
}

const char* omni_process_full(struct omni_context *ctx_omni, omni_context_params &params)
{
    omni_params all_params = get_omni_params_from_context_params(params);

    ggml_tensor *audio_embed = omni_process_audio(ctx_omni, all_params);
    return omni_process_prompt(ctx_omni, audio_embed, all_params, all_params.gpt.prompt);
}
