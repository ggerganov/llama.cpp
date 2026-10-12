// Measures model loading, array programming and evaluation of the analog backend against the
// CPU backend. This is not a general purpose benchmark: the numbers that llama-bench cannot
// report are the cost of programming the arrays and the accuracy of the memory estimates.
//
// See docs/backend/ANALOG.md and benches/analog/README.md

#include "llama.h"
#include "ggml.h"
#include "ggml-analog.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int64_t g_t0 = 0;

static double since_us() {
    const int64_t t1 = ggml_time_us();
    const double  ret = (double) (t1 - g_t0);
    g_t0 = t1;
    return ret;
}

struct result {
    double load_ms    = 0.0;
    double context_ms = 0.0;
    double first_ms   = 0.0; // first evaluation, includes programming the arrays
    double pp_ms      = 0.0; // median time of a prompt batch
    double tg_ms      = 0.0; // median time of a single token

    // modeled device, from the array model rather than from the emulation
    double pp_dev_ms = 0.0;  // time of one prompt batch
    double tg_dev_ms = 0.0;  // time of one token
    double tg_dev_uj = 0.0;  // energy of one token
};

static bool decode_prompt(llama_context * ctx, int n_prompt, uint32_t seed) {
    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens(n_prompt);

    for (int i = 0; i < n_prompt; i++) {
        if (i == 0 && llama_vocab_get_add_bos(vocab)) {
            tokens[i] = llama_vocab_bos(vocab);
        } else {
            seed = seed*1664525u + 1013904223u;
            tokens[i] = (llama_token) (seed % (uint32_t) n_vocab);
        }
    }

    return llama_decode(ctx, llama_batch_get_one(tokens.data(), n_prompt)) == 0;
}

static void reset_context(llama_context * ctx) {
    llama_memory_clear(llama_get_memory(ctx), true);
}

static double decode_token(llama_context * ctx, int n_gen) {
    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    llama_token token = llama_vocab_bos(vocab);

    for (int i = 0; i < n_gen; i++) {
        if (llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0) {
            fprintf(stderr, "%s: failed to decode\n", __func__);
            exit(1);
        }
        llama_synchronize(ctx);
        token = (llama_token) (rand() % n_vocab);
    }

    return 0.0;
}

// ---------------------------------------------------------------------------
// end to end accuracy: the logits of the same prompts on two backends
// ---------------------------------------------------------------------------

// writes the logits of npromt prompts of n_prompt tokens to a file. The prompts are generated
// from fixed seeds, so two runs of two different backends see exactly the same input
static void logits_measure(llama_context * ctx, const std::string & path, int n_prompt, int npromt) {
    const llama_model * model   = llama_get_model(ctx);
    const int32_t       n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    FILE * f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        fprintf(stderr, "%s: cannot write %s\n", __func__, path.c_str());
        exit(1);
    }

    int32_t hdr[2] = { npromt, n_prompt };
    fwrite(hdr, sizeof(hdr), 1, f);

    for (int i = 0; i < npromt; i++) {
        reset_context(ctx);

        if (!decode_prompt(ctx, n_prompt, 1000 + i)) {
            exit(1);
        }

        const float * logits = llama_get_logits_ith(ctx, n_prompt - 1);

        fwrite(logits, sizeof(float), n_vocab, f);
    }

    fclose(f);
}

struct logits_file {
    int32_t            npromt   = 0;
    int32_t            n_prompt = 0;
    std::vector<float> data;

    size_t n_vocab() const { return npromt > 0 ? data.size()/(size_t) npromt : 0; }
};

static logits_file logits_load(const std::string & path) {
    std::ifstream f(path, std::ios::binary);

    if (!f) {
        fprintf(stderr, "%s: cannot read %s\n", __func__, path.c_str());
        exit(1);
    }

    logits_file lf;

    int32_t hdr[2] = { 0, 0 };
    f.read((char *) hdr, sizeof(hdr));
    lf.npromt   = hdr[0];
    lf.n_prompt = hdr[1];

    f.seekg(0, std::ios::end);
    const std::streamoff bytes = f.tellg() - (std::streamoff) sizeof(hdr);

    if (bytes <= 0 || bytes % (std::streamoff) sizeof(float) != 0) {
        fprintf(stderr, "%s: %s is truncated\n", __func__, path.c_str());
        exit(1);
    }

    lf.data.resize((size_t) bytes/sizeof(float));

    f.seekg((std::streamoff) sizeof(hdr), std::ios::beg);
    f.read((char *) lf.data.data(), bytes);

    return lf;
}

// the same softmax that the samplers use, in double precision
static void logits_softmax(const float * logits, size_t n, double * probs) {
    double max = -HUGE_VAL;

    for (size_t i = 0; i < n; i++) {
        max = std::max(max, (double) logits[i]);
    }

    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        probs[i] = std::exp((double) logits[i] - max);
        sum += probs[i];
    }

    for (size_t i = 0; i < n; i++) {
        probs[i] /= sum;
    }
}

static void logits_compare(const std::string & path_ref, const std::string & path_cmp) {
    const logits_file a = logits_load(path_ref);
    const logits_file b = logits_load(path_cmp);

    if (a.npromt != b.npromt || a.n_vocab() != b.n_vocab()) {
        fprintf(stderr, "%s: the files do not have the same shape\n", __func__);
        exit(1);
    }

    const size_t n_vocab = a.n_vocab();

    printf("logits: %d prompts of %d tokens, vocabulary of %zu (reference: %s)\n",
            a.npromt, a.n_prompt, n_vocab, path_ref.c_str());

    double sum_sq    = 0.0;
    double sum_abs   = 0.0;
    double max_abs   = 0.0;
    double sum_kl_ab = 0.0;
    double sum_kl_ba = 0.0;
    int    n_agree   = 0;

    std::vector<double> pa(n_vocab);
    std::vector<double> pb(n_vocab);

    for (int32_t i = 0; i < a.npromt; i++) {
        const float * la = a.data.data() + (size_t) i*n_vocab;
        const float * lb = b.data.data() + (size_t) i*n_vocab;

        logits_softmax(la, n_vocab, pa.data());
        logits_softmax(lb, n_vocab, pb.data());

        size_t top_a = 0;
        size_t top_b = 0;

        for (size_t v = 0; v < n_vocab; v++) {
            const double d = (double) lb[v] - (double) la[v];

            sum_sq  += d*d;
            sum_abs += std::fabs(d);
            max_abs  = std::max(max_abs, std::fabs(d));

            if (pa[v] > pa[top_a]) { top_a = v; }
            if (pb[v] > pb[top_b]) { top_b = v; }

            if (pa[v] > 0.0) { sum_kl_ab += pa[v]*std::log(pa[v]/std::max(pb[v], 1e-300)); }
            if (pb[v] > 0.0) { sum_kl_ba += pb[v]*std::log(pb[v]/std::max(pa[v], 1e-300)); }
        }

        n_agree += top_a == top_b;
    }

    const size_t n = a.data.size();

    printf("| metric | value |\n");
    printf("| --- | ---: |\n");
    printf("| max abs logit error | %.6f |\n", max_abs);
    printf("| RMS logit error | %.6f |\n", std::sqrt(sum_sq/(double) n));
    printf("| mean abs logit error | %.6f |\n", sum_abs/(double) n);
    printf("| KL divergence (reference || compared) | %.6f |\n", sum_kl_ab/a.npromt);
    printf("| KL divergence (compared || reference) | %.6f |\n", sum_kl_ba/a.npromt);
    printf("| same argmax | %.1f %% |\n", 100.0*n_agree/a.npromt);
}

// modeled device counters, the difference of two snapshots is the cost of the work in between
static struct ggml_analog_stats analog_stats_now(void) {
    struct ggml_analog_stats stats = {};

    if (ggml_backend_analog_get_enabled()) {
        ggml_backend_analog_stats_get(&stats);
    }

    return stats;
}

static result run(const std::string & model_path, int n_prompt, int n_gen, int n_rep, int n_threads, int n_ctx, bool no_fa,
                  const std::string & logits_path, int n_logits) {
    result res;

    // the statistics printed at the end cover the whole session, including the programming of the
    // arrays caused by the first evaluation
    ggml_backend_analog_stats_reset();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.load_mode    = LLAMA_LOAD_MODE_MMAP;

    g_t0 = ggml_time_us();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    res.load_ms = since_us()/1000.0;

    if (model == nullptr) {
        fprintf(stderr, "%s: failed to load the model\n", __func__);
        exit(1);
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = n_ctx;
    cparams.n_batch         = n_ctx;
    cparams.n_ubatch        = n_ctx;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;

    if (no_fa) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    g_t0 = ggml_time_us();
    llama_context * ctx = llama_init_from_model(model, cparams);
    res.context_ms = since_us()/1000.0;

    if (ctx == nullptr) {
        fprintf(stderr, "%s: failed to create the context\n", __func__);
        exit(1);
    }

    // the first evaluation programs the arrays of the analog device
    reset_context(ctx);

    g_t0 = ggml_time_us();
    if (!decode_prompt(ctx, n_prompt, 1)) {
        exit(1);
    }
    res.first_ms = since_us()/1000.0;

    std::vector<double> pp_times;
    std::vector<double> tg_times;

    for (int i = 0; i < n_rep; i++) {
        struct ggml_analog_stats before;
        struct ggml_analog_stats after;

        reset_context(ctx);

        before = analog_stats_now();

        g_t0 = ggml_time_us();
        if (!decode_prompt(ctx, n_prompt, 100 + i)) {
            exit(1);
        }
        pp_times.push_back(since_us()/1000.0);

        after = analog_stats_now();

        res.pp_dev_ms += (after.est_time_ms - before.est_time_ms)/n_rep;

        reset_context(ctx);

        before = analog_stats_now();

        g_t0 = ggml_time_us();
        decode_token(ctx, n_gen);
        tg_times.push_back(since_us()/1000.0/(double) n_gen);

        after = analog_stats_now();

        res.tg_dev_ms += (after.est_time_ms - before.est_time_ms)/(n_rep*(double) n_gen);
        res.tg_dev_uj += (after.est_energy_mj - before.est_energy_mj)*1000.0/(n_rep*(double) n_gen);
    }

    std::sort(pp_times.begin(), pp_times.end());
    std::sort(tg_times.begin(), tg_times.end());

    res.pp_ms = pp_times[pp_times.size()/2];
    res.tg_ms = tg_times[tg_times.size()/2];

    if (!logits_path.empty()) {
        logits_measure(ctx, logits_path, n_prompt, n_logits);
    }

    llama_free(ctx);
    llama_model_free(model);

    return res;
}

int main(int argc, char ** argv) {
    std::string model_path;
    int n_prompt  = 64;
    int n_gen     = 16;
    int n_rep     = 3;
    int n_threads = 4;
    int n_ctx     = 512;
    bool quiet    = false;
    bool no_fa    = false;
    std::string logits_path;
    std::string compare_ref;
    int n_logits = 8;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];

        if (arg == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "-p" && i + 1 < argc) {
            n_prompt = atoi(argv[++i]);
        } else if (arg == "-n" && i + 1 < argc) {
            n_gen = atoi(argv[++i]);
        } else if (arg == "-r" && i + 1 < argc) {
            n_rep = atoi(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (arg == "-c" && i + 1 < argc) {
            n_ctx = atoi(argv[++i]);
        } else if (arg == "-q") {
            quiet = true;
        } else if (arg == "--no-fa") {
            no_fa = true;
        } else if (arg == "--logits" && i + 1 < argc) {
            logits_path = argv[++i];
        } else if (arg == "--n-logits" && i + 1 < argc) {
            n_logits = atoi(argv[++i]);
        } else if (arg == "--compare" && i + 2 < argc) {
            compare_ref = argv[++i];
            const std::string compare_cmp = argv[++i];

            logits_compare(compare_ref, compare_cmp);
            return 0;
        } else {
            fprintf(stderr, "usage: %s -m model.gguf [-p n_prompt] [-n n_gen] [-r n_rep] [-t n_threads]"
                            " [-c n_ctx] [--no-fa] [--logits FILE] [--n-logits N] [-q]\n"
                            "       %s --compare FILE REFERENCE\n", argv[0], argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf [-p n_prompt] [-n n_gen] [-r n_rep] [-t n_threads]"
                        " [-c n_ctx] [--no-fa] [--logits FILE] [--n-logits N] [-q]\n"
                        "       %s --compare FILE REFERENCE\n", argv[0], argv[0]);
        return 1;
    }

    if (quiet) {
        llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
    }

    llama_backend_init();
    srand(1234);

    printf("| backend        |    load_ms | context_ms |  first_ms |     pp_ms |     tg_ms |\n");
    fflush(stdout);

    const result res = run(model_path, n_prompt, n_gen, n_rep, n_threads, n_ctx, no_fa, logits_path, n_logits);

    printf("| %-14s | %9.2f | %9.2f | %9.2f | %9.2f | %9.2f |\n",
            ggml_backend_analog_get_enabled() ? "analog" : "cpu",
            res.load_ms, res.context_ms, res.first_ms, res.pp_ms, res.tg_ms);

    if (ggml_backend_analog_get_enabled()) {
        // what the emulated hardware would have needed for the same work
        printf("analog: modeled device: pp %.3f ms/batch, tg %.4f ms/token, %.3f uJ/token\n",
                res.pp_dev_ms, res.tg_dev_ms, res.tg_dev_uj);
        fflush(stdout);

        struct ggml_analog_stats stats;
        ggml_backend_analog_stats_get(&stats);

        fprintf(stderr,
                "analog: stats: %" PRId64 " arrays (%.3f MiB, %.1f ms programming), "
                "%" PRId64 " MVM (%.1f ms emulated, %.1f ms readout), %" PRId64 " array passes, "
                "%.2f ms / %.3f mJ modeled\n",
                stats.n_tensors, stats.bytes/1024.0/1024.0, stats.time_program_ms,
                stats.n_mvm, stats.time_compute_ms, stats.time_readout_ms, stats.n_array_passes,
                stats.est_time_ms, stats.est_energy_mj);
    }

    llama_backend_free();

    return 0;
}
