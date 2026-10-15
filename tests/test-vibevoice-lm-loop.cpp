// smoke test for the VibeVoice LM dual-KV audio_token loop, on top of llama_decode
// two contexts share the model: the positive branch has the prompt, the negative (CFG) branch starts at audio_bos
// every frame: last hidden of both -> CFG diffusion sample -> streaming decode -> streaming speech feedback embd fed to both branches
// the frames are forced, the LM is not sampled (random-weight fixtures do not argmax to audio_token)
// usage: test-vibevoice-lm-loop <model.gguf> [n_frames]

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "models/models.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool all_finite(const std::vector<float> & v) {
    return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        d = std::max(d, (double) std::fabs(a[i] - b[i]));
    }
    return d;
}

static bool check(const char * name, const std::vector<float> & v, size_t n_expected, bool require_finite = true) {
    bool ok = true;
    if (v.size() != n_expected) {
        fprintf(stderr, "%s: expected %zu values, got %zu\n", name, n_expected, v.size());
        ok = false;
    }

    double sum2 = 0.0;
    for (float x : v) {
        if (!std::isfinite(x)) {
            fprintf(stderr, "%s: non-finite value%s\n", name, require_finite ? "" : " (not checked)");
            ok = !require_finite && ok;
            break;
        }
        sum2 += (double) x * x;
    }

    printf("%s: n = %zu, rms = %.6f, first:", name, v.size(), v.empty() ? 0.0 : sqrt(sum2 / v.size()));
    for (size_t i = 0; i < 4 && i < v.size(); ++i) {
        printf(" %.6f", v[i]);
    }
    printf("\n");

    return ok;
}

// one branch of the CFG pair, its own context and KV cache on the shared model
struct branch {
    const char    * name = nullptr;
    llama_context * ctx  = nullptr;
    llama_pos       pos  = 0;

    llama_memory_t mem() const {
        return llama_get_memory(ctx);
    }

    // tokens, the hidden state of the last one is kept
    bool decode_tokens(const std::vector<llama_token> & tokens) {
        llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); ++i) {
            batch.token   [i]    = tokens[i];
            batch.pos     [i]    = pos + i;
            batch.n_seq_id[i]    = 1;
            batch.seq_id  [i][0] = 0;
            batch.logits  [i]    = i + 1 == tokens.size();
        }
        batch.n_tokens = tokens.size();

        const int ret = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (ret != 0) {
            fprintf(stderr, "%s: llama_decode of %zu tokens failed: %d\n", name, tokens.size(), ret);
            return false;
        }
        pos += tokens.size();
        return true;
    }

    // one input embedding in place of a token
    bool decode_embd(const float * embd, int64_t n_embd) {
        llama_batch batch = llama_batch_init(1, n_embd, 1);
        std::copy(embd, embd + n_embd, batch.embd);
        batch.pos     [0]    = pos;
        batch.n_seq_id[0]    = 1;
        batch.seq_id  [0][0] = 0;
        batch.logits  [0]    = true;
        batch.n_tokens = 1;

        const int ret = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (ret != 0) {
            fprintf(stderr, "%s: llama_decode of an embedding failed: %d\n", name, ret);
            return false;
        }
        pos += 1;
        return true;
    }

    // last hidden state of the last decoded position (upstream last_hidden_state[:, -1])
    std::vector<float> hidden(int64_t n_embd) const {
        const float * h = llama_get_embeddings_ith(ctx, -1);
        if (h == nullptr) {
            fprintf(stderr, "%s: no hidden state\n", name);
            exit(1);
        }
        return std::vector<float>(h, h + n_embd);
    }

    // drop the whole cache, the next decode starts at position 0
    void reset() {
        llama_memory_seq_rm(mem(), 0, -1, -1);
        pos = 0;
    }

    int64_t n_cached() const {
        const llama_pos p_min = llama_memory_seq_pos_min(mem(), 0);
        const llama_pos p_max = llama_memory_seq_pos_max(mem(), 0);
        return p_max < 0 ? 0 : p_max - p_min + 1;
    }
};

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_frames]\n", argv[0]);
        return 1;
    }
    const int64_t n_frames = argc > 2 ? std::atoll(argv[2]) : 4;
    if (n_frames < 1) {
        fprintf(stderr, "n_frames must be at least 1\n");
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * lmodel = llama_model_load_from_file(argv[1], mparams);
    if (lmodel == nullptr) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }

    const auto * model = dynamic_cast<const llama_model_vibevoice *>(lmodel);
    if (model == nullptr) {
        fprintf(stderr, "%s is not a VibeVoice model\n", argv[1]);
        return 1;
    }

    const auto & vv      = model->hparams.vibevoice;
    const int64_t n_embd = model->hparams.n_embd;
    const int64_t n_lat  = vv.diff_latent_size;

    if (vv.acoustic.vae_dim != vv.diff_latent_size) {
        fprintf(stderr, "latent size %u does not match acoustic vae_dim %u\n", vv.diff_latent_size, vv.acoustic.vae_dim);
        return 1;
    }

    int64_t hop = 1;
    for (uint32_t s = 0; s + 1 < vv.acoustic.n_dec_stages; ++s) {
        hop *= vv.acoustic.dec_ratios[s];
    }

    const llama_token audio_bos = llama_model_vibevoice::audio_bos_token_id;
    if (audio_bos >= llama_vocab_n_tokens(llama_model_get_vocab(lmodel))) {
        fprintf(stderr, "audio_bos_token_id %d is out of the vocab\n", audio_bos);
        return 1;
    }

    // the LM last hidden state is read through the embeddings of the last token
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = 256;
    cparams.n_batch      = 64;
    cparams.n_ubatch     = 64;
    cparams.n_seq_max    = 1;
    cparams.embeddings   = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;

    branch pos = { "pos", llama_init_from_model(lmodel, cparams) };
    branch neg = { "neg", llama_init_from_model(lmodel, cparams) };
    if (pos.ctx == nullptr || neg.ctx == nullptr) {
        fprintf(stderr, "failed to create the contexts\n");
        return 1;
    }
    llama_set_embeddings(pos.ctx, true);
    llama_set_embeddings(neg.ctx, true);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    bool ok = true;

    // the positive branch has the prompt, the negative branch only audio_bos
    ok &= pos.decode_tokens({ 1, 2, 3, 4, 5, audio_bos });
    ok &= neg.decode_tokens({ audio_bos });
    printf("prefill: pos cache %" PRId64 ", neg cache %" PRId64 "\n", pos.n_cached(), neg.n_cached());
    if (!ok || pos.n_cached() != 6 || neg.n_cached() != 1) {
        fprintf(stderr, "unexpected cache sizes after prefill\n");
        ok = false;
    }

    const float    cfg_scale = 1.3f;
    const uint32_t seed      = 1234;

    // restart the negative branch once, as on a new audio_bos from the positive branch
    const int64_t i_reset = n_frames / 2;

    // decoder and semantic encoder caches, each frame only runs the convs on its own latent and samples
    llama_model_vibevoice::tok_stream_cache dec_cache;
    llama_model_vibevoice::tok_stream_cache sem_cache;

    int64_t n_wav = 0;
    bool wav_finite_all = true;

    for (int64_t i = 0; i < n_frames && ok; ++i) {
        const std::vector<float> h_pos = pos.hidden(n_embd);
        const std::vector<float> h_neg = neg.hidden(n_embd);
        ok &= all_finite(h_pos) && all_finite(h_neg);

        const std::vector<float> lat = model->sample_speech_latents(backend, h_pos.data(), h_neg.data(), 1, cfg_scale, seed + i);
        ok &= check("frame latents", lat, n_lat);

        // CFG checks on the first frame: scale 1 is the cond-only sample, another scale moves away from it
        if (i == 0) {
            const double d_hidden = max_abs_diff(h_pos, h_neg);
            const std::vector<float> lat_cond = model->sample_speech_latents(backend, h_pos.data(), nullptr,      1, 1.0f, seed);
            const std::vector<float> lat_cfg1 = model->sample_speech_latents(backend, h_pos.data(), h_neg.data(), 1, 1.0f, seed);
            const double d1 = max_abs_diff(lat_cfg1, lat_cond);
            const double ds = max_abs_diff(lat, lat_cfg1);
            printf("pos vs neg hidden max diff = %g\n", d_hidden);
            printf("cfg=1 vs cond-only max diff = %g, cfg=%.1f vs cfg=1 max diff = %g\n", d1, cfg_scale, ds);
            if (d_hidden < 1e-6 || d1 > 1e-3 || ds < 1e-6) {
                fprintf(stderr, "unexpected CFG behavior\n");
                ok = false;
            }
        }

        std::vector<float> unscaled = lat;
        model->unscale_speech_latents(unscaled.data(), unscaled.size());

        // random-weight fixtures overflow the F16 conversions of the decoder, only the length is checked
        std::vector<float> wav = model->acoustic_dec_stream(backend, dec_cache, unscaled.data(), 1);
        ok &= check("wav", wav, hop, false);

        // with a non-finite waveform the feedback runs on a synthetic one, so the embedding injection still runs
        if (!all_finite(wav)) {
            wav_finite_all = false;
            for (size_t k = 0; k < wav.size(); ++k) {
                wav[k] = 0.05f * cosf(0.11f * (n_wav + k));
            }
        }
        n_wav += wav.size();

        // streaming feedback of this frame, same as the last column of the feedback over all frames so far
        const std::vector<float> emb = model->speech_feedback_stream(backend, sem_cache, lat.data(), wav.data(), 1);
        ok &= check("feedback", emb, n_embd);

        // the same embedding goes into both branches
        ok &= pos.decode_embd(emb.data(), n_embd);
        ok &= neg.decode_embd(emb.data(), n_embd);
        printf("frame %" PRId64 ": pos cache %" PRId64 ", neg cache %" PRId64 "\n", i, pos.n_cached(), neg.n_cached());

        if (i == i_reset) {
            // upstream: a new audio_bos goes into the positive branch, the negative branch restarts from it alone
            neg.reset();
            if (neg.n_cached() != 0) {
                fprintf(stderr, "neg cache not empty after reset\n");
                ok = false;
            }
            ok &= pos.decode_tokens({ audio_bos });
            ok &= neg.decode_tokens({ audio_bos });
            printf("audio_bos reset: pos cache %" PRId64 ", neg cache %" PRId64 "\n", pos.n_cached(), neg.n_cached());
            if (neg.n_cached() != 1 || neg.pos != 1) {
                fprintf(stderr, "neg cache should hold only audio_bos after the reset\n");
                ok = false;
            }
        }
    }

    if (!wav_finite_all) {
        printf("decoded wav was non-finite on this fixture, the feedback used a synthetic waveform\n");
    }

    const int64_t pos_expected = 6 + n_frames + 1;
    const int64_t neg_expected = n_frames - i_reset;
    printf("final: pos cache %" PRId64 " (expected %" PRId64 "), neg cache %" PRId64 " (expected %" PRId64 ")\n",
            pos.n_cached(), pos_expected, neg.n_cached(), neg_expected);
    if (ok && n_wav != n_frames * hop) {
        fprintf(stderr, "streamed %" PRId64 " samples, expected %" PRId64 "\n", n_wav, n_frames * hop);
        ok = false;
    }
    if (ok && (pos.n_cached() != pos_expected || neg.n_cached() != neg_expected)) {
        fprintf(stderr, "unexpected final cache sizes\n");
        ok = false;
    }

    ggml_backend_free(backend);
    llama_free(pos.ctx);
    llama_free(neg.ctx);
    llama_model_free(lmodel);
    llama_backend_free();

    printf("%s\n", ok ? "OK" : "FAIL");

    return ok ? 0 : 1;
}
