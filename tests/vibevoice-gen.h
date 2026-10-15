// host generation loop shared by the VibeVoice smokes and CLI, on top of llama_decode
// two contexts share the model: the positive branch has the prompt, the negative (CFG) branch starts at audio_bos

#pragma once

#include "ggml-backend.h"
#include "llama.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "models/models.h"

#include <cstdint>
#include <vector>

// one branch of the CFG pair, its own context and KV cache on the shared model
struct vv_branch {
    const char    * name = nullptr;
    llama_context * ctx  = nullptr;
    llama_pos       pos  = 0;

    llama_memory_t mem() const;

    // tokens, the logits and hidden state of the last one are kept
    bool decode_tokens(const std::vector<llama_token> & tokens);

    // one input embedding in place of a token
    bool decode_embd(const float * embd, int64_t n_embd);

    // last hidden state of the last decoded position (upstream last_hidden_state[:, -1]), empty if missing
    std::vector<float> hidden(int64_t n_embd) const;

    // drop the whole cache, the next decode starts at position 0
    void reset();

    int64_t n_cached() const;
};

// branch policy of step(), per sampled token (upstream argmax loop):
// - audio_bos: decoded into pos, neg is reset and gets a fresh audio_bos
// - audio_token: one CFG diffusion frame from the pos/neg hidden states, the speech feedback embd goes into both
// - other text tokens: decoded into pos only, neg only sees audio_bos and speech embds
// - eog: nothing is decoded, the caller stops
// the tokenizer stream caches are not reset on audio_bos, all frames are one continuous waveform
struct vv_generator {
    const llama_model_vibevoice * model = nullptr;
    const llama_vocab           * vocab = nullptr;

    ggml_backend_t backend = nullptr;

    vv_branch pos = { "pos" };
    vv_branch neg = { "neg" };

    int64_t n_embd = 0;
    int64_t hop    = 1; // waveform samples per frame

    float    cfg_scale = 1.3f;
    uint32_t seed      = 1234; // frame i draws its noise with seed + i

    llama_model_vibevoice::tok_stream_cache dec_cache;
    llama_model_vibevoice::tok_stream_cache sem_cache;

    // all decoded samples, non-finite ones are replaced by 0 (also in the feedback input)
    std::vector<float> wav;
    int64_t n_frames    = 0;
    int64_t n_nonfinite = 0;

    // outputs of the last frame
    std::vector<float> lat;
    std::vector<float> emb;

    vv_generator() = default;
    vv_generator(const vv_generator &) = delete;
    vv_generator & operator=(const vv_generator &) = delete;
    ~vv_generator();

    // the contexts have embeddings on (pooling none) for the hidden states, logits stay available
    bool init(llama_model * lmodel, uint32_t n_ctx);

    // pos gets the prompt, neg a single audio_bos
    bool prefill(const std::vector<llama_token> & prompt);

    bool audio_bos();
    bool audio_frame();

    // samples the next token from the pos logits and runs its branch, returns LLAMA_TOKEN_NULL on error
    llama_token step(llama_sampler * smpl);

    bool is_eog(llama_token tok) const;
};
