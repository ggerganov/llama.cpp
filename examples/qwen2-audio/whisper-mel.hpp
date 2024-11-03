#pragma once
#include "ggml-backend.h"
#include <vector>

struct whisper_mel {
    int n_len_org = 0;

    ggml_context * ctx = nullptr;
    ggml_tensor * tensor = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};

void whisper_mel_init(whisper_mel & mel, ggml_backend_t backend, int n_len, int n_len_org, int n_mel);

void whisper_mel_free(whisper_mel & mel);

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

template <typename T>
struct whisper_span {
    T * data;
    int len;
};

struct whisper_mel_calc {
    virtual ~whisper_mel_calc();
    virtual whisper_mel calculate(whisper_span<const float> samples, int n_threads) = 0;
    static whisper_span<const float> hann_window();
};
