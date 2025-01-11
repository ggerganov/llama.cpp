#pragma once

#include <vector>

struct wav_header {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunk_size;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_format = 1; // PCM
    uint16_t num_channels = 1; // Mono
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample = 16;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size;
};

std::vector<float> tts_embd_to_audio(
        const float * embd,
        const int n_codes,
        const int n_embd,
        const int n_thread);

llama_tokens tts_preprocess_prompt(const llama_model * model_ttc, const std::string & prompt_str);

int tts_get_embd(struct llama_context * ctx_cts, llama_tokens & codes, std::vector<float> & output);
