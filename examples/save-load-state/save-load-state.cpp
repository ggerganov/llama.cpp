#include <vector>
#include <cstdio>
#include <chrono>
#include <fstream>

#include "common.h"
#include "llama.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <vector>
#include <limits>
#include <cstdint>

template <typename T>
void writeValue(std::vector<uint8_t>& output, T value) {
    uint8_t* ptr = reinterpret_cast<uint8_t*>(&value);
    for (size_t i = 0; i < sizeof(T); ++i) {
        output.push_back(ptr[i]);
    }
}

template <typename T>
std::vector<uint8_t> rle_compress(const std::vector<T>& input) {
    std::vector<uint8_t> output;
    size_t inputSize = input.size();

    if (inputSize == 0) {
        return output;
    }

    size_t segment_begin = 0;
    while (segment_begin < inputSize) {
        T current_value = input[segment_begin];
        int counter = 0;
        size_t segment_end = segment_begin + 1;

        if (segment_end == inputSize) {
            counter += (counter >= 0) ? 1 : -1;
        }

        for (; segment_end < inputSize; ++segment_end) {
            T next_value = input[segment_end];
            bool equal_values = next_value == current_value;

            if (counter == 0) {
                counter = equal_values ? 1 : -1;
            }

            if (counter == std::numeric_limits<int>::max() || counter == std::numeric_limits<int>::min()) {
                break;
            }

            if (equal_values && counter > 0) {
                counter++;
            } else if (!equal_values && counter < 0) {
                current_value = next_value;
                counter--;
            } else {
                if (counter < 0) {
                    counter++;
                    segment_end--;
                }
                break;
            }
        }

        // Write counter value
        writeValue<int>(output, counter);

        if (counter > 0) {
            // Write compressed value
            writeValue<T>(output, input[segment_begin]);
            segment_begin = segment_end;
        } else if (counter < 0) {
            for (size_t i = segment_begin; i < segment_end; ++i) {
                // Write uncompressed values
                writeValue<T>(output, input[i]);
            }
            segment_begin = segment_end;
        }
    }

    return output;
}


template <typename T>
T readValue(const std::vector<uint8_t>& input, size_t& index) {
    T value;
    uint8_t* ptr = reinterpret_cast<uint8_t*>(&value);
    for (size_t i = 0; i < sizeof(T); ++i) {
        ptr[i] = input[index++];
    }
    return value;
}

template <typename T>
std::vector<T> rle_decompress(const std::vector<uint8_t>& input) {
    std::vector<T> output;
    size_t inputSize = input.size();
    size_t index = 0;

    while (index < inputSize) {
        // Read counter value
        int counter = readValue<int>(input, index);

        if (counter > 0) {
            // Read compressed value
            T value = readValue<T>(input, index);

            // Decompress repeated value
            for (int i = 0; i < counter; ++i) {
                output.push_back(value);
            }
        } else if (counter < 0) {
            // Read and decompress uncompressed values
            for (int i = 0; i < -counter; ++i) {
                T value = readValue<T>(input, index);
                output.push_back(value);
            }
        }
    }

    return output;
}


int main2() {
    std::vector<int> input = {1, 1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 5, 5};
    std::vector<uint8_t> compressedData = rle_compress(input);
    std::vector<int> decompressedData = rle_decompress<int>(compressedData);

    std::cout << "Compressed data (" << compressedData.size() << " bytes): ";
    for (uint8_t val : compressedData) {
        std::cout << static_cast<int>(val) << " ";
    }
    std::cout << std::endl;

    std::cout << "Decompressed data (" << decompressedData.size() * sizeof(*decompressedData.data()) << " bytes): ";
    for (int val : decompressedData) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}



int main(int argc, char ** argv) {
    gpt_params params;
    params.model = "models/llama-7B/ggml-model.bin";
    params.seed = 42;
    params.n_threads = 4;
    params.repeat_last_n = 64;
    params.prompt = "The quick brown fox";
    // params.n_predict = 10;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    auto lparams = llama_context_default_params();

    lparams.n_ctx     = params.n_ctx;
    lparams.n_parts   = params.n_parts;
    lparams.seed      = params.seed;
    lparams.f16_kv    = params.memory_f16;
    lparams.use_mmap  = params.use_mmap;
    lparams.use_mlock = params.use_mlock;

    // init
    {
        size_t n_past = 0;
        std::vector<llama_token> last_n_tokens(params.repeat_last_n, 0);

        auto ctx = llama_init_from_file(params.model.c_str(), lparams);
        auto tokens = std::vector<llama_token>(params.n_ctx);
        auto n_prompt_tokens = llama_tokenize(ctx, params.prompt.c_str(), tokens.data(), tokens.size(), true);

        if (n_prompt_tokens < 1) {
            fprintf(stderr, "%s : failed to tokenize prompt\n", __func__);
            return 1;
        }

        // evaluate prompt

        llama_eval(ctx, tokens.data(), n_prompt_tokens, n_past, params.n_threads);

        last_n_tokens.insert(last_n_tokens.end(), tokens.data(), tokens.data() + n_prompt_tokens);
        n_past += n_prompt_tokens;

        // Save state (rng, logits, embedding and kv_cache) to file
        {
            // auto file = std::fstream("dump_state.bin", std::ios::out | std::ios::binary);
            // auto state_size = llama_get_state_size(ctx);
            // std::vector<uint8_t> state_mem(state_size);
            // llama_copy_state_data(ctx, state_mem.data()); // could also copy directly to memory mapped file
            // file.write(reinterpret_cast<char*>(&state_size), sizeof(state_size));
            // file.write(reinterpret_cast<char*>(state_mem.data()), state_size);
            //
            // // save state (last tokens)
            // file.write(reinterpret_cast<char*>(&n_past), sizeof(n_past));
            // size_t last_n_tokens_size = last_n_tokens.size();
            // file.write(reinterpret_cast<char*>(&last_n_tokens_size), sizeof(last_n_tokens_size));
            // file.write(reinterpret_cast<char*>(last_n_tokens.data()), last_n_tokens_size * sizeof(llama_token));

            // write everything to a vector, then compress the vector and write to file
            std::vector<uint8_t> raw_data;
            size_t state_size = llama_get_state_size(ctx);
            raw_data.insert(raw_data.end(), reinterpret_cast<uint8_t*>(&state_size), reinterpret_cast<uint8_t*>(&state_size) + sizeof(state_size));
            raw_data.resize(raw_data.size() + state_size);
            llama_copy_state_data(ctx, raw_data.data() + sizeof(state_size));
            raw_data.insert(raw_data.end(), reinterpret_cast<uint8_t*>(&n_past), reinterpret_cast<uint8_t*>(&n_past) + sizeof(n_past));
            size_t last_n_tokens_size = last_n_tokens.size();
            raw_data.insert(raw_data.end(), reinterpret_cast<uint8_t*>(&last_n_tokens_size), reinterpret_cast<uint8_t*>(&last_n_tokens_size) + sizeof(last_n_tokens_size));
            raw_data.insert(raw_data.end(), reinterpret_cast<uint8_t*>(last_n_tokens.data()), reinterpret_cast<uint8_t*>(last_n_tokens.data()) + last_n_tokens_size * sizeof(llama_token));

            std::vector<uint8_t> compressed_data = rle_compress(raw_data);
            std::ofstream file("dump_state.bin.rle", std::ios::out | std::ios::binary);
            file.write(reinterpret_cast<char*>(compressed_data.data()), compressed_data.size());
        }

        // first run
        printf("\n%s", params.prompt.c_str());
        for (auto i = 0; i < params.n_predict; i++) {
            auto logits = llama_get_logits(ctx);
            auto n_vocab = llama_n_vocab(ctx);
            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
            }
            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
            auto next_token = llama_sample_token(ctx, &candidates_p);
            auto next_token_str = llama_token_to_str(ctx, next_token);
            last_n_tokens.push_back(next_token);
            printf("%s", next_token_str);
            if (llama_eval(ctx, &next_token, 1, n_past, params.n_threads)) {
                fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
                return 1;
            }
            n_past += 1;
        }
        printf("\n\n");

        // free old model
        llama_print_timings(ctx);
        llama_free(ctx);
    }

    // load new model
    {
        auto ctx = llama_init_from_file(params.model.c_str(), lparams);

        // Load state (rng, logits, embedding and kv_cache) from file
        size_t n_past = 0;
        std::vector<llama_token> last_n_tokens;
        {
            // auto file = std::fstream("dump_state.bin", std::ios::in | std::ios::binary);
            // size_t state_size;
            // file.read(reinterpret_cast<char*>(&state_size), sizeof(state_size));
            // if (state_size != llama_get_state_size(ctx)) {
            //     fprintf(stderr, "%s : state size mismatch\n", __func__);
            //     return 1;
            // }
            // std::vector<uint8_t> state_mem(state_size);
            // file.read(reinterpret_cast<char*>(state_mem.data()), state_size);
            // llama_set_state_data(ctx, state_mem.data()); // could also copy directly to memory mapped file
            //
            // // restore state (last tokens)
            // file.read(reinterpret_cast<char*>(&n_past), sizeof(n_past));
            // size_t last_n_tokens_size;
            // file.read(reinterpret_cast<char*>(&last_n_tokens_size), sizeof(last_n_tokens_size));
            // last_n_tokens.resize(last_n_tokens_size);
            // file.read(reinterpret_cast<char*>(last_n_tokens.data()), last_n_tokens.size() * sizeof(llama_token));


            // read everything to a vector, then uncompress the vector and write to file
            std::ifstream file("dump_state.bin.rle", std::ios::in | std::ios::binary);
            std::vector<uint8_t> compressed_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            std::vector<uint8_t> raw_data = rle_decompress<uint8_t>(compressed_data);
            size_t state_size;
            memcpy(&state_size, raw_data.data(), sizeof(state_size));
            if (state_size != llama_get_state_size(ctx)) {
                fprintf(stderr, "%s : state size mismatch\n", __func__);
                return 1;
            }
            llama_set_state_data(ctx, raw_data.data() + sizeof(state_size));
            memcpy(&n_past, raw_data.data() + sizeof(state_size) + llama_get_state_size(ctx), sizeof(n_past));
            size_t last_n_tokens_size;
            memcpy(&last_n_tokens_size, raw_data.data() + sizeof(state_size) + llama_get_state_size(ctx) + sizeof(n_past), sizeof(last_n_tokens_size));
            last_n_tokens.resize(last_n_tokens_size);
            memcpy(last_n_tokens.data(), raw_data.data() + sizeof(state_size) + llama_get_state_size(ctx) + sizeof(n_past) + sizeof(last_n_tokens_size), last_n_tokens_size * sizeof(llama_token));
        }

        // second run
        for (auto i = 0; i < params.n_predict; i++) {
            auto logits = llama_get_logits(ctx);
            auto n_vocab = llama_n_vocab(ctx);
            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
            }
            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
            auto next_token = llama_sample_token(ctx, &candidates_p);
            auto next_token_str = llama_token_to_str(ctx, next_token);
            last_n_tokens.push_back(next_token);
            printf("%s", next_token_str);
            if (llama_eval(ctx, &next_token, 1, n_past, params.n_threads)) {
                fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
                return 1;
            }
            n_past += 1;
        }
        printf("\n\n");

        // free model (for sanity check)
        llama_print_timings(ctx);
        llama_free(ctx);
    }
    return 0;
}
