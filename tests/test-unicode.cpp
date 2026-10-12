#include "../src/unicode.h"
#include "llama.h"
#include "gguf.h"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

static bool test_nfc_tokenizer() {
    gguf_context * gguf = gguf_init_empty();
    gguf_set_val_str(gguf, "general.architecture", "bert");
    gguf_set_val_u32(gguf, "bert.context_length", 32);
    gguf_set_val_u32(gguf, "bert.embedding_length", 8);
    gguf_set_val_u32(gguf, "bert.block_count", 1);
    gguf_set_val_u32(gguf, "bert.feed_forward_length", 16);
    gguf_set_val_u32(gguf, "bert.attention.head_count", 2);
    gguf_set_val_f32(gguf, "bert.attention.layer_norm_epsilon", 1e-5f);
    gguf_set_val_str(gguf, "tokenizer.ggml.model", "t5");
    const char * tokens[] = {"<pad>", "</s>", "<unk>", "e", "\xCC\x81", "\xC3\xA9", "\xEF\xBF\xBD"};
    const float scores[] = {0, 0, 0, -1, -1, -1, -1};
    const int32_t types[] = {3, 3, 2, 1, 1, 1, 1};
    gguf_set_arr_str(gguf, "tokenizer.ggml.tokens", tokens, 7);
    gguf_set_arr_data(gguf, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores, 7);
    gguf_set_arr_data(gguf, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types, 7);
    gguf_set_val_bool(gguf, "tokenizer.ggml.add_space_prefix", false);
    gguf_set_val_bool(gguf, "tokenizer.ggml.add_bos_token", false);
    gguf_set_val_bool(gguf, "tokenizer.ggml.add_eos_token", false);
    gguf_set_val_bool(gguf, "tokenizer.ggml.normalizer.nfc", true);
    std::vector<uint8_t> metadata(gguf_get_meta_size(gguf));
    gguf_get_meta_data(gguf, metadata.data());
    gguf_free(gguf);
    FILE * file = std::tmpfile();
    if (!file) {
        return false;
    }
    if (std::fwrite(metadata.data(), 1, metadata.size(), file) != metadata.size()) {
        std::fclose(file);
        return false;
    }
    std::rewind(file);
    auto params = llama_model_default_params();
    params.vocab_only = true;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_model * model = llama_model_load_from_file_ptr(file, params);
    std::fclose(file);
    if (!model) {
        return false;
    }
    const auto * vocab = llama_model_get_vocab(model);
    const std::vector<std::pair<std::string, std::vector<llama_token>>> cases = {
        {"e\xCC\x81", {5}},
        {"\xC3\xA9", {5}},
        {"\xFF" "e\xCC\x81", {6, 5}},
        {"\x80", {6}},
        {"\xE2\x82", {6, 6}},
        {"\xE2" "e", {6, 3}},
        {"\xF4\x90\x80\x80", {2}},
    };
    bool success = true;
    for (const auto & test : cases) {
        std::vector<llama_token> actual(16);
        const int32_t n = llama_tokenize(vocab, test.first.data(), test.first.size(), actual.data(), actual.size(), false, false);
        if (n < 0) {
            success = false;
            break;
        }
        actual.resize(n);
        if (actual != test.second) {
            success = false;
            break;
        }
    }
    llama_model_free(model);
    return success;
}

int main() {
    if (!test_nfc_tokenizer()) {
        fprintf(stderr, "unexpected NFC tokenization\n");
        return 1;
    }

    const std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> normalization = {
        {{}, {}},
        {{0x65, 0x301}, {0xE9}},
        {{0xE9}, {0xE9}},
        {{0x212B}, {0xC5}},
        {{0x61, 0x315, 0x300}, {0xE0, 0x315}},
        {{0x61, 0x305, 0x301}, {0x61, 0x305, 0x301}},
        {{0x301, 0x327}, {0x327, 0x301}},
        {{0x1100, 0x1161, 0x11A8}, {0xAC01}},
        {{0xAC01}, {0xAC01}},
        {{0xFB01, 0xFF21}, {0xFB01, 0xFF21}},
        {{0x958}, {0x915, 0x93C}},
        {{0x1D15E}, {0x1D157, 0x1D165}},
    };
    for (const auto & test : normalization) {
        if (unicode_cpts_normalize_nfc(test.first) != test.second || unicode_cpts_normalize_nfc(test.second) != test.second) {
            fprintf(stderr, "unexpected NFC normalization\n");
            return 1;
        }
    }

    const std::vector<std::string> regex_exprs = {
        "[~][A-Za-z]+| ?[\\p{S}]+|\\s+",
    };
    const std::vector<std::string> expected = { " ~", "foo" };
    const auto actual = unicode_regex_split(" ~foo", regex_exprs, false);

    if (actual != expected) {
        fprintf(stderr, "unexpected split:");
        for (const auto & piece : actual) {
            fprintf(stderr, " [%s]", piece.c_str());
        }
        fprintf(stderr, "\n");
        return 1;
    }

    return 0;
}
