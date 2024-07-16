#include "llama-vocab.h"

int llama_vocab::find_bpe_rank(const std::string & token_left, const std::string & token_right) const {
    GGML_ASSERT(token_left.find(' ') == std::string::npos);
    GGML_ASSERT(token_left.find('\n') == std::string::npos);
    GGML_ASSERT(token_right.find(' ') == std::string::npos);
    GGML_ASSERT(token_right.find('\n') == std::string::npos);

    auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
    if (it == bpe_ranks.end()) {
        return -1;
    }

    return it->second;
}
