#pragma once

#include "llama-impl.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <map>

struct llama_vocab {
    using id    = llama_token;
    using token = std::string;
    using tattr = llama_token_attr;

    struct token_data {
        token text;
        float score;
        tattr attr;
    };

    enum llama_vocab_type     type     = LLAMA_VOCAB_TYPE_SPM;
    enum llama_vocab_pre_type type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;

    int max_token_len = 0; // used for optimizing longest token search

    std::unordered_map<token, id> token_to_id;
    std::vector<token_data>       id_to_token;

    std::vector<id>    cache_special_tokens;
    std::vector<token> cache_token_to_piece; // llama_token_to_piece(special = true);

    std::map<std::pair<std::string, std::string>, int> bpe_ranks;

    // default LLaMA special tokens
    id special_bos_id  = 1;
    id special_eos_id  = 2;
    id special_unk_id  = 0;
    id special_sep_id  = -1;
    id special_pad_id  = -1;
    id special_cls_id  = -1;
    id special_mask_id = -1;

    id linefeed_id       = 13;
    id special_prefix_id = -1;
    id special_suffix_id = -1;
    id special_middle_id = -1;
    id special_eot_id    = -1; // TODO: move above after "eos_id", and here add "file separator" token

    // tokenizer flags
    bool tokenizer_add_space_prefix = false;
    bool tokenizer_add_bos          = false;
    bool tokenizer_add_eos          = false;
    bool tokenizer_ignore_merges    = false;
    bool tokenizer_clean_spaces     = false;  // clean_up_tokenization_spaces
    bool tokenizer_remove_extra_whitespaces   = false;
    bool tokenizer_escape_whitespaces         = true;
    bool tokenizer_treat_whitespace_as_suffix = false;

    std::vector<char> precompiled_charsmap;

    int find_bpe_rank(const std::string & token_left, const std::string & token_right) const;
};

const struct llama_vocab * llama_get_vocab(const struct llama_context * ctx);
