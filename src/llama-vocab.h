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
const struct llama_vocab * llama_get_vocab(const struct llama_model   * model);

// TODO: This should probably be in llama.h
std::vector<llama_vocab::id> llama_tokenize_internal(
        const llama_vocab & vocab,
        std::string raw_text,
        bool add_special,
        bool parse_special = false);

llama_token llama_byte_to_token(const llama_vocab & vocab, uint8_t ch);

const char * llama_token_get_text(const struct llama_vocab & vocab, llama_token token);

float llama_token_get_score(const struct llama_vocab & vocab, llama_token token);

llama_token_attr llama_token_get_attr(const struct llama_vocab & vocab, llama_token token);

bool llama_token_is_eog(const struct llama_vocab & vocab, llama_token token);

bool llama_token_is_control(const struct llama_vocab & vocab, llama_token token);

llama_token llama_token_bos(const struct llama_vocab & vocab);
llama_token llama_token_eos(const struct llama_vocab & vocab);
llama_token llama_token_cls(const struct llama_vocab & vocab);
llama_token llama_token_sep(const struct llama_vocab & vocab);
llama_token llama_token_nl (const struct llama_vocab & vocab);
llama_token llama_token_pad(const struct llama_vocab & vocab);

int32_t llama_add_bos_token(const struct llama_vocab & vocab);
int32_t llama_add_eos_token(const struct llama_vocab & vocab);

llama_token llama_token_prefix(const struct llama_vocab & vocab);
llama_token llama_token_middle(const struct llama_vocab & vocab);
llama_token llama_token_suffix(const struct llama_vocab & vocab);
llama_token llama_token_eot   (const struct llama_vocab & vocab);

int32_t llama_tokenize(
    const struct llama_vocab & vocab,
                  const char * text,
                     int32_t   text_len,
                 llama_token * tokens,
                     int32_t   n_tokens_max,
                        bool   add_special,
                        bool   parse_special);

// does not write null-terminator to buf
int32_t llama_token_to_piece(
        const struct llama_vocab & vocab,
                     llama_token   token,
                            char * buf,
                         int32_t   length,
                         int32_t   lstrip,
                            bool   special);

int32_t llama_detokenize(
        const struct llama_vocab & vocab,
               const llama_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special);
