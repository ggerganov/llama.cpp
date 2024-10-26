#pragma once

#include "jarvis-impl.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>

struct llm_tokenizer;

struct jarvis_vocab {
    using id    = jarvis_token;
    using token = std::string;
    using tattr = jarvis_token_attr;

    struct token_data {
        token text;
        float score;
        tattr attr;
    };

    uint32_t n_vocab = 0; // TODO: not great because has to keep in sync with hparams.n_vocab

    enum jarvis_vocab_type     type     = JARVIS_VOCAB_TYPE_SPM;
    enum jarvis_vocab_pre_type type_pre = JARVIS_VOCAB_PRE_TYPE_DEFAULT;

    int max_token_len = 0; // used for optimizing longest token search

    std::unordered_map<token, id> token_to_id;
    std::vector<token_data>       id_to_token;

    std::vector<id>    cache_special_tokens;
    std::vector<token> cache_token_to_piece; // jarvis_token_to_piece(special = true);

    std::map<std::pair<std::string, std::string>, int> bpe_ranks;

    // default LLaMA special tokens
    // TODO: should we set all of these to JARVIS_TOKEN_NULL?
    id special_bos_id  = 1;
    id special_eos_id  = 2;
    id special_eot_id  = JARVIS_TOKEN_NULL;
    id special_eom_id  = JARVIS_TOKEN_NULL;
    id special_unk_id  = 0;
    id special_sep_id  = JARVIS_TOKEN_NULL;
    id special_pad_id  = JARVIS_TOKEN_NULL;
    id special_cls_id  = JARVIS_TOKEN_NULL;
    id special_mask_id = JARVIS_TOKEN_NULL;

    id linefeed_id = 13;

    // fim tokens
    id special_fim_pre_id = JARVIS_TOKEN_NULL;
    id special_fim_suf_id = JARVIS_TOKEN_NULL;
    id special_fim_mid_id = JARVIS_TOKEN_NULL;
    id special_fim_pad_id = JARVIS_TOKEN_NULL;
    id special_fim_rep_id = JARVIS_TOKEN_NULL; // repo
    id special_fim_sep_id = JARVIS_TOKEN_NULL; // file separator

    // set of all tokens that cause "end of generation"
    std::set<id> special_eog_ids;

    // tokenizer flags
    bool tokenizer_add_space_prefix           = false;
    bool tokenizer_add_bos                    = false;
    bool tokenizer_add_eos                    = false;
    bool tokenizer_ignore_merges              = false;
    bool tokenizer_clean_spaces               = false;  // clean_up_tokenization_spaces
    bool tokenizer_remove_extra_whitespaces   = false;
    bool tokenizer_escape_whitespaces         = true;
    bool tokenizer_treat_whitespace_as_suffix = false;

    std::vector<char> precompiled_charsmap;

    llm_tokenizer * tokenizer = nullptr;

    jarvis_vocab() = default;
    ~jarvis_vocab();

    int find_bpe_rank(const std::string & token_left, const std::string & token_right) const;

    void init_tokenizer();
};

//
// internal API
//

// TODO: rename to jarvis_tokenize_impl
// TODO: This should probably be in jarvis.h
std::vector<jarvis_vocab::id> jarvis_tokenize_internal(
        const jarvis_vocab & vocab,
        std::string raw_text,
        bool add_special,
        bool parse_special = false);

// TODO: move the API below as member functions of jarvis_vocab
jarvis_token jarvis_byte_to_token_impl(const jarvis_vocab & vocab, uint8_t ch);

const char * jarvis_token_get_text_impl(const struct jarvis_vocab & vocab, jarvis_token token);

float jarvis_token_get_score_impl(const struct jarvis_vocab & vocab, jarvis_token token);

jarvis_token_attr jarvis_token_get_attr_impl(const struct jarvis_vocab & vocab, jarvis_token token);

bool jarvis_token_is_eog_impl(const struct jarvis_vocab & vocab, jarvis_token token);

bool jarvis_token_is_control_impl(const struct jarvis_vocab & vocab, jarvis_token token);

jarvis_token jarvis_token_bos_impl(const struct jarvis_vocab & vocab);
jarvis_token jarvis_token_eos_impl(const struct jarvis_vocab & vocab);
jarvis_token jarvis_token_eot_impl(const struct jarvis_vocab & vocab);
jarvis_token jarvis_token_eom_impl(const struct jarvis_vocab & vocab);
jarvis_token jarvis_token_cls_impl(const struct jarvis_vocab & vocab);
jarvis_token jarvis_token_sep_impl(const struct jarvis_vocab & vocab);
jarvis_token jarvis_token_nl_impl (const struct jarvis_vocab & vocab);
jarvis_token jarvis_token_pad_impl(const struct jarvis_vocab & vocab);

jarvis_token jarvis_token_prefix_impl(const struct jarvis_vocab & vocab);
jarvis_token jarvis_token_middle_impl(const struct jarvis_vocab & vocab);
jarvis_token jarvis_token_suffix_impl(const struct jarvis_vocab & vocab);

jarvis_token jarvis_token_fim_pre_impl(const struct jarvis_vocab & vocab);
jarvis_token jarvis_token_fim_suf_impl(const struct jarvis_vocab & vocab);
jarvis_token jarvis_token_fim_mid_impl(const struct jarvis_vocab & vocab);
jarvis_token jarvis_token_fim_pad_impl(const struct jarvis_vocab & vocab);
jarvis_token jarvis_token_fim_rep_impl(const struct jarvis_vocab & vocab);
jarvis_token jarvis_token_fim_sep_impl(const struct jarvis_vocab & vocab);

bool jarvis_add_bos_token_impl(const struct jarvis_vocab & vocab);
bool jarvis_add_eos_token_impl(const struct jarvis_vocab & vocab);

int32_t jarvis_tokenize_impl(
        const struct jarvis_vocab & vocab,
                      const char * text,
                         int32_t   text_len,
                     jarvis_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);

// does not write null-terminator to buf
int32_t jarvis_token_to_piece_impl(
        const struct jarvis_vocab & vocab,
                     jarvis_token   token,
                            char * buf,
                         int32_t   length,
                         int32_t   lstrip,
                            bool   special);

// check if token0 is contained as a prefix in token1
bool jarvis_token_is_prefix_impl(
        const struct jarvis_vocab & vocab,
                     jarvis_token   token0,
                     jarvis_token   token1);

int32_t jarvis_detokenize_impl(
        const struct jarvis_vocab & vocab,
               const jarvis_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special);

std::string jarvis_detokenize(
        const struct jarvis_vocab & vocab,
  const std::vector<jarvis_token> & tokens,
                            bool   special);
