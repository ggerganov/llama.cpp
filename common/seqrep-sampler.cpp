#include "llama.h"

#include "ggml.h"
#include "common.h"

#include "seqrep-sampler.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <ctime>
#include <initializer_list>
#include <map>
#include <vector>
#include <queue>
#include <random>
#include <unordered_map>
#include <fstream>
#include <sstream>

#include <codecvt>
#include <locale>
#include <cwctype>


#define SR_FLAG(flags, flag_val) (((flags) & (flag_val)) != 0)

static std::wstring utf8_to_wstring(const char * start, const char * end) {
    if (end == NULL) {
        const size_t len = strlen(start);
        end = len > 0 ? start + (len - 1) : start;
    }
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> temp;
    return temp.from_bytes(start, end);
}

static std::string wstring_to_string(const std::wstring & s) {

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> temp;
    return temp.to_bytes(s);
}

void seqrep_sampler_params_init(llama_sampler_seqrep_params * params) {
    assert(params != NULL);
    *params = {};
    params->max_length = 24;
    params->last_n = 256;
    params->mid_word_scale = 0.1f;
    params->rewind_max_visits = 2;
    params->rewind_ban_length = 1;
}

static void seqrep_params_dump_flags(const int flags) {
    const char *flag_names[] = {
         "tolerance_no_consecutive",
         "tolerance_no_first",
         "penalize_length_max_seen",
         "absolute_penalty",
         "rewind_mode",
         "rewind_skip_ws_punct",
         "rewind_use_shortest_match",
         "rewind_require_wbound",
         "rewind_persist_require_wbound"
    };
    for (int i = 0, fcount = 0; i <= 8; i++) {
        if ((flags & (1 << i)) != 0) {
            printf("%s%s", fcount > 0 ? " + " : "", flag_names[i]);
            fcount++;
        }
    }
}

void seqrep_sampler_params_dump(const llama_sampler_seqrep_params * params) {
    assert(params != NULL);
    printf("seqrep(last_n = %d, min_length = %zd, max_length = %zd, start_offset = %zd, presence_penalty = %.4f, length_penalty = %.4f, tolerance = %.4f, mid_word_scale = %.4f, tolerance_match_credit = %.4f, tolerance_cap = %.4f, rewind_min_length = %zd, rewind_seek_word_boundary = %zd, flags = [",
        params->last_n, params->min_length, params->max_length, params->start_offset, params->presence_penalty,
        params->length_penalty, params->tolerance, params->mid_word_scale,
        params->tolerance_match_credit, params->tolerance_cap,
        params->rewind_min_length, params->rewind_seek_word_boundary);
    seqrep_params_dump_flags(params->flags);
    puts("])");
}

// FIXME: Error handling.
static bool seqrep_load_file(const std::string & filename, std::vector<std::wstring> & result) {
    std::ifstream fp(filename);
    if (!fp) {
        return false;
    }
    std::string buf;

    while (std::getline(fp, buf)) {
        while (!buf.empty() && (buf.back() == L'\r' || buf.back() == L'\n')) {
            buf.resize(buf.size() - 1);
        }
        if (!buf.empty()) {
            std::wstring temp = utf8_to_wstring(buf.data(), buf.data() + buf.size());
            result.push_back(std::move(temp));
        }
    }
    return true;
}

// FIXME: Error handling. More efficient loading?
static bool seqrep_load_regex_file(const std::string & filename, std::vector<std::wregex> & result) {
    std::vector<std::wstring> buf;

    if (!seqrep_load_file(filename, buf)) {
        return false;
    }
    result.clear();
    for (const std::wstring & line : buf) {
        if (line.empty() || line.front() == L'#') {
            continue;
        }
        result.emplace_back(line);
    }
    return true;
}

void seqrep_sampler_help() {
    llama_sampler_seqrep_params p;

    seqrep_sampler_params_init(&p);
    printf("==== Sequence Repetition Sampler Help ====\n\n");
    printf("  The sequence repetition sampler takes a configuration string in the format:\n");
    printf("  arg1:arg2:argN\n");
    printf("  A colon separated argument can be a key value pair like xyz=1 or flag like xyz\n");
    printf("\n- Available key/value arguments\n");
    printf("  * repetition_mode=REPEAT_PENALTY\n    emulates the repetition penalty sampler. warning: 1.0 disables penalties since this preset enables flag_divide_by_penalty. using 0.0 is probably not what you want\n");
    printf("  * presence_mode=PRESENCE_PENALTY\n    emulates the presence penalty sampler\n");
    printf("  * frequency_mode=FREQUENCY_PENALTY\n    Emulates the repetition penalty sampler\n");
    printf("  * rewind_mode\n    Enables rewind mode and sets skip_ws_punct, require_wbound and persist_require_wbound flags\n");
    printf("  * last_n\n    last n tokens to consider for sequence penalizing (default: %d, 0 = disabled, -1 = ctx_size)\n", p.last_n);
    printf("  * min_length\n    minimum matching sequence length (default: %zd, < 2 = disabled)\n", p.min_length);
    printf("  * presence_penalty\n    presence penalty for tokens that can continue a sequence (default: %f)\n", p.presence_penalty);
    printf("  * length_penalty\n    penalty for tokens that can continue a sequence, scaled by length (default: %f)\n", p.length_penalty);
    printf("  * tolerance\n    tolerance for fuzzy matching sequences (default: %f, 0 = disabled)\n", p.tolerance);
    printf("  * mid_word_scale\n    scale penalty when for mid-word tokens. 1.0 would mean apply the full penalty (default: %f, 1.0 = disabled)\n", p.mid_word_scale);
    printf("  * tolerance_match_credit\n    credit tolerance on matched tokens (default: %f, 0.0 = disabled)\n", p.tolerance_match_credit);
    printf("  * tolerance_cap\n    Caps tolerance at the specified value. Only meaningful when tolerance_match_credit > 0 (default: %f)\n", p.tolerance_cap);
    printf("  * start_offset\n    advanced option to set the initial offset for pattern matching. This is relative to the start of last_n. For example, you can set last_n=-1:start_offset=NUM_PROMPT_TOKENS to limit sequence matching to the prompt (default: %zu)\n", p.start_offset);
    printf("  * rewind_min_length\n    Ensure the sequence is at least the specified length in rewind mode after whitespace skipping and other modifications (default: %zu)\n", p.rewind_min_length);
    printf("  * rewind_max_visits\n    A position is limited to the specified number of rewinds. When the limit is exceeded, future rewinds cannot target it or earlier tokens. (default: %zu)\n", p.rewind_max_visits);
    printf("  * rewind_persist_bans\n    Tokens banned by rewind remain banned for an additional number of positions equal to the value. i.e. setting this to 1 would mean the token is banned for 2 positions. (default: %zu)\n", p.rewind_persist_bans);
    printf("  * rewind_ban_length\n    Number of tokens from the sequence to ban when rewinding. (default: %zu)\n", p.rewind_ban_length);
    printf("  * include_re_file=FILENAME\n    loads a list of regexps from the file, seqrep matching will only occur if the regex matches\n");
    printf("  * exclude_re_file=FILENAME\n    loads a list of regexps from the file, seqrep matching will only occur if the regex matches\n");
    printf("\n- Available flags arguments (currently all default to disabled)\n");
    printf("  * flag_tolerance_no_consecutive\n    do not allow using tolerance consecutively\n");
    printf("  * flag_tolerance_no_first\n    do not allow using tolerance before the first match\n");
    printf("  * flag_penalize_length_max_seen\n    when applying length_penalty, use the maximum seen sequence length rather than the total length of seen sequences\n");
    printf("  * flag_absolute_penalty\n    Apply an absolute penalty rather than dividing the logit by the penalty.\n");
    printf("  * flag_rewind_mode\n    Rather than penalizing tokens that can continue a sequence, this mode will actually rewind and ban the token that _starts_ the sequence. Note: Requires support in the caller. Also only applies when min_length is at least 2. Most other settings will be ignored in this mode\n");
    printf("  * flag_rewind_skip_ws_punct\n    When rewinding, skip past whitespace and punctuation. For example, if the matched sequence was \"<NL>'hello\" then we will rewind to the token starting with 'h' and ban it.\n");
    printf("  * flag_rewind_use_shortest_match\n    Rewind to the shortest matching sequence of at least min_length rather than the longest. Only meaningful when multiple rewind seqrep samplers are defined.\n");
    printf("  * flag_rewind_require_wbound\n    Rewinding requires a word boundary. Only has an effect when rewind_seek_word_boundary isn't 0.\n");
    printf("  * flag_rewind_persist_require_wbound\n    Persisted bans are only applied if at a word bound.\n");
    printf("\n- Regex file notes:\n");
    printf("  The regex file should contain one regex per line. Blank lines or lines that start with # are ignored.\n");
    printf("  When matching, the last max_length tokens are converted to a string and invalid unicode is trimmed from the beginning/end.\n");
    printf("  Note: Current regexes only apply for rewind mode seqrep samplers.\n");
    printf("\n- Examples:\n");
    printf("  * repetition_mode=1.2:last_n=32\n    same as --repeat-last-n 32 --repeat-penalty 1.2\n");
    printf("  * presence_mode=.2:last_n=32\n    same as --repeat-last-n 32 --presence-penalty .2\n");
    printf("  * frequency_mode=.2:last_n=32\n    same as --repeat-last-n 32 --frequency-penalty .2\n");
    printf("  * min_length=3:tolerance=1:length_penalty=1.1:last_n=-1\n    match repeated sequences of at least 3 tokens within the entire context and apply a penalty of 1 + 0.1*total_length to the token that would continue the sequence. allow one non-matching token in matched sequences.\n");
}

bool seqrep_sampler_params_parse(char * s, llama_sampler_seqrep_params * params) {
    assert(params != NULL);
    assert(s != NULL);
    size_t offset = 0;
    std::string sparams = s;
    size_t slen = sparams.size();

    while (offset < slen) {
        size_t argsep = sparams.find_first_of(':', offset);
        std::string argchunk;
        if (argsep == std::string::npos) {
            argchunk = sparams.substr(offset);
        } else if (argsep > offset) {
            argchunk = sparams.substr(offset, argsep - offset);
        }
        std::string argval;
        size_t valsep = argchunk.find_first_of('=');
        if (valsep != std::string::npos && valsep < argchunk.size()) {
            argval = argchunk.substr(valsep + 1);
            argchunk.resize(valsep);
        }
        if (argchunk.empty() && argval.empty()) {
            // pass
        } else if (argchunk == "repetition_mode") {
            params->last_n = 64;
            params->min_length = 1;
            params->mid_word_scale = 1.0f;
            params->flags = 0;
            params->length_penalty = 1.0f;
            params->presence_penalty = argval.empty() ? 1.1f : std::atof(argval.c_str());
        } else if (argchunk == "presence_mode") {
            params->last_n = 64;
            params->min_length = 1;
            params->mid_word_scale = 1.0f;
            params->flags = LLAMA_SEQREP_ABSOLUTE_PENALTY;
            params->length_penalty = 0.0f;
            params->presence_penalty = std::atof(argval.c_str());
        } else if (argchunk == "frequency_mode") {
            params->last_n = 64;
            params->min_length = 1;
            params->mid_word_scale = 1.0f;
            params->flags = LLAMA_SEQREP_ABSOLUTE_PENALTY;
            params->length_penalty = std::atof(argval.c_str());
            params->presence_penalty = 0.0f;
        } else if (argchunk == "rewind_mode") {
            params->flags = LLAMA_SEQREP_REWIND_REQUIRE_WBOUND
                | LLAMA_SEQREP_REWIND_PERSIST_REQUIRE_WBOUND
                | LLAMA_SEQREP_REWIND_SKIP_WS_PUNCT
                | LLAMA_SEQREP_REWIND_MODE;
        } else if (argchunk == "flag_tolerance_no_consecutive") {
            params->flags |= LLAMA_SEQREP_TOLERANCE_NO_CONSECUTIVE;
        } else if (argchunk == "flag_tolerance_no_first") {
            params->flags |= LLAMA_SEQREP_TOLERANCE_NO_FIRST;
        } else if (argchunk == "flag_penalize_length_max_seen") {
            params->flags |= LLAMA_SEQREP_PENALIZE_LENGTH_MAX_SEEN;
        } else if (argchunk == "flag_absolute_penalty") {
            params->flags |= LLAMA_SEQREP_ABSOLUTE_PENALTY;
        } else if (argchunk == "flag_rewind_mode") {
            params->flags |= LLAMA_SEQREP_REWIND_MODE;
        } else if (argchunk == "flag_rewind_skip_ws_punct") {
            params->flags |= LLAMA_SEQREP_REWIND_SKIP_WS_PUNCT | LLAMA_SEQREP_REWIND_MODE;
        } else if (argchunk == "flag_rewind_use_shortest_match") {
            params->flags |= LLAMA_SEQREP_REWIND_USE_SHORTEST_MATCH | LLAMA_SEQREP_REWIND_MODE;
        } else if (argchunk == "flag_rewind_require_wbound") {
            params->flags |= LLAMA_SEQREP_REWIND_REQUIRE_WBOUND | LLAMA_SEQREP_REWIND_MODE;
        } else if (argchunk == "flag_rewind_persist_require_wbound") {
            params->flags |= LLAMA_SEQREP_REWIND_PERSIST_REQUIRE_WBOUND | LLAMA_SEQREP_REWIND_MODE;
        } else if (argchunk == "min_length") {
            params->min_length = std::atoi(argval.c_str());
        } else if (argchunk == "rewind_min_length") {
            params->rewind_min_length = std::atoi(argval.c_str());
        } else if (argchunk == "rewind_seek_word_boundary") {
            params->rewind_seek_word_boundary = std::atoi(argval.c_str());
        } else if (argchunk == "rewind_max_visits") {
            params->rewind_max_visits = std::atoi(argval.c_str());
        } else if (argchunk == "rewind_persist_bans") {
            params->rewind_persist_bans = std::atoi(argval.c_str());
        } else if (argchunk == "rewind_ban_length") {
            params->rewind_ban_length = std::atoi(argval.c_str());
        } else if (argchunk == "start_offset") {
            params->start_offset = std::atoi(argval.c_str());
        } else if (argchunk == "last_n") {
            params->last_n = std::atoi(argval.c_str());
        } else if (argchunk == "tolerance") {
            params->tolerance = std::atof(argval.c_str());
        } else if (argchunk == "tolerance_cap") {
            params->tolerance_cap = std::atof(argval.c_str());
        } else if (argchunk == "presence_penalty") {
            params->presence_penalty = std::atof(argval.c_str());
        } else if (argchunk == "length_penalty") {
            params->length_penalty = std::atof(argval.c_str());
        } else if (argchunk == "mid_word_scale") {
            params->mid_word_scale = std::atof(argval.c_str());
        } else if (argchunk == "tolerance_match_credit") {
            params->tolerance_match_credit = std::atof(argval.c_str());
        } else if (argchunk == "include_re_file" && !argval.empty()) {
            if (!seqrep_load_regex_file(argval, params->include_re)) {
                fprintf(stderr, "seqrep: Failed to read include_re file: %s\n", argval.c_str());
                return false;
            };
        } else if (argchunk == "exclude_re_file" && !argval.empty()) {
            if (!seqrep_load_regex_file(argval, params->exclude_re)) {
                   fprintf(stderr, "seqrep: Failed to read exclude_re file: %s\n", argval.c_str());
                return false;
            }
        } else {
            fprintf(stderr, "seqrep: Bad argument [%s]=[%s]!\n", argchunk.c_str(), argval.c_str());
            return false;
        }
        if (argsep != std::string::npos) {
            offset = argsep + 1;
        } else {
            break;
        }
    }
    if (params->tolerance_cap == 0.0f) {
        params->tolerance_cap = params->tolerance;
    }
    return true;
}


// Internal helper function for sequence matching.
static size_t seqrep_find_match(
        const llama_token * tail_tokens,
        const size_t tail_tokens_size,
        const llama_token * search_tokens,
        const size_t search_tokens_size,
        const bool overlaps,
        const llama_sampler_seqrep_params *params) {

    if (params->min_length < 2
            || tail_tokens_size < params->min_length
            || search_tokens_size < params->min_length) {
        return 0;
    }

    int flags = params->flags;
    float tolerance = params->tolerance;
    size_t tail_steps = 0, search_steps = 0;

    int matches = 0, pending_matches = 0;
    bool last_matched = true;

    while (search_steps < search_tokens_size && tail_steps < tail_tokens_size) {
        if (*(search_tokens - search_steps) == *(tail_tokens - tail_steps)) {
            tail_steps++;
            search_steps++;
            matches += 1 + pending_matches;
            pending_matches = 0;
            tolerance += params->tolerance_match_credit;
            if (params->tolerance_cap > 0.0f) {
                tolerance = std::min(params->tolerance_cap, tolerance);
            }
            last_matched = true;
            continue;
        }


        if (SR_FLAG(flags, LLAMA_SEQREP_TOLERANCE_NO_FIRST)
               && search_steps + tail_steps == 0) {
            break;
        } else if (SR_FLAG(flags, LLAMA_SEQREP_TOLERANCE_NO_CONSECUTIVE)
                && last_matched == false) {
            break;
        }

        last_matched = false;

        if (tolerance < 1.0f) {
            break;
        }
        tolerance -= 1.0f;
        if (search_steps + 1 < search_tokens_size
                && *(search_tokens - (search_steps + 1)) == *(tail_tokens - tail_steps)) {
            search_steps++;
            continue;
        } else if (!overlaps || tail_steps + 1 <= search_steps) {
            if (tail_steps + 1 < tail_tokens_size &&
                    *(tail_tokens - (tail_steps + 1)) == *(search_tokens - search_steps)) {
                tail_steps++;
                continue;
            }
        }

        // A tolerance charge can count as a match, but only if we can find a
        // real match before the search is terminated.
        pending_matches++;

        tail_steps++;
        search_steps++;
    }
    return matches;
}

// Note: Only handles partial sequences, can't handle ones that are simply malformed.
static void seqrep_check_utf8(
        const char * s, const size_t len,
        const char ** first_valid,
        const char ** last_valid) {
    size_t expect_bytes = 0;
    const char * maybe_valid = NULL;
    *first_valid = *last_valid = NULL;

    for (size_t i = 0; i < len; i++) {
        const uint8_t c = uint8_t(s[i]);

        if (expect_bytes > 0) {
            expect_bytes--;
            // 10xxxxxxb -> 10b == 2
            if (c >> 6 == 2) {
                if (expect_bytes == 0) {
                    if (*first_valid == NULL) {
                        *first_valid = maybe_valid != NULL ? maybe_valid : s + i;
                    }
                    *last_valid = s + i;
                    maybe_valid = NULL;
                }
            } else {
                // Invalid sequence
                maybe_valid = *first_valid = *last_valid = NULL;
                expect_bytes = 0;
            }
            continue;
        }

        // Not in a sequence. First check for a single byte character.
        if ((c & 128) == 0) {
            if (*first_valid == NULL) {
                *first_valid = s + i;
            }
            *last_valid = s + i;
            maybe_valid = NULL;
            continue;
        }

        // If we end up here it's either the start of a multi byte sequence or invalid.
        maybe_valid = s + i;
        // 110xxxxxb -> 110b == 6
        if (c >> 5 == 6) {
            expect_bytes = 1;
        // 1110xxxxb -> 1110b == 14
        } else if (c >> 4 == 14) {
            expect_bytes = 2;
        // 11110xxxb -> 11110b == 30
        } else if (c >> 3 == 30) {
            expect_bytes = 3;
        // Invalid
        } else {
            maybe_valid = *first_valid = *last_valid = NULL;
        }
    }
}

// FIXME: Make this efficient.
static std::wstring seqrep_get_tail_string(const struct llama_context * ctx, const std::vector<llama_token> & last_tokens, size_t len) {
    const size_t last_tokens_len = last_tokens.size();
    std::string buf;

    len = std::min(len, last_tokens_len);
    if (len == 0) return std::wstring();

    buf.reserve(8 * len);

    const llama_token *curr_token = last_tokens.data() + (last_tokens_len - len);

    for (size_t i = 0; i < len; i++, curr_token++) {
        buf.append(llama_token_to_piece(ctx, *curr_token));
    }

    const char * first_valid = NULL, * last_valid = NULL;

    if (!buf.empty()) {
        seqrep_check_utf8(buf.data(), buf.size(), &first_valid, &last_valid);
    }
    if (first_valid == NULL) return std::wstring();

    return utf8_to_wstring(first_valid, last_valid + 1);
}

// Helper function for sequence matching.
// Bit 1 set indicates token is a word boundary. NL, " blah", "," - word boundary. "blah", "blah:" - not a word boundary.
// Bit 2 set indicates token ends on a word boundary. NL, "blah:", "blah " - ends on word boundary. " blah", "blah" - doesn't end on word boundary.
// Bit 3 set indicates all codepoints in the character count as boundary.
// FIXME: Handle partial/invalid UTF8 (just crashes currently).
int llama_seqrep_check_word(
        const struct llama_context * ctx,
        const llama_token token,
        std::vector<char> & buf) {
    const llama_model * model = llama_get_model(ctx);
    if (token == llama_token_bos(model) || token == llama_token_eos(model) || token == llama_token_nl(model)) {
        // BOS, EOS, NL are always a boundary.
        return SEQREP_CW_START_IS_WBOUND | SEQREP_CW_END_IS_WBOUND | SEQREP_CW_ALL_WS_PUNCT;
    }
    if (buf.size() < 128) buf.resize(128);

    int n_chars = llama_token_to_piece(model, token, buf.data(), buf.size() - 1);
    if (n_chars < 0) {
        buf.resize(size_t(-n_chars) + 128);
        const int check = llama_token_to_piece(model, token, buf.data(), buf.size() - 1);
        GGML_ASSERT(check == -n_chars);
        n_chars = check;
    } else if (n_chars == 0) {
        return 0;
    }
    buf[n_chars] = 0;

    const char * first_valid = NULL, * last_valid = NULL;

    seqrep_check_utf8(buf.data(), n_chars, &first_valid, &last_valid);

    // If first_valid != NULL then last_valid also must be != NULL.
    if (first_valid == NULL) {
        return SEQREP_CW_START_IS_WBOUND | SEQREP_CW_END_IS_WBOUND
            | SEQREP_CW_START_IS_INVALID | SEQREP_CW_END_IS_INVALID;
    }

    int result = 0;
    const bool start_invalid = first_valid > buf.data();
    const bool end_invalid = last_valid < (buf.data() + (n_chars - 1));
    std::wstring decoded = utf8_to_wstring(first_valid, last_valid + 1);
    size_t decoded_len = decoded.size();

    if (start_invalid) result |= SEQREP_CW_START_IS_INVALID;
    if (end_invalid) result |= SEQREP_CW_END_IS_INVALID;
    if (decoded_len == 0) return result;

    // Can only be all punctuation if the full sequence is valid.
    result |= !start_invalid && !end_invalid ? SEQREP_CW_ALL_WS_PUNCT : 0;

    for (size_t i = 0; i < decoded_len; i++) {
        wchar_t c = decoded[i];
        bool iswbound = c != L'\'' && c != L'’' && (std::iswpunct(c) || std::iswspace(c));

        if (!iswbound) {
            result &= ~SEQREP_CW_ALL_WS_PUNCT;
            continue;
        }

        if (i == 0 && !start_invalid)
            result |= SEQREP_CW_START_IS_WBOUND;
        if (i == decoded_len - 1 && !end_invalid)
            result |= SEQREP_CW_END_IS_WBOUND;
    }
    return result;
}

static void seqrep_apply_penalties(
        const struct llama_context        * ctx,
        const llama_token                 * last_tokens_p,
        const size_t                        last_tokens_size,
        llama_token_data_array            * candidates,
        const llama_sampler_seqrep_params * params,
        const std::unordered_map<llama_token, size_t> & penalize_tokens) {
    std::vector<char> buf(128, 0);
    const int flags = params->flags;

    const bool ends_on_word = params->mid_word_scale == 1.0f
        || SR_FLAG(llama_seqrep_check_word(ctx, last_tokens_p[last_tokens_size - 1], buf), SEQREP_CW_END_IS_WBOUND);

    for (size_t i = 0; i < candidates->size; ++i) {
        auto pt_iter = penalize_tokens.find(candidates->data[i].id);
        if (pt_iter == penalize_tokens.end()) {
            continue;
        }

        const size_t count = pt_iter->second;
        const bool pt_starts_word = params->mid_word_scale == 1.0f ||
            SR_FLAG(llama_seqrep_check_word(ctx, candidates->data[i].id, buf), SEQREP_CW_START_IS_WBOUND);
        float penalty_scale = ends_on_word || pt_starts_word ? 1.0f : params->mid_word_scale;
        float logit = candidates->data[i].logit;

        if (SR_FLAG(flags, LLAMA_SEQREP_ABSOLUTE_PENALTY)) {
            float penalty =
                ( float(count) * params->length_penalty
                + float(count > 0) * params->presence_penalty );
            logit -= penalty * penalty_scale;
        } else {
            const float l_penalty = (params->length_penalty != 0 ? params->length_penalty : 1.0) - 1.0;
            const float p_penalty = (params->presence_penalty != 0 ? params->presence_penalty : 1.0) - 1.0;

            // This looks complicated. The point is to scale be able to scale penalties like
            // 1.2. For example, suppose length penalty is 1.2 and length is 3. 1.2 * 3 = 3.6
            // would be ridiculous. What we actually want is more like 1.6.
            // An alternative approach would be to iteratively apply the scale.
            // 10.0 / 1.6 == 6.25, however ((10.0 / 1.2) / 1.2) / 1.2 == 5.787
            float penalty =
                ( (float(count) * l_penalty)
                + (float(count > 0) * p_penalty) ) * penalty_scale
                + 1.0f;
            if (logit <= 0) {
                logit *= penalty;
            } else if (penalty != 0.0f) {
                logit /= penalty;
            }
        }
        candidates->data[i].logit = logit;
    }

}


size_t llama_sample_seqrep_penalty(
        struct llama_context * ctx,
        llama_token_data_array * candidates,
        const std::vector<llama_token> & last_tokens,
        const llama_sampler_seqrep_params * params) {

    const size_t min_length = params->min_length;
    const int flags = params->flags;
    size_t last_tokens_size = last_tokens.size();
    const llama_token *last_tokens_p = last_tokens.data();

    if (params->last_n == 0 || params->min_length < 1) {
        return 0;
    } else if (params->last_n > 0) {
        size_t window_offset = last_tokens_size - std::min(size_t(params->last_n), last_tokens_size);

        last_tokens_size -= window_offset;
        last_tokens_p += window_offset;
    }

    if (last_tokens_size == 0 || (min_length > 1 && last_tokens_size <= min_length)) {
        return 0;
    } else if (!SR_FLAG(params->flags, LLAMA_SEQREP_REWIND_MODE)) {
        const float disabled = SR_FLAG(params->flags, LLAMA_SEQREP_ABSOLUTE_PENALTY) ? 0.0f : 1.0f;
        // We accept 0.0 here even when the penalty isn't absolute because a non-absolute
        // penalty of 0.0 implies divide by zero which makes no sense.
        if (   (params->presence_penalty == disabled || params->presence_penalty == 0)
            && (params->length_penalty == disabled || params->length_penalty == 0)) {
            return 0;
        }
    }

    if (params->mid_word_scale != 1.0f || SR_FLAG(params->flags, LLAMA_SEQREP_REWIND_SKIP_WS_PUNCT)) {
        // Only need ctx when mid_word_scale or REWIND_SKIP_WS_PUNCT flag is in effect.
        assert(ctx);
    }

    // const int64_t t_start_sample_us = ggml_time_us();

    // This will hold a map of token ids that can continue the sequence with its sequence length.
    std::unordered_map<llama_token, size_t> penalize_tokens;

    if (min_length > 1) {
        // Normal sequence matching mode.
        size_t start_offset       = params->start_offset;
        size_t max_matched_length = 0;
        size_t min_matched_length = last_tokens_size;

        if (start_offset == 0 || start_offset >= last_tokens_size - 1) {
            start_offset = last_tokens_size - 2;
        }

        const llama_token * tail_p = last_tokens_p + (last_tokens_size - 1);
        const size_t tail_len = std::min(params->max_length, last_tokens_size);

        for (size_t offset = start_offset; offset >= min_length - 1; offset--) {
            const llama_token * search_p = last_tokens_p + offset;
            const size_t search_len = std::min(params->max_length, last_tokens_size - (offset + 1));
            const size_t matched_length =
                seqrep_find_match(tail_p, tail_len, search_p, search_len, true, params);

            if (matched_length < min_length) {
                continue;
            }

            max_matched_length = std::max(max_matched_length, matched_length);
            min_matched_length = std::min(min_matched_length, matched_length);

            // The token one past where we started trying to match is the one that could continue
            // the previously observed sequence.
            llama_token penalize_token = last_tokens_p[offset + 1];

            auto pt_iter = penalize_tokens.find(penalize_token);
            if (pt_iter == penalize_tokens.end()
                    || !SR_FLAG(flags, LLAMA_SEQREP_PENALIZE_LENGTH_MAX_SEEN)) {
                penalize_tokens[penalize_token] += matched_length;
            } else {
                penalize_tokens[penalize_token] = std::max(pt_iter->second, matched_length);
            }
        }

        if ((flags & LLAMA_SEQREP_REWIND_MODE) != 0) {
            size_t result = !SR_FLAG(flags, LLAMA_SEQREP_REWIND_USE_SHORTEST_MATCH) || max_matched_length < min_length
                ? max_matched_length
                : min_matched_length;

            if (max_matched_length > 0 && SR_FLAG(params->flags, LLAMA_SEQREP_REWIND_SKIP_WS_PUNCT)) {
                std::vector<char> buf(128, 0);
                for (size_t i = last_tokens_size - result; i < last_tokens_size; i++) {
                    if (SR_FLAG(llama_seqrep_check_word(ctx, last_tokens_p[i], buf), SEQREP_CW_ALL_WS_PUNCT)) {
                        result--;
                    } else {
                        break;
                    }
                }
            }
            return result;
        }
    } else {
        // Single token matching mode. Can emulate existing repetition, presence and frequency samplers.
        size_t start_offset = params->start_offset;

        if (start_offset == 0 || start_offset >= last_tokens_size) {
            start_offset = last_tokens_size - 1;
        }

        for (int i = int(start_offset); i >= 0; i--) {
            llama_token penalize_token = last_tokens_p[i];

            if (SR_FLAG(flags, LLAMA_SEQREP_PENALIZE_LENGTH_MAX_SEEN)) {
                penalize_tokens[penalize_token] = 1;
            } else {
                penalize_tokens[penalize_token]++;
            }
        }
    }

    seqrep_apply_penalties(ctx, last_tokens_p, last_tokens_size, candidates, params, penalize_tokens);

    if (!penalize_tokens.empty()) {
        candidates->sorted = false;
    }

    // FIXME: Find a way to set stuff in ctx
    // if (ctx) {
    //     ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    // }
    return 0;
}

seqrep_logit_info::seqrep_logit_info(llama_context * ctx, const size_t k, const int32_t ith)
  : n_vocab(llama_n_vocab(llama_get_model(ctx)))
  , token_data(top_k(llama_get_logits_ith(ctx, ith), k))
  { }

const std::vector<llama_token_data> & seqrep_logit_info::get_token_data(void) {
    return token_data;
}

llama_token_data seqrep_logit_info::get_token_id(const llama_token token_id) const {
    for (const llama_token_data & td : token_data) {
        if (td.id == token_id)
            return td;
    }
    return {-1, 0, 0};
}

void seqrep_logit_info::rebuild(llama_context *ctx, const size_t k, const int32_t ith) {
    token_data = top_k(llama_get_logits_ith(ctx, ith), k);
}

void seqrep_logit_info::populate_logits(float * logits) {
    const float neginf = std::numeric_limits<float>::infinity() * -1;
    for (int i = 0; i < n_vocab; i++) {
        logits[i] = neginf;
    }
    for (const llama_token_data & td : token_data) {
        logits[td.id] = td.logit;
    }
}

// Yoinked from beam search code.
// Return top k token_data by logit.
std::vector<llama_token_data> seqrep_logit_info::top_k(
        const float * const logits,
        const size_t k) {

    std::vector<llama_token_data> min_heap;  // min-heap by logit
    const llama_token k_min = std::min(static_cast<llama_token>(k), n_vocab);
    min_heap.reserve(k_min);
    constexpr auto p = std::numeric_limits<float>::quiet_NaN();  // never used
    for (llama_token token_id = 0 ; token_id < k_min ; ++token_id) {
        const llama_token_data td = {token_id, logits[token_id], p};
        min_heap.push_back(td);
    }
    auto comp = [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; };
    std::make_heap(min_heap.begin(), min_heap.end(), comp);
    for (llama_token token_id = k_min ; token_id < n_vocab ; ++token_id) {
        if (min_heap.front().logit < logits[token_id]) {
            std::pop_heap(min_heap.begin(), min_heap.end(), comp);
            min_heap.back().id = token_id;
            min_heap.back().logit = logits[token_id];
            std::push_heap(min_heap.begin(), min_heap.end(), comp);
        }
    }
    return min_heap;
}


seqrep_rewind_state::seqrep_rewind_state(
        const size_t n_vocab,
        const size_t n_ctx,
        const size_t k)
  : n_vocab(n_vocab)
  , n_ctx(n_ctx)
  , k(k)
{
    logit_slots.reserve(n_ctx);
    rewind_slots.resize(n_ctx);
}

void seqrep_rewind_state::set_logits_slot(llama_context * ctx, const size_t idx, const int32_t ith) {
    GGML_ASSERT(idx <= logit_slots.size());
    if (idx == logit_slots.size()) {
        logit_slots.emplace_back(ctx, k, ith);
    } else {
        logit_slots[idx].rebuild(ctx, k, ith);
    }
}

struct seqrep_rewind_slot & seqrep_rewind_state::get_rewind_slot(const size_t idx) {
    GGML_ASSERT(idx <= rewind_slots.size());
    return rewind_slots[idx];
}

void seqrep_rewind_state::populate_logits(llama_context * ctx, const size_t idx, const int32_t ith) {
    logit_slots[idx].populate_logits(llama_get_logits_ith(ctx, ith));
}

static size_t seqrep_check_rewind_internal(
        struct llama_context * ctx,
        const std::vector<llama_token> & last_tokens,
        const std::vector<llama_sampler_seqrep_params> & params_list,
        const llama_sampler_seqrep_params & merged_params,
        size_t * high_water_mark) {
    const size_t last_tokens_size = last_tokens.size();

    size_t min_matched_len = 0, max_matched_len = 0;

    for (auto & sr_params : params_list) {
        if (!SR_FLAG(sr_params.flags, LLAMA_SEQREP_REWIND_MODE))
            continue;
        const size_t matched_len = llama_sample_seqrep_penalty(ctx, NULL, last_tokens, &sr_params);
        max_matched_len = std::max(max_matched_len, matched_len);
        min_matched_len = min_matched_len == 0
            ? matched_len
            : std::min(min_matched_len, matched_len);
    }
    if (max_matched_len < 2 || max_matched_len >= last_tokens_size) {
        return 0;
    }

    const size_t matched_len = !SR_FLAG(merged_params.flags, LLAMA_SEQREP_REWIND_USE_SHORTEST_MATCH)
        ? max_matched_len
        : min_matched_len;
    size_t idx = last_tokens_size - matched_len;

    if (idx < *high_water_mark) {
        if (*high_water_mark >= last_tokens_size - 2) {
            return 0;
        }
        idx = *high_water_mark;
    }

    if (merged_params.rewind_seek_word_boundary > 0) {
        std::vector<char> buf(128, 0);
        const size_t orig_idx = idx;
        bool found_idx = false;

        for (size_t steps = merged_params.rewind_seek_word_boundary + 1; idx >= *high_water_mark && steps > 0; idx--, steps--) {
            if (SR_FLAG(llama_seqrep_check_word(ctx, last_tokens[idx], buf), SEQREP_CW_START_IS_WBOUND)
                || SR_FLAG(llama_seqrep_check_word(ctx, last_tokens[idx - 1], buf), SEQREP_CW_END_IS_WBOUND)) {
                found_idx = true;
                break;
            }
        }
        if (!found_idx) {
            idx = orig_idx;
            for (size_t steps = merged_params.rewind_seek_word_boundary + 1; idx < last_tokens_size && steps > 0; idx++, steps--) {
                if (SR_FLAG(llama_seqrep_check_word(ctx, last_tokens[idx], buf), SEQREP_CW_START_IS_WBOUND)
                    || SR_FLAG(llama_seqrep_check_word(ctx, last_tokens[idx - 1], buf), SEQREP_CW_END_IS_WBOUND)) {
                    found_idx = true;
                    break;
                }
            }
            if (!found_idx || last_tokens_size - idx < merged_params.rewind_min_length) {
                if (SR_FLAG(merged_params.flags, LLAMA_SEQREP_REWIND_REQUIRE_WBOUND)) {
                    return 0;
                }
                idx = orig_idx;
            }
        }
    }

    const size_t rewind_distance = last_tokens.size() - idx;
    if (merged_params.rewind_min_length != 0 && rewind_distance < merged_params.rewind_min_length) {
        return 0;
    }

    return rewind_distance;
}

size_t llama_seqrep_handle_rewind(
        struct llama_context * ctx,
        struct seqrep_rewind_state & rewind_state,
        const std::vector<llama_token> & generated_tokens,
        const size_t n_generated,
        const std::vector<llama_token> & prompt_tokens,
        const std::vector<llama_sampler_seqrep_params> & params_list,
        size_t * high_water_mark,
        const int32_t ith) {
    const size_t prompt_tokens_size = prompt_tokens.size();

    if (n_generated < 3) {
        return 0;
    }

    // FIXME: This copying is inefficient.
    std::vector<llama_token> last_tokens;
    // printf("<%zu,%zu,%zu>", prompt_tokens_size, generated_tokens.size(), n_generated);
    // fflush(stdout);
    last_tokens.resize(n_generated + prompt_tokens.size());
    std::copy(prompt_tokens.begin(), prompt_tokens.end(), last_tokens.begin());
    std::copy(generated_tokens.begin(), generated_tokens.end(), last_tokens.begin() + prompt_tokens.size());

    llama_sampler_seqrep_params merged_params = llama_seqrep_merge_params(params_list, LLAMA_SEQREP_REWIND_MODE, 0);
    size_t rewind_distance = 0;
    size_t slot_idx, token_idx;
    std::vector<char> rewind_token_text_buf(128, 0);

    while (true) {
        rewind_distance = seqrep_check_rewind_internal(
            ctx, last_tokens, params_list, merged_params, high_water_mark );

        if (rewind_distance == 0)
            break;

        GGML_ASSERT(rewind_distance < n_generated);
        slot_idx = n_generated - rewind_distance;
        token_idx = n_generated + prompt_tokens_size - rewind_distance;

        const size_t ban_length = std::min(rewind_distance, merged_params.rewind_ban_length);
        struct seqrep_rewind_slot &rw_slot = rewind_state.get_rewind_slot(slot_idx);
        const bool at_wbound = token_idx == 0 ||
            SR_FLAG(llama_seqrep_check_word(ctx, last_tokens[token_idx - 1], rewind_token_text_buf), SEQREP_CW_END_IS_WBOUND);

        for (size_t i = slot_idx; i < slot_idx + ban_length; i++) {
            const llama_token penalize_token = generated_tokens[i];
            if (i > slot_idx
                    && !at_wbound
                    && !SR_FLAG(llama_seqrep_check_word(ctx, penalize_token, rewind_token_text_buf), SEQREP_CW_START_IS_WBOUND)) {
                continue;
            }
            if (std::find(rw_slot.tokens.begin(), rw_slot.tokens.end(), penalize_token) == rw_slot.tokens.end()) {
                rw_slot.tokens.push_back(penalize_token);
            }
        }

        if (++rw_slot.count >= merged_params.rewind_max_visits) {
            // This slot already hit max visits so we can set the HWM to the index one past it.
            *high_water_mark = token_idx + 1;
        }

        GGML_ASSERT(slot_idx > 0);
        break;
    }

    if (rewind_distance == 0) return 0;

    {
        const std::wstring tail = !merged_params.include_re.empty() || !merged_params.exclude_re.empty()
        ? seqrep_get_tail_string(ctx, last_tokens, rewind_distance + 8)
        : std::wstring();
        for (const auto & re : merged_params.include_re) {
           if (!std::regex_search(tail, re)) return 0;
        }
        for (const auto & re : merged_params.exclude_re) {
            if (std::regex_search(tail, re)) return 0;
        }
        // {
        //     std::string x = wstring_to_string(tail);
        //     printf(" [[ %s ]] ", x.c_str());
        // }
    }

    GGML_ASSERT(slot_idx > 0 && "Invalid slot for populate logits");
    rewind_state.populate_logits(ctx, slot_idx, ith);

    float * logits = llama_get_logits_ith(ctx, ith);
    const float neg_infinity = std::numeric_limits<float>::infinity() * -1;
    const size_t target_idx = token_idx;
    const bool at_wbound = target_idx == 0 ||
        SR_FLAG(llama_seqrep_check_word(ctx, last_tokens[target_idx - 1], rewind_token_text_buf), SEQREP_CW_END_IS_WBOUND);
    const bool persist_require_wbound = SR_FLAG(merged_params.flags, LLAMA_SEQREP_REWIND_PERSIST_REQUIRE_WBOUND);
    const size_t persist_count = std::min(prompt_tokens_size - target_idx, merged_params.rewind_persist_bans);

    for (size_t i = target_idx - persist_count; i <= target_idx; i++) {
        // FIXME: There's a better way to calculate this.
        if (i < prompt_tokens_size) {
            continue;
        }
        if (persist_require_wbound && i != target_idx && !at_wbound) {
            // We don't apply this logic when i == target_idx because the previous
            // checks should have taken it into account when the specific ban was applied
            // initially.
            continue;
        }
        for (const llama_token token_id : rewind_state.get_rewind_slot(i - prompt_tokens_size).tokens) {
           logits[token_id] = neg_infinity;
        }
    }

    return rewind_distance;
}


// Note: Doesn't merge presence or length penalties because of the divide_by_penalty flag.
struct llama_sampler_seqrep_params llama_seqrep_merge_params(
        const std::vector<llama_sampler_seqrep_params> & params_list,
        const int and_flags,
        const int not_flags) {
    struct llama_sampler_seqrep_params result = {};

    for (auto & sr_params : params_list) {
        if ((sr_params.flags & and_flags) != and_flags || (sr_params.flags & not_flags) != 0) {
            continue;
        }
        result.flags |= sr_params.flags;
        result.min_length = std::max(result.min_length, sr_params.min_length);
        result.max_length = std::max(result.max_length, sr_params.max_length);
        result.last_n = sr_params.last_n < 0 || result.last_n < 0
            ? -1
            : std::max(result.last_n, sr_params.last_n);
        result.tolerance = std::max(result.tolerance, sr_params.tolerance);
        result.mid_word_scale = std::max(result.mid_word_scale, sr_params.mid_word_scale);
        result.tolerance_match_credit = std::max(result.tolerance_match_credit, sr_params.tolerance_match_credit);
        result.rewind_min_length = std::max(result.rewind_min_length, sr_params.rewind_min_length);
        result.rewind_seek_word_boundary = std::max(result.rewind_seek_word_boundary, sr_params.rewind_seek_word_boundary);
        result.rewind_max_visits = std::max(result.rewind_max_visits, sr_params.rewind_max_visits);
        result.rewind_persist_bans = std::max(result.rewind_persist_bans, sr_params.rewind_persist_bans);
        result.rewind_ban_length = std::max(result.rewind_ban_length, sr_params.rewind_ban_length);
        // FIXME: Copying like this isn't ideal.
        result.include_re.insert(result.include_re.end(), sr_params.include_re.begin(), sr_params.include_re.end());
        result.exclude_re.insert(result.exclude_re.end(), sr_params.exclude_re.begin(), sr_params.exclude_re.end());
    }
    return result;
}
