#include "grammar-parser.h"
#include <cstdint>
#include <cwchar>
#include <string>
#include <utility>

namespace grammar_parser {
    uint16_t get_symbol_id(parse_state & state, const char * src, size_t len) {
        uint16_t next_id = static_cast<uint16_t>(state.symbol_ids.size());
        auto result = state.symbol_ids.insert(std::make_pair(std::string(src, len), next_id));
        return result.first->second;
    }

    uint16_t generate_symbol_id(parse_state & state, const std::string & base_name) {
        uint16_t next_id = static_cast<uint16_t>(state.symbol_ids.size());
        state.symbol_ids[base_name + '_' + std::to_string(next_id)] = next_id;
        return next_id;
    }

    bool is_word_char(char c) {
        return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' || ('0' <= c && c <= '9');
    }

    int hex_to_int(char c) {
        if ('a' <= c && c <= 'f') {
            return c - 'a' + 10;
        } else if ('A' <= c && c <= 'F') {
            return c - 'A' + 10;
        } else if ('0' <= c && c <= '9') {
            return c - '0';
        }
        return -1;
    }

    const char * parse_space(const char * src, bool newline_ok) {
        const char * pos = src;
        while (*pos == ' ' || *pos == '\t' || *pos == '#' ||
                (newline_ok && (*pos == '\r' || *pos == '\n'))) {
            if (*pos == '#') {
                while (*pos && *pos != '\r' && *pos != '\n') {
                    pos++;
                }
            } else {
                pos++;
            }
        }
        return pos;
    }

    const char * parse_name(const char * src) {
        const char * pos = src;
        while (is_word_char(*pos)) {
            pos++;
        }
        if (pos == src) {
            throw std::runtime_error(std::string("expecting name at ") + src);
        }
        return pos;
    }

    std::pair<uint16_t, const char *> parse_char(const char * src) {
        if (*src == '\\') {
            char esc = src[1];
            if (esc == 'x') {
                int first = hex_to_int(src[2]);
                if (first > -1) {
                    int second = hex_to_int(src[3]);
                    if (second > -1) {
                        return std::make_pair((first << 4) + second, src + 4);
                    }
                }
                throw std::runtime_error(std::string("expecting \\xNN at ") + src);
            } else if (esc == '"' || esc == '[' || esc == ']') {
                return std::make_pair(esc, src + 2);
            } else if (esc == 'r') {
                return std::make_pair('\r', src + 2);
            } else if (esc == 'n') {
                return std::make_pair('\n', src + 2);
            } else if (esc == 't') {
                return std::make_pair('\t', src + 2);
            }
            throw std::runtime_error(std::string("unknown escape at ") + src);
        } else if (*src) {
            return std::make_pair(*src, src + 1);
        }
        throw std::runtime_error("unexpected end of input");
    }

    const char * parse_alternates(
            parse_state       & state,
            const char        * src,
            const std::string & rule_name,
            uint16_t            rule_id,
            bool                is_nested);

    const char * parse_sequence(
            parse_state           & state,
            const char            * src,
            const std::string     & rule_name,
            std::vector<uint16_t> & outbuf,
            bool                    is_nested) {
        size_t out_start = outbuf.size();

        // sequence size, will be replaced at end when known
        outbuf.push_back(0);

        size_t last_sym_start = outbuf.size();
        const char * pos = src;
        while (*pos) {
            if (*pos == '"') { // literal string
                pos++;
                last_sym_start = outbuf.size();
                while (*pos != '"') {
                    auto char_pair = parse_char(pos);
                         pos       = char_pair.second;

                    // each char of a literal is encoded as a "range" of char - char
                    outbuf.push_back(2);
                    outbuf.push_back(char_pair.first);
                    outbuf.push_back(char_pair.first);
                }
                pos = parse_space(pos + 1, is_nested);
            } else if (*pos == '[') { // char range(s)
                pos++;
                last_sym_start = outbuf.size();
                // num chars in range - replaced at end of loop
                outbuf.push_back(0);
                while (*pos != ']') {
                    auto char_pair = parse_char(pos);
                         pos       = char_pair.second;

                    outbuf.push_back(char_pair.first);
                    if (pos[0] == '-' && pos[1] != ']') {
                        auto endchar_pair = parse_char(pos + 1);
                             pos          = endchar_pair.second;
                        outbuf.push_back(endchar_pair.first);
                    } else {
                        // chars that aren't part of a c1-c2 range are just doubled (i.e., c-c)
                        outbuf.push_back(char_pair.first);
                    }
                }
                // replace num chars with actual
                outbuf[last_sym_start] = static_cast<uint16_t>(outbuf.size() - last_sym_start - 1);
                pos = parse_space(pos + 1, is_nested);
            } else if (is_word_char(*pos)) { // rule reference
                const char * name_end    = parse_name(pos);
                uint16_t     ref_rule_id = get_symbol_id(state, pos, name_end - pos);
                pos = parse_space(name_end, is_nested);
                last_sym_start = outbuf.size();
                outbuf.push_back(1);
                outbuf.push_back(ref_rule_id);
            } else if (*pos == '(') { // grouping
                // parse nested alternates into synthesized rule
                pos = parse_space(pos + 1, true);
                uint16_t sub_rule_id = generate_symbol_id(state, rule_name);
                pos = parse_alternates(state, pos, rule_name, sub_rule_id, true);
                last_sym_start = outbuf.size();
                // output reference to synthesized rule
                outbuf.push_back(1);
                outbuf.push_back(sub_rule_id);
                if (*pos != ')') {
                    throw std::runtime_error(std::string("expecting ')' at ") + pos);
                }
                pos = parse_space(pos + 1, is_nested);
            } else if (*pos == '*' || *pos == '+' || *pos == '?') { // repetition operator
                if (outbuf.size() - out_start - 1 == 0) {
                    throw std::runtime_error(std::string("expecting preceeding item to */+/? at ") + pos);
                }
                std::vector<uint16_t> & out_grammar = state.out_grammar;

                // apply transformation to previous symbol (last_sym_start -
                // end) according to rewrite rules:
                // S* --> S' ::= S S' |
                // S+ --> S' ::= S S' | S
                // S? --> S' ::= S |
                uint16_t sub_rule_id = generate_symbol_id(state, rule_name);
                out_grammar.push_back(sub_rule_id);
                size_t   sub_rule_start = out_grammar.size();
                // placeholder for size of 1st alternate
                out_grammar.push_back(0);
                // add preceding symbol to generated rule
                out_grammar.insert(out_grammar.end(), outbuf.begin() + last_sym_start, outbuf.end());
                if (*pos == '*' || *pos == '+') {
                    // cause generated rule to recurse
                    out_grammar.push_back(1);
                    out_grammar.push_back(sub_rule_id);
                }
                // apply actual size
                out_grammar[sub_rule_start] = out_grammar.size() - sub_rule_start;
                // mark end of 1st alternate
                out_grammar.push_back(0);
                sub_rule_start = out_grammar.size();
                // placeholder for size of 2nd alternate
                out_grammar.push_back(0);
                if (*pos == '+') {
                    // add preceding symbol as alternate only for '+'
                    out_grammar.insert(out_grammar.end(), outbuf.begin() + last_sym_start, outbuf.end());
                }
                // apply actual size of 2nd alternate
                out_grammar[sub_rule_start] = out_grammar.size() - sub_rule_start;
                // mark end of 2nd alternate, then end of rule
                out_grammar.push_back(0);
                out_grammar.push_back(0);

                // in original rule, replace previous symbol with reference to generated rule
                outbuf.resize(last_sym_start);
                outbuf.push_back(1);
                outbuf.push_back(sub_rule_id);

                pos = parse_space(pos + 1, is_nested);
            } else {
                break;
            }
        }
        // apply actual size of this alternate sequence
        outbuf[out_start] = static_cast<uint16_t>(outbuf.size() - out_start);
        // mark end of alternate
        outbuf.push_back(0);
        return pos;
    }

    const char * parse_alternates(
            parse_state       & state,
            const char        * src,
            const std::string & rule_name,
            uint16_t            rule_id,
            bool                is_nested) {
        std::vector<uint16_t> outbuf;
        const char * pos = parse_sequence(state, src, rule_name, outbuf, is_nested);
        while (*pos == '|') {
            pos = parse_space(pos + 1, true);
            pos = parse_sequence(state, pos, rule_name, outbuf, is_nested);
        }
        state.out_grammar.push_back(rule_id);
        state.out_grammar.insert(state.out_grammar.end(), outbuf.begin(), outbuf.end());
        state.out_grammar.push_back(0);
        return pos;
    }

    const char * parse_rule(parse_state & state, const char * src) {
        const char * name_end = parse_name(src);
        const char * pos      = parse_space(name_end, false);
        size_t       name_len = name_end - src;
        uint16_t     rule_id  = get_symbol_id(state, src, name_len);
        const std::string name(src, name_len);

        if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) {
            throw std::runtime_error(std::string("expecting ::= at ") + pos);
        }
        pos = parse_space(pos + 3, true);

        pos = parse_alternates(state, pos, name, rule_id, false);

        if (*pos == '\r') {
            pos += pos[1] == '\n' ? 2 : 1;
        } else if (*pos == '\n') {
            pos++;
        } else if (*pos) {
            throw std::runtime_error(std::string("expecting newline or end at ") + pos);
        }
        return parse_space(pos, true);
    }

    parse_state parse(const char * src) {
        try {
            parse_state state;
            const char * pos = parse_space(src, true);
            while (*pos) {
                pos = parse_rule(state, pos);
            }
            state.out_grammar.push_back(0xffff);
            return state;
        } catch (const std::exception & err) {
            fprintf(stderr, "%s: error parsing grammar: %s\n", __func__, err.what());
            return parse_state();
        }
    }

    const uint16_t * print_rule(
            FILE           * file,
            const uint16_t * base,
            const uint16_t * src,
            const std::map<uint16_t, std::string> & symbol_id_names) {
        uint16_t rule_id = *src;
        fprintf(file, "<%zu>%s ::= ", src - base, symbol_id_names.at(rule_id).c_str());
        const uint16_t * pos = src + 1;
        while (*pos) {
            if (pos - 1 > src) {
                fprintf(file, "| ");
            }
            pos++; // sequence size, not needed here
            while (*pos) {
                if (*pos == 1) {
                    uint16_t ref_rule_id = pos[1];
                    fprintf(file, "<%zu>%s ", pos - base, symbol_id_names.at(ref_rule_id).c_str());
                    pos += 2;
                } else {
                    fprintf(file, "<%zu>[", pos - base);
                    uint16_t num_chars = *pos;
                    pos++;

                    for (uint16_t i = 0; i < num_chars; i += 2) {
                        fprintf(file, "%lc-", static_cast<wint_t>(pos[i])); // REVIEW
                        if (i + 1 < num_chars) {
                            fprintf(file, "%lc", static_cast<wint_t>(pos[i + 1]));
                        }
                    }
                    fprintf(file, "] ");
                    pos += num_chars;
                }
            }
            pos++;
        }
        fprintf(file, "\n");
        return pos + 1;
    }

    void print_grammar(FILE * file, const parse_state & state) {
        std::map<uint16_t, std::string> symbol_id_names;
        for (auto kv : state.symbol_ids) {
            symbol_id_names[kv.second] = kv.first;
        }
        const uint16_t * pos = state.out_grammar.data();
        while (*pos != 0xffff) {
            pos = print_rule(file, state.out_grammar.data(), pos, symbol_id_names);
        }
    }
}
