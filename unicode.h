#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <locale>
#include <codecvt>

#define CODEPOINT_TYPE_UNIDENTIFIED 0
#define CODEPOINT_TYPE_DIGIT        1
#define CODEPOINT_TYPE_LETTER       2
#define CODEPOINT_TYPE_WHITESPACE   3
#define CODEPOINT_TYPE_ACCENT_MARK  4
#define CODEPOINT_TYPE_PUNCTUATION  5
#define CODEPOINT_TYPE_SYMBOL       6
#define CODEPOINT_TYPE_CONTROL      7

std::string unicode_cpt_to_utf8(uint32_t cp);
std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8);

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts);

int unicode_cpt_type(uint32_t cp);
int unicode_cpt_type(const std::string & utf8);

std::string unicode_byte_to_utf8(uint8_t byte);
uint8_t unicode_utf8_to_byte(const std::string & utf8);

char32_t unicode_tolower(char32_t cp);

std::vector<std::wstring> get_gpt2_regex();
std::vector<std::wstring> get_deepseek_coder_regex();
std::vector<std::wstring> get_deepseek_llm_regex();

inline std::wstring from_utf8(const std::string & s);
inline std::string to_utf8(const std::wstring & ws);