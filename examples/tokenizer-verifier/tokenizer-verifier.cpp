#include "common.h"
#include "llama.h"

#include <cstdio>
#include <string>
#include <vector>

static int unicode_to_utf8(int codepoint, char *dest) {
    // https://stackoverflow.com/a/4609989 — who needs iconv?
    if (codepoint < 0x80) {
        *dest++ = codepoint;
    } else if (codepoint < 0x800) {
        *dest++ = 192 + codepoint / 64, *dest++ = 128 + codepoint % 64;
        // we also support reserved utf-16 surrogates 0xd800 - 0xdfff for simplicity
    } else if (codepoint < 0x10000) {
        *dest++ = 224 + codepoint / 4096, *dest++ = 128 + codepoint / 64 % 64,
            *dest++ = 128 + codepoint % 64;
    } else if (codepoint < 0x110000) {
        *dest++ = 240 + codepoint / 262144, *dest++ = 128 + codepoint / 4096 % 64,
            *dest++ = 128 + codepoint / 64 % 64, *dest++ = 128 + codepoint % 64;
    } else {
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: %s MODEL_PATH\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = true;
    llama_model *model = llama_load_model_from_file(model_path, model_params);

    std::vector<llama_token> tokens;

    int failed_ascii = 0;
    int ascii_max = 127;
    for (int c = 0; c <= ascii_max; c++) {
        const char prompt[] = {(char)c, '\0'};
        try {
            tokens = ::llama_tokenize(model, prompt, false, true);
        } catch (...) {
            printf("%#x -> Tokenization failed for char '%c'\n", c, (char)c);
            failed_ascii += 1;
            continue;
        }
    }
    printf("%d/%d 7-bit ascii characters could not be tokenized\n", failed_ascii, ascii_max);

    int failed_unicode = 0;
    int utf8_max = 0x10FFFF;
    // Now let's do all potential codepoints
    for (int cp = 0; cp <= utf8_max; cp++) {
        char buf[5] = {};
        if (unicode_to_utf8(cp, buf)) {
            printf("Impossible to encode codepoint %#x\n", cp);
            continue;
        }
        try {
            tokens = ::llama_tokenize(model, buf, false, true);
        } catch (...) {
            // printf("%#x -> Tokenization failed for codepoint '%s'\n", cp, buf);
            failed_unicode += 1;
            continue;
        }
    }
    printf("%d/%d potential unicode codepoints not tokenized\n", failed_unicode,
            utf8_max);

    return (failed_ascii != 0 || failed_unicode != 0);
}
