
#include "extra.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <regex>
#include <iostream>
#include <iterator>
#include <queue>
#include <string>
#include <math.h>

 #if defined(_MSC_VER) || defined(__MINGW32__)
 #include <malloc.h> // using malloc.h with MSC/MINGW
 #elif !defined(__FreeBSD__) && !defined(__NetBSD__)
 #include <alloca.h>
 #endif

// TODO: Calculate this constant from the vocabulary
#define MAX_TOKEN_LEN 18
// SentencePiece implementation after https://guillaume-be.github.io/2020-05-30/sentence_piece
std::vector<gpt_vocab::id> legacy_llama_tokenize(const gpt_vocab & vocab, const std::string & text, bool bos) {
    std::vector<gpt_vocab::id> res;
    std::vector<int> score;
    std::vector<gpt_vocab::id> prev;
    int len = text.length();

    score.resize(len + 1);
    prev.resize(len + 1);

    // Forward pass
    for (int i = 0; i < len; i++) {
        int max_len = std::min(len - i, MAX_TOKEN_LEN);
        for (int sub_len = 1; sub_len <= max_len; sub_len++) {
            auto sub = text.substr(i, sub_len);
            auto token = vocab.token_to_id.find(sub);
            if (token != vocab.token_to_id.end()) {
                int token_score = sub.length() * sub.length();
                int local_score = score[i] + token_score;
                int next = i + sub_len;
                if (score[next] < local_score) {
                    score[next] = local_score;
                    prev[next] = (*token).second;
                }
            }
        }
    }

    // Backward pass
    int i = len;
    while (i > 0) {
        gpt_vocab::id token_id = prev[i];
        if (token_id == 0) {
	    // TODO: Return error or something more meaningful
            printf("failed to tokenize string!\n");
	    break;
        }
        res.push_back(token_id);
        auto token = (*vocab.id_to_token.find(token_id)).second;
        i -= token.length();
    }

    if (bos) {
        res.push_back(1); // TODO: replace with vocab.bos
    }

    // Pieces are in reverse order so correct that
    std::reverse(res.begin(), res.end());

    return res;
}