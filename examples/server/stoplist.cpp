#include "utils.hpp"

std::set<const char *> SWordsFilter::stoplist = {
    "<|endoftext|>",
    "<|im_end|>",
    "<|startoftext|>",
    "<|im_start|>"
};

SWordsFilter stopped_filter;
