#include "utils.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

std::vector<gpt_vocab::id> legacy_llama_tokenize(const gpt_vocab & vocab, const std::string & text, bool bos);
