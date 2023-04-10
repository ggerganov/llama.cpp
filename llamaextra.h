#pragma once
#include "common.h"

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

#include "llama.h"
#include "ggml.h"



std::vector<llama_token> legacy_llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos);