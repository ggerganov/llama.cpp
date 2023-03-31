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

//return val: 0=fail, 1=(original ggml, alpaca), 2=(ggmf), 3=(ggjt) 
enum FileFormat
{
    FAIL=0,
    GGML=1,
    GGHF=2,
    GGJT=3    
};

FileFormat check_file_format(const std::string & fname);

std::vector<llama_token> legacy_llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos);
static bool legacy_llama_model_load(const std::string & fname, llama_context & lctx, int n_ctx, int n_parts, ggml_type memory_type, bool vocab_only, llama_progress_callback progress_callback, void *progress_callback_user_data);
struct llama_context * legacy_llama_init_from_file(const char * path_model, struct llama_context_params   params);