#pragma once

#include "llama.h"
#include "utils.h"

int run(llama_context * ctx,
        gpt_params params,
        std::istream & instream,
        FILE *outstream,
        FILE *errstream);
