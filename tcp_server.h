#pragma once

#include "utils.h"
#include "llama.h"

int listen_tcp(
    gpt_params params,
    gpt_vocab vocab,
    llama_model model,
    int64_t t_main_start_us,
    int64_t t_load_us);
