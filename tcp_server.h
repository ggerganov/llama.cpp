#pragma once

#include "utils.h"
#include "llama.h"
#include "run.h"

int listen_tcp(llama_context * ctx, gpt_params params);
