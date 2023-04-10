#pragma once

struct load_model_inputs
{
    const int threads;
    const int max_context_length;
    const int batch_size;
    const bool f16_kv;
    const char *model_filename;
    const int n_parts_overwrite = -1;
    const bool use_mmap;
};
struct generation_inputs
{
    const int seed;
    const char *prompt;
    const int max_context_length;
    const int max_length;
    const float temperature;
    const int top_k;
    const float top_p;
    const float rep_pen;
    const int rep_pen_range;
};
struct generation_outputs
{
    int status = -1;
    char text[16384]; //16kb should be enough for any response
};

