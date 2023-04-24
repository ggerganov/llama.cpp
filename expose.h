#pragma once

const int stop_token_max = 10;
struct load_model_inputs
{
    const int threads;
    const int max_context_length;
    const int batch_size;
    const bool f16_kv;
    const char * executable_path;
    const char * model_filename;
    const char * lora_filename;
    const bool use_mmap;
    const bool use_smartcontext;
    const bool unban_tokens;
    const int clblast_info = 0;
    const int blasbatchsize = 512;
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
    const char * stop_sequence[stop_token_max];
};
struct generation_outputs
{
    int status = -1;
    char text[16384]; //16kb should be enough for any response
};

extern std::string executable_path;
extern std::string lora_filename;