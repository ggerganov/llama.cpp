#pragma once

#include <cassert>
#include <cstring>
#include <fstream>
#include <regex>
#include <iostream>
#include <iterator>
#include <queue>
#include <string>
#include <math.h>
#include <vector>

#include "expose.h"

//return val: 0=fail, 1=(original ggml, alpaca), 2=(ggmf), 3=(ggjt) 
enum FileFormat
{
    FAIL=0,
    GGML=1,
    GGHF=2,
    GGJT=3,

    GPTJ1=100,
    GPTJ2=101,

    GPT2=200,
};

bool llama_load_model(const load_model_inputs inputs, FileFormat file_format);
generation_outputs llama_generate(const generation_inputs inputs, generation_outputs &output);
bool gptj_load_model(const load_model_inputs inputs, FileFormat in_file_format);
generation_outputs gptj_generate(const generation_inputs inputs, generation_outputs &output);


void timer_start();
double timer_check();
void print_tok_vec(std::vector<int> &embd);
FileFormat check_file_format(const std::string & fname);