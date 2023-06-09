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

enum FileFormat
{
    BADFORMAT=0, //unknown, uninit, or failed to load
    GGML=1, // 1=(original llama ggml, alpaca, GPT4ALL, GPTJ header)
    GGHF=2, // 2=(llama ggmf)
    GGJT=3, // 3=(llama ggjt)
    GGJT_2=4, //newer llama format unshuffled
    GGJT_3=5, //using 16bit scalar

    GPTJ_1=100, //the very first super old GPTJ format
    GPTJ_2=101, //pygmalion, uses old ggml lib
    GPTJ_3=102, //uses new ggml lib
    GPTJ_4=103, //unshuffled
    GPTJ_5=104, //using 16bit scalar

    GPT2_1=200,
    GPT2_2=201,
    GPT2_3=202, //unshuffled
    GPT2_4=203, //using 16bit scalar

    RWKV_1=300,
    RWKV_2=301,

    NEOX_1=400,
    NEOX_2=401,
    NEOX_3=402, //redpajama
    NEOX_4=403, //unshuffled
    NEOX_5=404, //unshuffled redpajama
    NEOX_6=405, //using 16bit scalar
    NEOX_7=406, //using 16bit scalar redpajama

    MPT_1=500, //first supported mpt version
};

enum ModelLoadResult
{
    FAIL = 0,
    SUCCESS = 1,
    RETRY_LOAD = 2, //used if it's suspected that the model is an older format
};

ModelLoadResult gpttype_load_model(const load_model_inputs inputs, FileFormat in_file_format);
generation_outputs gpttype_generate(const generation_inputs inputs, generation_outputs &output);


void timer_start();
double timer_check();
void print_tok_vec(std::vector<int> &embd);
void print_tok_vec(std::vector<float> &embd);
void print_vec(std::vector<std::string> &embd);
std::vector<int> LongestCommonSubseq(const std::vector<int> x, const std::vector<int> y);
bool ArrStartWith(const std::vector<int> targetArray, const std::vector<int> searchSeq);
int ArrFindIndexOf(const std::vector<int> targetArray, const std::vector<int> searchSeq);

FileFormat check_file_format(const std::string & fname);
void ContextFastForward(std::vector<int> &current_context_tokens, std::vector<int> &embd_inp,
 int &n_past, std::vector<int> &last_n_tokens, const int nctx, std::vector<int> &smartcontext,
 const bool useSmartContext, const bool requireFullSubset);