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

#include "model_adapter.h"

#include <chrono>

static auto bench_timer = std::chrono::high_resolution_clock().now();

void timer_start()
{
    bench_timer = std::chrono::high_resolution_clock().now();
}
double timer_check()
{
    auto endtime = std::chrono::high_resolution_clock().now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - bench_timer);    
    double time_taken = duration.count()/1000.0;
    return time_taken;
}

void print_tok_vec(std::vector<int> &embd)
{
    std::cout << "[";
    bool first = true;
    for (auto i : embd)
    {
        if (!first)
        {
            std::cout << ',';
        }
        first = false;
        std::cout << i;
    }
    std::cout << "]\n";
}
void print_tok_vec(std::vector<float> &embd)
{
    std::cout << "[";
    bool first = true;
    int n = 0;
    for (auto i : embd)
    {
        if (!first)
        {
            std::cout << ',';
        }
        first = false;
        std::cout << i;
        if(++n>20)
        {
            break;
        }
    }
    std::cout << "]\n";
}

//return val: 0=fail, 1=(original ggml, alpaca), 2=(ggmf), 3=(ggjt) 
 FileFormat check_file_format(const std::string & fname)
 {
    std::vector<char> f_buf(1024*1024);

    auto fin = std::ifstream(fname, std::ios::binary);
    fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return FileFormat::BADFORMAT;
    }

    FileFormat fileformat = FileFormat::BADFORMAT;
    uint32_t magic;
    fin.read((char *) &magic, sizeof(magic));
    if (magic == 0x67676d6c) {  //v1 format ggml, alpaca, old gptj and gpt2 models
       fileformat = FileFormat::GGML;
       //we need to read more to determine
       int32_t vocabsiz = 0;
       fin.read((char *) &vocabsiz, sizeof(int32_t));
       if(vocabsiz==50400) //know GPT-J vocab size
       {
           fileformat = FileFormat::GPTJ_1;
           uint32_t temp;
           fin.read((char *)&temp, sizeof(temp)); //ctx
           fin.read((char *)&temp, sizeof(temp)); //n_embd
           fin.read((char *)&temp, sizeof(temp)); //n_head
           fin.read((char *)&temp, sizeof(temp)); //n_layer
           fin.read((char *)&temp, sizeof(temp)); //n_rot
           fin.read((char *)&temp, sizeof(temp)); //f16
           if(temp!=0 && temp!=1)
           {
                fileformat = FileFormat::GPTJ_3; //quantized format cannot be legacy type
           }
       }
       if(vocabsiz==50257)
       {
           fileformat = FileFormat::GPT2_1;
           uint32_t temp;
           fin.read((char *)&temp, sizeof(temp)); //ctx
           fin.read((char *)&temp, sizeof(temp)); //n_embd
           fin.read((char *)&temp, sizeof(temp)); //n_head
           fin.read((char *)&temp, sizeof(temp)); //n_layer
           fin.read((char *)&temp, sizeof(temp)); //f16
           if(temp!=0 && temp!=1)
           {
                fileformat = FileFormat::GPT2_2; //quantized format cannot be legacy type
           }
           
       }
    }
    else if(magic == 0x67676d66) //v2 format ggmf
    {
        fileformat = FileFormat::GGHF;
    }
    else if(magic == 0x67676a74) //v3 format ggjt
    {
        fileformat = FileFormat::GGJT; //ggjt by default
    }
    fin.close();
    
    return fileformat;
 }