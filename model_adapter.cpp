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

void print_vec(std::vector<std::string> &embd)
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

 bool ArrStartWith(const std::vector<int> targetArray, const std::vector<int> searchSeq)
 {
     int ss = searchSeq.size();
     if(targetArray.size()<ss)
     {
         return false;
     }
     for(int i=0;i<ss;++i)
     {
         if(targetArray[i]!=searchSeq[i])
         {
             return false;
         }
     }
     return true;
 }

 int ArrFindIndexOf(const std::vector<int> targetArray, const std::vector<int> searchSeq)
 {
     int ss = searchSeq.size();
     int tas = targetArray.size();
     if(tas<ss)
     {
         return -1;
     }
     for(int i=0;i<tas;++i)
     {
         int srch = 0;
         bool fail = false;
         for(int srch=0;srch<ss;++srch)
         {
             if ((i + srch) >= tas || targetArray[i + srch] != searchSeq[srch])
             {
                 fail = true;
                 break;
             }
         }
         if(!fail)
         {
             return i;
         }
     }
     return -1;
 }

 std::vector<int> LongestCommonSubseq(const std::vector<int> x, const std::vector<int> y)
 {
     int m = x.size(), n = y.size();
     
     //int LCSuff[m+1][n+1];
     std::vector<std::vector<int>> LCSuff(m+1, std::vector<int>(n+1));    

     for (int j = 0; j <= n; j++)
         LCSuff[0][j] = 0;
     for (int i = 0; i <= m; i++)
         LCSuff[i][0] = 0;

     for (int i = 1; i <= m; i++)
     {
         for (int j = 1; j <= n; j++)
         {
             if (x[i - 1] == y[j - 1])
                 LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1;
             else
                 LCSuff[i][j] = 0;
         }
     }

     std::vector<int> longest;
     for (int i = 1; i <= m; i++)
     {
         for (int j = 1; j <= n; j++)
         {
             if (LCSuff[i][j] > longest.size())
             {
                 auto off1 = ((i - LCSuff[i][j] + 1) - 1);
                 auto off2 = off1 + LCSuff[i][j];
                 longest.clear();
                //  std::vector<int>().swap(longest);                 
                 longest = std::vector<int>(x.begin() + off1, x.begin() + off2);
                // x.substr((i - LCSuff[i][j] + 1) - 1, LCSuff[i][j]);
             }
         }
     }
     return longest;
 }

 void ContextFastForward(std::vector<int> &current_context_tokens, std::vector<int> &embd_inp,
 int &n_past, std::vector<int> &last_n_tokens, const int nctx, std::vector<int> &smartcontext, bool useSmartContext)
 {     
     const int SCCtxLenThreshold = nctx * 0.8; //how much context length must be reach to trigger smartcontext
     const int SCInpLenThreshold = nctx * 0.6; //how big must the input array be to trigger smartcontext
     const int SCPastLenThreshold = nctx * 0.5; //how wide of a gap between the fast forwarded past and the present to trigger smart context
     const float SCTruncationRatio = 0.5; //ratio for how many tokens to fast forward
     const int SCTokThreshold = 32 + (nctx*0.05); //how many tokens of similarity triggers smartcontext

    // printf("\nORIGINAL CTX:\n");
    // print_tok_vec(current_context_tokens);
    // printf("\nORIGINAL EMBD:\n");
    // print_tok_vec(embd_inp);

    //fast forward the past based on identical tokens, stop once a divergence is noted
    int embd_inp_len = embd_inp.size();
    for (int i = 0; i < current_context_tokens.size(); ++i)
    {
        if (current_context_tokens[i] == embd_inp[i])
        {
            n_past += 1;
            last_n_tokens.push_back(current_context_tokens[i]);
        }
        else
        {
            break;
        }
        if ((i + 2) >= embd_inp_len)
        {
            break;
        }
    }


    last_n_tokens.erase(last_n_tokens.begin(), last_n_tokens.begin() + n_past);
    embd_inp.erase(embd_inp.begin(), embd_inp.begin() + n_past);
    embd_inp_len = embd_inp.size();

    //smart context mode, detect if we have a shifted context at max length
    //requirement: previous context was at least nctx/2 longer than current,
    //mode is on, and current context already maxed.

    // printf("\nconds: %d %d %d\n",current_context_tokens.size() >= nctx*0.8
    // ,embd_inp_len >= nctx*0.6 ,current_context_tokens.size() - n_past > nctx*0.5);
    // printf("csiz:%d par:%d eilen:%d np:%d",current_context_tokens.size(), (int)(nctx*0.8),embd_inp_len,n_past);

    if (useSmartContext && smartcontext.size() > 0 && embd_inp_len >= SCInpLenThreshold)
    {
        // printf("curfullcontext:\n");
        // print_tok_vec(current_context_tokens);

        //see if smartcontext is still usable
        // printf("smartctx:\n");
        // print_tok_vec(smartcontext);
        // printf("embinp:\n");
        // print_tok_vec(embd_inp);
        auto shared = LongestCommonSubseq(smartcontext, embd_inp);       
        if (shared.size() > SCTokThreshold && ArrStartWith(smartcontext, shared)) //at least 32 tokens in common
        {            
            int found = ArrFindIndexOf(embd_inp,shared);
            if(found>=0)
            {
                auto trimmed = std::vector<int>(embd_inp.begin() + found, embd_inp.end());
                embd_inp = trimmed;
                embd_inp_len = embd_inp.size();
                // printf("trimmed:\n");
                // print_tok_vec(embd_inp,&vocab.id_to_token);
                printf("\n[Reusing Smart Context: %d allowance remaining]", found);

                int old_n_past = n_past;
                int offset_fix = old_n_past;
                if (current_context_tokens[n_past] != embd_inp[0])
                {
                    offset_fix = 0;
                }
                                
                for (int i = n_past; i < current_context_tokens.size(); ++i)
                {
                    //printf("\n%s and %s\n",vocab.id_to_token[current_context_tokens[i]].c_str(), vocab.id_to_token[embd_inp[i-offset_fix]].c_str());
                    if (current_context_tokens[i] == embd_inp[i-offset_fix])
                    {
                        n_past += 1;
                        last_n_tokens.push_back(current_context_tokens[i]);
                    }
                    else
                    {
                        break;
                    }
                    if ((i + 2 - offset_fix) >= embd_inp_len)
                    {
                        break;
                    }
                }

                last_n_tokens.erase(last_n_tokens.begin(), last_n_tokens.begin() + (n_past-old_n_past));
                embd_inp.erase(embd_inp.begin(), embd_inp.begin() + (n_past-old_n_past));
                // printf("np:%d newembinp: \n",n_past);
                // print_tok_vec(embd_inp);
            }else{
                smartcontext.clear();
            }
        }
        else
        {
            smartcontext.clear();
        }
    }
    else
    {
        smartcontext.clear();
    }

    if(useSmartContext 
    && smartcontext.size()==0 && current_context_tokens.size() >= SCCtxLenThreshold 
    && embd_inp_len >= SCInpLenThreshold 
    && current_context_tokens.size() - n_past > SCPastLenThreshold)
    {       
        //determine longest common substring after removing start part
        int shiftamt = embd_inp.size() * SCTruncationRatio;
        smartcontext = std::vector<int>(embd_inp.begin() + shiftamt, embd_inp.end());
         printf("\n[New Smart Context Triggered! Buffered Token Allowance: %d]",shiftamt);
        // printf("smartctx:\n");
        // print_tok_vec(smartcontext,&vocab.id_to_token);
        embd_inp = smartcontext;
        //if max ctx length is exceeded, chop the prompt in half after the start part, and memorize it. The memorized part becomes LCS marker.
        //when a future prompt comes in, find the LCS again. If LCS > a length and LCS starts with memorized LCS
        //remove all tokens between start part and start of LCS in new prompt, thus avoiding shift
        //if LCS not found or mismatched, regenerate. chop new prompt and repeat from step B
    }
 }