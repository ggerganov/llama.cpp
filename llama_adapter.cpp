//This is Concedo's shitty adapter for adding python bindings for llama

//Considerations:
//Don't want to use pybind11 due to dependencies on MSVCC
//ZERO or MINIMAL changes as possible to main.cpp - do not move their function declarations here!
//Leave main.cpp UNTOUCHED, We want to be able to update the repo and pull any changes automatically.
//No dynamic memory allocation! Setup structs with FIXED (known) shapes and sizes for ALL output fields
//Python will ALWAYS provide the memory, we just write to it.

// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include "./examples/main/main.cpp"
#include "ggml.h"
#include "model_adapter.h"

//for easier compilation
#include "llamaextra.cpp"

//return val: 0=fail, 1=(original ggml, alpaca), 2=(ggmf), 3=(ggjt)
static FileFormat file_format = FileFormat::BADFORMAT;
static llama_context_params ctx_params;
static gpt_params params;
static int n_past = 0;
static int n_threads = 4;
static int n_batch = 8;
static bool useSmartContext = false;
static int blasbatchsize = 512;
static std::string modelname;
static llama_context *ctx;
static std::vector<llama_token> last_n_tokens;
static std::vector<llama_token> current_context_tokens;
static std::vector<llama_token> smartcontext;
static std::vector<std::string> stop_sequence;

bool llama_load_model(const load_model_inputs inputs, FileFormat in_file_format)
{
    printf("System Info: %s\n", llama_print_system_info());

    ctx_params = llama_context_default_params();

    n_threads = inputs.threads;
    n_batch = inputs.batch_size;
    modelname = inputs.model_filename;
    useSmartContext = inputs.use_smartcontext;
    blasbatchsize = inputs.blasbatchsize;

    ctx_params.n_ctx = inputs.max_context_length;
    ctx_params.n_parts = -1;//inputs.n_parts_overwrite;
    ctx_params.seed = -1;
    ctx_params.f16_kv = inputs.f16_kv;
    ctx_params.logits_all = false;
    ctx_params.use_mmap = inputs.use_mmap;
    ctx_params.use_mlock = false;

    file_format = in_file_format;
   
    ctx = llama_init_from_file(modelname.c_str(), ctx_params);
    
    if (ctx == NULL)
    {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, modelname.c_str());
        return false;
    }

    if (file_format < FileFormat::GGJT)
    {
        printf("\n---\nWarning: Your model has an INVALID or OUTDATED format (ver %d). Please reconvert it for better results!\n---\n", file_format);
    }

    //determine mem per token
    const std::vector<llama_token> tmp = {0, 1, 2, 3};
    llama_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);

    return true;
}

generation_outputs llama_generate(const generation_inputs inputs, generation_outputs &output)
{
    stop_sequence.clear();
    for(int x=0;x<stop_token_max;++x)
    {
        std::string stopper = inputs.stop_sequence[x];
        if(stopper!="")
        {
            stop_sequence.push_back(stopper);
        }
    }
    params.prompt = inputs.prompt;
    params.seed = inputs.seed;
    params.n_predict = inputs.max_length;
    params.top_k = inputs.top_k;
    params.top_p = inputs.top_p;
    params.temp = inputs.temperature;
    params.repeat_last_n = inputs.rep_pen_range;
    params.repeat_penalty = inputs.rep_pen;
    params.n_ctx = inputs.max_context_length;
    params.n_batch = n_batch;
    params.n_threads = n_threads;

    if (params.repeat_last_n < 1)
    {
        params.repeat_last_n = 1;
    }
    if (params.top_k < 1)
    {
        params.top_k = 300; //to disable top_k we actually need to increase this value to a very high number
    }
    if (params.seed <= 0)
    {
        params.seed = time(NULL);
    }

    params.prompt.insert(0, 1, ' ');

    // tokenize the prompt
    std::vector<llama_token> embd_inp;
    if (file_format == 1)
    {
        embd_inp = ::legacy_llama_tokenize(ctx, params.prompt, true);
    }
    else
    {
        embd_inp = ::llama_tokenize(ctx, params.prompt, true);
    }

    //truncate to front of the prompt if its too long
    int32_t nctx = params.n_ctx;
    if (embd_inp.size() + params.n_predict > nctx)
    {
        int offset = embd_inp.size() - nctx + params.n_predict;
        embd_inp = std::vector<llama_token>(embd_inp.begin() + offset, embd_inp.end());
    }

    //determine how much npast we have to rewind from the current state

    std::vector<llama_token> embd;

    int last_n_size = params.repeat_last_n;
    last_n_tokens.resize(last_n_size);

    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    n_past = 0;

    ContextFastForward(current_context_tokens, embd_inp, n_past, last_n_tokens, nctx, smartcontext, useSmartContext,false);

    //if using BLAS and prompt is big enough, switch to single thread and use a huge batch
    bool blasmode = (embd_inp.size() >= 32 && ggml_cpu_has_blas());
    int original_batch = params.n_batch;
    int original_threads = params.n_threads;
    if (blasmode)
    {
        params.n_batch = blasbatchsize; //received reports of 1024 and above crashing on some models
        params.n_threads = 1;
    }

    current_context_tokens.resize(n_past);

    int remaining_tokens = params.n_predict;
    int stopper_unused_tokens = 0;
    int input_consumed = 0;
    std::mt19937 rng(params.seed);
    std::string concat_output = "";

    bool startedsampling = false;

    timer_start();
    double time1 = 0, time2 = 0;
    unsigned int embd_inp_size = embd_inp.size();
    printf("\n");

    while (remaining_tokens > 0)
    {
        llama_token id = 0;
        // predict
        unsigned int embdsize = embd.size();
        if (embdsize > 0)
        {
            //print progress
            if (!startedsampling)
            {
                printf("\rProcessing Prompt%s (%d / %d tokens)", (blasmode ? " [BLAS]" : ""), input_consumed, embd_inp_size);
            }
            else
            {
                printf("\rGenerating (%d / %d tokens)", (1 + params.n_predict - remaining_tokens), params.n_predict);
            }
           
            if (llama_eval(ctx, embd.data(), embdsize, n_past, params.n_threads))
            {
                fprintf(stderr, "Failed to predict\n");
                snprintf(output.text, sizeof(output.text), "%s", "");
                output.status = 0;
                return output;
            }
        }

        n_past += embd.size();
        embd.clear();
        if ((int)embd_inp_size <= input_consumed)
        {
            // out of user input, sample next token
            const float top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp = params.temp;
            const float repeat_penalty = params.repeat_penalty;

            if (!startedsampling)
            {
                startedsampling = true;
                params.n_batch = original_batch;
                params.n_threads = original_threads;
                time1 = timer_check();
                timer_start();
                printf("\n");
            }

            {
                auto logits = llama_get_logits(ctx);
                // set the logit of the eos token (2) to zero to avoid sampling it
                logits[llama_token_eos()] = 0;
                //set logits of opening square bracket to zero.
                logits[518] = 0;
                logits[29961] = 0;

                id = llama_sample_top_p_top_k(ctx, last_n_tokens.data(), last_n_tokens.size(), top_k, top_p, temp, repeat_penalty);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
                current_context_tokens.push_back(id);
            }

            // add it to the context
            embd.push_back(id);

            // decrement remaining sampling budget
            --remaining_tokens;
            //printf("\nid:%d word:%s\n",id,llama_token_to_str(ctx, id));
            concat_output += llama_token_to_str(ctx, id);           
            for (const auto &matched : stop_sequence)
            {
                if (concat_output.find(matched) != std::string::npos)
                {
                    stopper_unused_tokens = remaining_tokens;
                    remaining_tokens = 0;
                    printf("\n(Stop sequence triggered: <%s>)",matched.c_str());
                    break;
                }
            }
        }
        else
        {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int)embd_inp_size > input_consumed)
            {
                embd.push_back(embd_inp[input_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[input_consumed]);
                current_context_tokens.push_back(embd_inp[input_consumed]);
                ++input_consumed;
                if ((int)embd.size() >= params.n_batch)
                {
                    break;
                }
            }
        }
    }
    time2 = timer_check();
    float pt1 = (time1*1000.0/(embd_inp_size==0?1:embd_inp_size));
    int realnpredict = params.n_predict-stopper_unused_tokens;
    float pt2 = (time2*1000.0/(realnpredict==0?1:realnpredict));
    printf("\nTime Taken - Processing:%.1fs (%.0fms/T), Generation:%.1fs (%.0fms/T), Total:%.1fs", time1, pt1, time2, pt2, (time1 + time2));
    fflush(stdout);
    output.status = 1;
    snprintf(output.text, sizeof(output.text), "%s", concat_output.c_str());
    return output;
}
