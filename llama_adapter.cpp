//This is Concedo's shitty adapter for adding python bindings for llama

//Considerations:
//Don't want to use pybind11 due to dependencies on MSVCC
//ZERO or MINIMAL changes as possible to main.cpp - do not move their function declarations here!
//Leave main.cpp UNTOUCHED, We want to be able to update the repo and pull any changes automatically.
//No dynamic memory allocation! Setup structs with FIXED (known) shapes and sizes for ALL output fields
//Python will ALWAYS provide the memory, we just write to it.

#include <time.h>
#include "./examples/main/main.cpp"
#include "ggml.h"
#include "model_adapter.h"

//return val: 0=fail, 1=(original ggml, alpaca), 2=(ggmf), 3=(ggjt)
static FileFormat file_format = FileFormat::FAIL;
static llama_context_params ctx_params;
static gpt_params params;
static int n_past = 0;
static int n_threads = 4;
static int n_batch = 8;
static std::string modelname;
static llama_context *ctx;
static std::vector<llama_token> last_n_tokens;
static std::vector<llama_token> current_context_tokens;

bool llama_load_model(const load_model_inputs inputs, FileFormat in_file_format)
{
    printf("System Info: %s\n", llama_print_system_info());

    ctx_params = llama_context_default_params();

    n_threads = inputs.threads;
    n_batch = inputs.batch_size;
    modelname = inputs.model_filename;

    ctx_params.n_ctx = inputs.max_context_length;
    ctx_params.n_parts = inputs.n_parts_overwrite;
    ctx_params.seed = -1;
    ctx_params.f16_kv = inputs.f16_kv;
    ctx_params.logits_all = false;

    file_format = in_file_format;

    if (file_format == FileFormat::GGML || file_format == FileFormat::GGHF)
    {
        ctx = legacy_llama_init_from_file(modelname.c_str(), ctx_params);
    }
    else
    {
        ctx = llama_init_from_file(modelname.c_str(), ctx_params);
    }

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

    //params.n_predict = std::min(params.n_predict, params.n_ctx - (int) embd_inp.size());
    //truncate to front of the prompt if its too long
    if (embd_inp.size() + params.n_predict > params.n_ctx)
    {
        int offset = embd_inp.size() - params.n_ctx + params.n_predict;
        embd_inp = std::vector<llama_token>(embd_inp.begin() + offset, embd_inp.end());
    }

    //determine how much npast we have to rewind from the current state

    std::vector<llama_token> embd;

    int last_n_size = params.repeat_last_n;
    last_n_tokens.resize(last_n_size);

    //display usage
    // std::string tst = " ";
    // char * tst2 = (char*)tst.c_str();
    // gpt_print_usage(1,&tst2,params);

    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    n_past = 0;

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

    //if using BLAS and prompt is big enough, switch to single thread and use a huge batch
    bool blasmode = (embd_inp.size() >= 32 && ggml_cpu_has_blas());
    int original_batch = params.n_batch;
    int original_threads = params.n_threads;
    if (blasmode)
    {
        params.n_batch = 512;
        params.n_threads = 1;
    }

    current_context_tokens.resize(n_past);

    int remaining_tokens = params.n_predict;
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
            //printf("\nnp:%d embd:%d txt:%s",n_past,embd.size(),llama_token_to_str(ctx, embd[0]));
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
        if ((int)embd_inp.size() <= input_consumed)
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
        }
        else
        {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int)embd_inp.size() > input_consumed)
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
    printf("\nTime Taken - Processing:%.1fs, Generation:%.1fs, Total:%.1fs", time1, time2, (time1 + time2));

    output.status = 1;
    snprintf(output.text, sizeof(output.text), "%s", concat_output.c_str());
    return output;
}
