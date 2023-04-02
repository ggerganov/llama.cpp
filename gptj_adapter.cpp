//This is Concedo's shitty adapter for adding python bindings for llama

//Considerations:
//Don't want to use pybind11 due to dependencies on MSVCC
//ZERO or MINIMAL changes as possible to main.cpp - do not move their function declarations here!
//Leave main.cpp UNTOUCHED, We want to be able to update the repo and pull any changes automatically.
//No dynamic memory allocation! Setup structs with FIXED (known) shapes and sizes for ALL output fields
//Python will ALWAYS provide the memory, we just write to it.

#include <time.h>
#include "model_adapter.h"
#include "otherarch/otherarch.h"

//concat source files into one file for compilation purposes
#include "otherarch/utils.cpp"
#include "otherarch/gptj_v1.cpp"
#include "otherarch/gptj_v2.cpp"

//return val: 0=fail, 1=(original ggml, alpaca), 2=(ggmf), 3=(ggjt)
static FileFormat file_format = FileFormat::FAIL;
static gpt_vocab vocab;
static gptj_model_v1 model_v1;
static gptj_model model_v2;
static gpt_params params;
static int n_past = 0;
static int n_threads = 4;
static int n_batch = 8;
static std::string modelname;
static std::vector<gpt_vocab::id> current_context_tokens;
static size_t mem_per_token = 0;
static std::vector<float> logits;

bool gptj_load_model(const load_model_inputs inputs, FileFormat in_file_format)
{
    
    ggml_time_init();

    file_format = in_file_format;
    n_threads = params.n_threads = inputs.threads;
    n_batch = params.n_batch = inputs.batch_size;
    modelname = params.model = inputs.model_filename;

    if (!legacy_gptj_model_load(params.model, model_v1, vocab)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
        return false;
    }

    if (file_format != FileFormat::GPTJ2)
    {
        printf("\n---\nWarning: Your model has an INVALID or OUTDATED format (ver %d). Please reconvert it for better results!\n---\n", file_format);
    }

    // determine the required inference memory per token:    
    legacy_gptj_eval(model_v1, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);    

    return true;
}



generation_outputs gptj_generate(const generation_inputs inputs, generation_outputs &output)
{
    params.prompt = inputs.prompt;
    params.seed = inputs.seed;
    params.n_predict = inputs.max_length;
    params.top_k = inputs.top_k;
    params.top_p = inputs.top_p;
    params.temp = inputs.temperature;
    params.n_batch = n_batch;
    params.n_threads = n_threads;

    if (params.top_k < 1)
    {
        params.top_k = 300; //to disable top_k we actually need to increase this value to a very high number
    }
    if (params.seed <= 0)
    {
        params.seed = time(NULL);
    }
    
    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);

    //truncate to front of the prompt if its too long
    if (embd_inp.size() + params.n_predict > model_v1.hparams.n_ctx)
    {
        int offset = embd_inp.size() - model_v1.hparams.n_ctx + params.n_predict;
        embd_inp = std::vector<llama_token>(embd_inp.begin() + offset, embd_inp.end());
    }

    //determine how much npast we have to rewind from the current state
    std::vector<gpt_vocab::id> embd;

    n_past = 0;

    //fast forward the past based on identical tokens, stop once a divergence is noted
    int embd_inp_len = embd_inp.size();
    for (int i = 0; i < current_context_tokens.size(); ++i)
    {
        if (current_context_tokens[i] == embd_inp[i])
        {
            n_past += 1;
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

    embd_inp.erase(embd_inp.begin(), embd_inp.begin() + n_past);

    //if using BLAS and prompt is big enough, switch to single thread and use a huge batch
    bool blasmode = false;// (embd_inp.size() >= 32 && ggml_cpu_has_blas());
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
    const int n_vocab = model_v1.hparams.n_vocab;

    printf("\n");

    while (remaining_tokens > 0)
    {
        gpt_vocab::id id = 0;
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
           
            if (!legacy_gptj_eval(model_v1, params.n_threads, n_past, embd, logits, mem_per_token))
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
                // set the logit of the eos token (2) to zero to avoid sampling it
                logits[50256] = 0;
                //set logits of opening square bracket to zero.
                
                
                id = gpt_sample_top_k_top_p(vocab, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);

                current_context_tokens.push_back(id);
            }

            // add it to the context
            embd.push_back(id);

            // decrement remaining sampling budget
            --remaining_tokens;
            
            for (auto id : embd) {
                concat_output += vocab.id_to_token[id].c_str();
            }
        }
        else
        {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int)embd_inp.size() > input_consumed)
            {
                embd.push_back(embd_inp[input_consumed]);               
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
