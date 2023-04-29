//This is Concedo's shitty adapter for adding python bindings for llama

//Considerations:
//Don't want to use pybind11 due to dependencies on MSVCC
//ZERO or MINIMAL changes as possible to main.cpp - do not move their function declarations here!
//Leave main.cpp UNTOUCHED, We want to be able to update the repo and pull any changes automatically.
//No dynamic memory allocation! Setup structs with FIXED (known) shapes and sizes for ALL output fields
//Python will ALWAYS provide the memory, we just write to it.

#include <time.h>
#include "model_adapter.h"
#include "otherarch.h"

//for easier compilation
#include "llamaextra.cpp"

//concat source files into one file for compilation purposes
#include "common-ggml.cpp"
#include "utils.cpp"
#include "gptj_v1.cpp"
#include "gptj_v2.cpp"
#include "gpt2_v1.cpp"
#include "gpt2_v2.cpp"
#include "rwkv.cpp"
#include "neox.cpp"


//return val: 0=fail, 1=(original ggml, alpaca), 2=(ggmf), 3=(ggjt)
static FileFormat file_format = FileFormat::BADFORMAT;

static gpt_vocab vocab;
static gptj_model_v1 gptj_ctx_v1;
static gptj_model gptj_ctx_v2;
static gpt2_v1_model gpt2_ctx_v1;
static gpt2_model gpt2_ctx_v2;
static stablelm_model neox_ctx;
static rwkv_context * rwkv_ctx_v1;
static llama_context_params llama_ctx_params;
static llama_context * llama_ctx_v1;

static gpt_params params;
static int n_past = 0;
static int n_threads = 4;
static int n_batch = 8;
static bool useSmartContext = false;
static bool unbanTokens = false;
static int blasbatchsize = 512;
static bool debugmode = false;
static std::string modelname;
static std::vector<gpt_vocab::id> last_n_tokens;
static std::vector<gpt_vocab::id> current_context_tokens;
static size_t mem_per_token = 0;
static std::vector<float> logits;
static std::vector<int> smartcontext;
static std::vector<std::string> stop_sequence;

inline bool IsNanCheck(float f)
{
    const unsigned int u = *(unsigned int*)&f;
    return (u&0x7F800000) == 0x7F800000 && (u&0x7FFFFF);    // Both NaN and qNan.
}

inline bool LogitsDuplicated(std::vector<float> & arr1, std::vector<float> & arr2)
{
    int compareQty = 5;
    if(arr1.size() < compareQty || arr2.size() < compareQty || arr1.size()!=arr2.size())
    {
        printf("\nError: Logit array sizes are bad!\n");
        return false;
    }
    for(int i=0;i<compareQty;++i)
    {
        if(arr1[i]!=arr2[i])
        {
            return false;
        }
    }
    return true;
}


llama_token sample_token(llama_token_data_array * candidates, std::mt19937 & rng) 
{
    const int64_t t_start_sample_us = ggml_time_us();
    llama_sample_softmax(nullptr, candidates);
    std::vector<float> probs;
    probs.reserve(candidates->size);
    for (size_t i = 0; i < candidates->size; ++i) {
        probs.push_back(candidates->data[i].p);
    }
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);
    llama_token result = candidates->data[idx].id;
    return result;
}

int SampleLogits(const float * logits, int n_ctx, int n_vocab, int rep_pen_range, float rep_pen, float top_k, float top_p, float typical_p, float tfs, float temp, std::mt19937 & rng)
{
    int id = 0;
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

    // Apply penalties
    auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), rep_pen_range), n_ctx);				
    llama_sample_repetition_penalty(nullptr, &candidates_p,
        last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
        last_n_repeat, rep_pen);
        			
    // llama_sample_frequency_and_presence_penalties(nullptr, &candidates_p,
    //     last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
    //     last_n_repeat, alpha_frequency, alpha_presence);
        
    if (temp <= 0) {
        // Greedy sampling
        id = llama_sample_token_greedy(nullptr, &candidates_p);
    } else {                    
        // Temperature sampling
        llama_sample_top_k(nullptr, &candidates_p, top_k);
        llama_sample_tail_free(nullptr, &candidates_p, tfs);
        llama_sample_typical(nullptr, &candidates_p, typical_p);
        llama_sample_top_p(nullptr, &candidates_p, top_p);
        llama_sample_temperature(nullptr, &candidates_p, temp);
        id = sample_token(&candidates_p, rng);                    
    }

    return id;
}

ModelLoadResult gpttype_load_model(const load_model_inputs inputs, FileFormat in_file_format)
{
    ggml_time_init();

    file_format = in_file_format;
    n_threads = params.n_threads = inputs.threads;
    n_batch = params.n_batch = inputs.batch_size;
    modelname = params.model = inputs.model_filename;
    useSmartContext = inputs.use_smartcontext;
    debugmode = inputs.debugmode;
    unbanTokens = inputs.unban_tokens;
    blasbatchsize = inputs.blasbatchsize;
    params.memory_f16 = inputs.f16_kv;
    params.n_ctx = inputs.max_context_length;
    neox_ctx.hparams.n_ctx = gptj_ctx_v1.hparams.n_ctx = gptj_ctx_v2.hparams.n_ctx = gpt2_ctx_v1.hparams.n_ctx = gpt2_ctx_v2.hparams.n_ctx = params.n_ctx;

    printf("System Info: %s\n", llama_print_system_info());

    if(file_format == FileFormat::GGML || file_format == FileFormat::GGHF || file_format == FileFormat::GGJT)
    {
        llama_ctx_params = llama_context_default_params();
        llama_ctx_params.n_ctx = inputs.max_context_length;
        llama_ctx_params.n_parts = -1;
        llama_ctx_params.seed = -1;
        llama_ctx_params.f16_kv = inputs.f16_kv;
        llama_ctx_params.logits_all = false;
        llama_ctx_params.use_mmap = inputs.use_mmap;
        llama_ctx_params.use_mlock = false;
        
        llama_ctx_v1 = llama_init_from_file(modelname.c_str(), llama_ctx_params);
        
        if (llama_ctx_v1 == NULL)
        {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, modelname.c_str());
            return ModelLoadResult::FAIL;
        }
        if (file_format < FileFormat::GGJT)
        {
            printf("\n---\nWarning: Your model has an INVALID or OUTDATED format (ver %d). Please reconvert it for better results!\n---\n", file_format);
        }

        if (lora_filename != "")
        {
            printf("\nAttempting to apply LORA adapter: %s\n", lora_filename.c_str());
     
            int err = llama_apply_lora_from_file(llama_ctx_v1,
                                                 lora_filename.c_str(),
                                                 NULL,
                                                 n_threads);
            if (err != 0)
            {
                fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
                return ModelLoadResult::FAIL;
            }
        }

        //determine mem per token
        const std::vector<int> tmp = {0, 1, 2, 3};
        llama_eval(llama_ctx_v1, tmp.data(), tmp.size(), 0, params.n_threads);
        return ModelLoadResult::SUCCESS;

    }
    else if (file_format == FileFormat::RWKV_1)
    {
        rwkv_ctx_v1 = rwkv_init_from_file(modelname.c_str(), n_threads);

        //setup buffers for rwkv state
        auto padding = 512u;
        auto statebufsiz = rwkv_get_state_buffer_element_count(rwkv_ctx_v1) * sizeof(float) + padding;
        auto logitbufsiz = rwkv_get_logits_buffer_element_count(rwkv_ctx_v1) * sizeof(float) + padding;

        printf("\nRWKV Init: State Buffer:%u, Logit Buffer:%u\n", statebufsiz, logitbufsiz);
        rwkv_ctx_v1->state_out = (float *)malloc(statebufsiz);
        rwkv_ctx_v1->logits_out = (float *)malloc(logitbufsiz);
        rwkv_ctx_v1->state_in = nullptr;
        n_batch = 1;

        std::string word;
        read_rwkv_vocab();
        int vocabsiz = rwkv_vocab.size();
        for (int i = 0; i < vocabsiz; i++) {
            uint32_t len;
            word = rwkv_vocab[i];
            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
        printf("\nRWKV Vocab: %u\n",vocabsiz);

        bool testeval = rwkv_eval(rwkv_ctx_v1, 0, rwkv_ctx_v1->state_in, rwkv_ctx_v1->state_out, rwkv_ctx_v1->logits_out);
        if(!testeval)
        {
            printf("\nError: RWKV Init Eval Failed!\n");
        }
        logits.resize(vocabsiz);
        memcpy(logits.data(), rwkv_ctx_v1->logits_out, sizeof(float)*vocabsiz);

        if (rwkv_ctx_v1 == NULL)
        {
            return ModelLoadResult::FAIL;
        }
        return ModelLoadResult::SUCCESS;
    }
    else if (file_format == FileFormat::GPT2_1)
    {
        ModelLoadResult res = legacy_gpt2_model_load(params.model, gpt2_ctx_v1, vocab, file_format);
        if(res==ModelLoadResult::FAIL)
        {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return res;
        }
        else if(res==ModelLoadResult::RETRY_LOAD)
        {
            printf("\nTensor Transposition Detected! Retrying GPT-2 model loading...");
            return res;
        }
         // determine the required inference memory per token:    
        legacy_gpt2_eval(gpt2_ctx_v1, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token, file_format);    
        return ModelLoadResult::SUCCESS;
    }
    else if (file_format == FileFormat::GPT2_2)
    {
        ModelLoadResult res = gpt2_model_load(params.model, gpt2_ctx_v2, vocab, file_format);
        if(res==ModelLoadResult::FAIL)
        {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return res;
        } 
        else if(res==ModelLoadResult::RETRY_LOAD)
        {
            printf("\nTensor Transposition Detected! Retrying GPT-2 model loading...");
            return res;
        }
         // determine the required inference memory per token:    
        gpt2_eval(gpt2_ctx_v2, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token, file_format);    
        return ModelLoadResult::SUCCESS;
    }
    else if (file_format == FileFormat::GPTJ_1 || file_format == FileFormat::GPTJ_2)
    {
        ModelLoadResult res = legacy_gptj_model_load(params.model, gptj_ctx_v1, vocab, file_format);
        if(res==ModelLoadResult::FAIL)
        {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return res;
        }
        else if(res==ModelLoadResult::RETRY_LOAD)
        {
            printf("\nTensor Transposition Detected! Retrying GPT-J model loading...");
            return res;
        }
         // determine the required inference memory per token:    
        legacy_gptj_eval(gptj_ctx_v1, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token, file_format);    
        
        //if the logits are NAN or duplicated, it means the model is incompatible
        if(logits.size()>0 && IsNanCheck(logits[0]))
        {
            printf("\nBad Logits detected! Retrying GPT-J model loading...");
            ggml_v1_free(gptj_ctx_v1.ctx);
            return ModelLoadResult::RETRY_LOAD;
        }

        return ModelLoadResult::SUCCESS;
    }
    else if(file_format==FileFormat::NEOX_1 || file_format==FileFormat::NEOX_2)
    {
        ModelLoadResult res = stablelm_model_load(params.model, neox_ctx, vocab, file_format);       
        if(res==ModelLoadResult::FAIL)
        {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return res;
        }
        else if(res==ModelLoadResult::RETRY_LOAD)
        {
            printf("\nIncorrect Tensor Size Detected! Retrying GPT-NeoX model loading...");
            return res;
        }
         // determine the required inference memory per token:    
        stablelm_eval(neox_ctx, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);    
        return ModelLoadResult::SUCCESS;
    }
    else
    {
        ModelLoadResult loadresult = gptj_model_load(params.model, gptj_ctx_v2, vocab);
        if (loadresult == ModelLoadResult::FAIL)
        {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return loadresult;
        }
        else if (loadresult == ModelLoadResult::RETRY_LOAD)
        {
            printf("\nTensor Transposition Detected! Retrying GPT-J model loading...");
            return loadresult;
        }

        // determine the required inference memory per token:    
        gptj_eval(gptj_ctx_v2, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);
      
        //if the logits are NAN or duplicated, it means the model is incompatible
        std::vector<float> oldlogits(logits);

        //this is another hack because they change the library - we run the eval through the model
        //twice and compare logits. if they give the same logits for different inputs, model is broken
        gptj_eval(gptj_ctx_v2, params.n_threads, 0, {4, 5, 6, 7}, logits, mem_per_token);
                
        if(logits.size()>0 && (IsNanCheck(logits[0]) || LogitsDuplicated(oldlogits,logits)))
        {
            printf("\nBad Logits detected! Retrying GPT-J model loading...");
            ggml_free(gptj_ctx_v2.ctx);
            return ModelLoadResult::RETRY_LOAD;
        }

        return ModelLoadResult::SUCCESS;
    }
   
}



generation_outputs gpttype_generate(const generation_inputs inputs, generation_outputs &output)
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
    params.typical_p = inputs.typical_p;
    params.tfs_z = inputs.tfs;
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

    // tokenize the prompt
    std::vector<int> embd_inp;

    if (file_format == FileFormat::GGML || file_format == FileFormat::GGHF || file_format == FileFormat::GGJT)
    {
        params.prompt.insert(0, 1, ' ');
        if (file_format == FileFormat::GGML)
        {
            embd_inp = ::legacy_llama_tokenize(llama_ctx_v1, params.prompt, true);
        }
        else
        {
            embd_inp = ::llama_tokenize(llama_ctx_v1, params.prompt, true);
        }
    }
    else
    {
        // tokenize the prompt
        embd_inp = ::gpt_tokenize(vocab, params.prompt);
    }

    //truncate to front of the prompt if its too long
    int32_t nctx = params.n_ctx;

    if (embd_inp.size() + params.n_predict > nctx)
    {
        int offset = embd_inp.size() - nctx + params.n_predict;
        embd_inp = std::vector<int>(embd_inp.begin() + offset, embd_inp.end());
    }

    //determine how much npast we have to rewind from the current state
    std::vector<gpt_vocab::id> embd;

    int last_n_size = params.repeat_last_n;
    last_n_tokens.resize(last_n_size);

    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    n_past = 0;

    if (file_format == FileFormat::RWKV_1)
    {
        ContextFastForward(current_context_tokens, embd_inp, n_past, last_n_tokens, nctx, smartcontext, false, true);
    }
    else
    {
        ContextFastForward(current_context_tokens, embd_inp, n_past, last_n_tokens, nctx, smartcontext, useSmartContext, false);
    }

    //if using BLAS and prompt is big enough, switch to single thread and use a huge batch
    bool approved_format = (file_format == FileFormat::GGML ||
                            file_format == FileFormat::GGHF || 
                            file_format == FileFormat::GGJT ||
                            file_format == FileFormat::GPT2_2 || 
                            file_format == FileFormat::GPTJ_3 ||
                            file_format == FileFormat::NEOX_1 || 
                            file_format == FileFormat::NEOX_2);
    bool blasmode = (approved_format && embd_inp.size() >= 32 && ggml_cpu_has_blas());
    // bool blasmode = false;
    int original_batch = params.n_batch;
    int original_threads = params.n_threads;
    if (blasmode)
    {
        //for non llama, limit to 256
        int bbs = blasbatchsize;
        if (file_format != FileFormat::GGML && file_format != FileFormat::GGHF && file_format != FileFormat::GGJT)
        {
            bbs = (blasbatchsize > 256 ? 256 : blasbatchsize);
        }

        params.n_batch = bbs; //received reports of 1024 and above crashing on some models
        if(!ggml_cpu_has_gpublas())
        {
            params.n_threads = 1; //do not limit here anymore.
        }
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
    int32_t n_vocab = 0;

    if (file_format == FileFormat::GGML || file_format == FileFormat::GGHF || file_format == FileFormat::GGJT)
    {
        n_vocab = llama_n_vocab(llama_ctx_v1);
    }
    else if (file_format == FileFormat::GPTJ_1 || file_format == FileFormat::GPTJ_2)
    {
        n_vocab = gptj_ctx_v1.hparams.n_vocab;
    }    
    else if(file_format == FileFormat::GPTJ_3)
    {
        n_vocab = gptj_ctx_v2.hparams.n_vocab;
    }
    else if(file_format == FileFormat::GPT2_1)
    {
        n_vocab = gpt2_ctx_v1.hparams.n_vocab;
    }
    else if(file_format == FileFormat::GPT2_2)
    {
        n_vocab = gpt2_ctx_v2.hparams.n_vocab;
    }
    else if(file_format == FileFormat::NEOX_1 || file_format == FileFormat::NEOX_2)
    {
        n_vocab = neox_ctx.hparams.n_vocab;
    }
    else if(file_format == FileFormat::RWKV_1)
    {
        n_vocab = vocab.id_to_token.size(); //handled seperately
        if(n_past==0)
        {
            rwkv_ctx_v1->state_in = nullptr;
        }
        else
        {
            rwkv_ctx_v1->state_in = rwkv_ctx_v1->state_out;
            //if it's empty, push in the final previous token
            if(embd_inp.size()==0 && current_context_tokens.size()>0)
            {
                embd_inp.push_back(current_context_tokens[current_context_tokens.size()-1]);
            }
        }
    }
    else
    {
        printf("Bad format!");
    }

    printf("\n");

    if(debugmode)
    {
        printf("\n[Debug: Dump Input Tokens]\n");
        if (file_format == FileFormat::GGML || file_format == FileFormat::GGHF || file_format == FileFormat::GGJT)
        {
            for (auto id : embd_inp)
            {
                printf("'%s', ",llama_token_to_str(llama_ctx_v1, id));
            }
        }
        else
        {
            for (auto id : embd_inp)
            {
                printf("'%s', ",vocab.id_to_token[id].c_str());
            }
        }
        printf("\n");
    }
    
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
                printf("\rProcessing Prompt%s (%d / %d tokens)", (blasmode ? " [BLAS]" : ""), input_consumed, embd_inp.size());
            }
            else
            {
                printf("\rGenerating (%d / %d tokens)", (1 + params.n_predict - remaining_tokens), params.n_predict);
            }
            fflush(stdout);

            bool evalres = false;

            if (file_format == FileFormat::GGML || file_format == FileFormat::GGHF || file_format == FileFormat::GGJT)
            {
                evalres = (llama_eval(llama_ctx_v1, embd.data(), embdsize, n_past, params.n_threads)==0);
            }
            else if(file_format==FileFormat::RWKV_1)
            {
                evalres = rwkv_eval(rwkv_ctx_v1, embd[0], rwkv_ctx_v1->state_in, rwkv_ctx_v1->state_out, rwkv_ctx_v1->logits_out);
                memcpy(logits.data(), rwkv_ctx_v1->logits_out, sizeof(float)*rwkv_vocab.size());
                rwkv_ctx_v1->state_in = rwkv_ctx_v1->state_out;
            }
            else if(file_format==FileFormat::GPT2_1)
            {
                evalres = legacy_gpt2_eval(gpt2_ctx_v1, params.n_threads, n_past, embd, logits, mem_per_token, file_format);
            }
            else if(file_format==FileFormat::GPT2_2)
            {
                evalres = gpt2_eval(gpt2_ctx_v2, params.n_threads, n_past, embd, logits, mem_per_token, file_format);
            }
            else if(file_format==FileFormat::NEOX_1 || file_format == FileFormat::NEOX_2)
            {
                evalres = stablelm_eval(neox_ctx, params.n_threads, n_past, embd, logits, mem_per_token);
            }
            else if(file_format==FileFormat::GPTJ_1 || file_format==FileFormat::GPTJ_2)
            {
                evalres = legacy_gptj_eval(gptj_ctx_v1, params.n_threads, n_past, embd, logits, mem_per_token, file_format);
            }
            else
            {
                evalres = gptj_eval(gptj_ctx_v2, params.n_threads, n_past, embd, logits, mem_per_token);
            }
            if (!evalres)
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
            const float typical_p = params.typical_p;
            const float tfs_z = params.tfs_z;

            if (!startedsampling)
            {
                startedsampling = true;
                params.n_batch = original_batch;
                params.n_threads = original_threads;
                time1 = timer_check();
                timer_start();
                printf("\n");
            }

            if(file_format == FileFormat::GGML || file_format == FileFormat::GGHF || file_format == FileFormat::GGJT)
            {
                auto logits = llama_get_logits(llama_ctx_v1);

                if (!unbanTokens)
                {
                    // set the logit of the eos token (2) to zero to avoid sampling it
                    logits[llama_token_eos()] = 0;
                    //set logits of opening square bracket to zero.
                    logits[518] = 0;
                    logits[29961] = 0;
                }

                id = SampleLogits(logits, nctx, n_vocab, last_n_size, repeat_penalty, top_k, top_p, typical_p, tfs_z, temp, rng);

            }
            else
            {
                if (!unbanTokens)
                {
                    // set the logit of the eos token (2) to zero to avoid sampling it
                    if ((file_format == FileFormat::GPT2_1 ||
                         file_format == FileFormat::GPT2_2 ||
                         file_format == FileFormat::GPTJ_1 ||
                         file_format == FileFormat::GPTJ_2 ||
                         file_format == FileFormat::GPTJ_3) &&
                        logits.size() > 50256)
                    {
                        logits[50256] = (logits[50256] < 0 ? logits[50256] : 0);
                    }
                    //gpt2 uses negative logits, so we cant zero it
                }

                id = SampleLogits(logits.data(), nctx, n_vocab, last_n_size, repeat_penalty, top_k, top_p, typical_p, tfs_z, temp, rng);
            }

            last_n_tokens.erase(last_n_tokens.begin());
            last_n_tokens.push_back(id);
            current_context_tokens.push_back(id);

            // add it to the context
            embd.push_back(id);

            // decrement remaining sampling budget
            --remaining_tokens;

            if (file_format == FileFormat::GGML || file_format == FileFormat::GGHF || file_format == FileFormat::GGJT)
            {
                concat_output += llama_token_to_str(llama_ctx_v1, id);
            }
            else
            {
                for (auto id : embd)
                {
                    concat_output += vocab.id_to_token[id].c_str();
                }
            }
            for (const auto &matched : stop_sequence)
            {
                if (concat_output.find(matched) != std::string::npos)
                {
                    stopper_unused_tokens = remaining_tokens;
                    remaining_tokens = 0;
                    printf("\n(Stop sequence triggered: <%s>)", matched.c_str());
                    break;
                }
            }
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
    float pt1 = (time1*1000.0/(embd_inp.size()==0?1:embd_inp.size()));
    int realnpredict = params.n_predict-stopper_unused_tokens;
    float pt2 = (time2*1000.0/(realnpredict==0?1:realnpredict));
    printf("\nTime Taken - Processing:%.1fs (%.0fms/T), Generation:%.1fs (%.0fms/T), Total:%.1fs", time1, pt1, time2, pt2, (time1 + time2));
    fflush(stdout);
    output.status = 1;
    snprintf(output.text, sizeof(output.text), "%s", concat_output.c_str());

    return output;
}
