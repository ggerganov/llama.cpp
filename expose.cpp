//This is Concedo's shitty adapter for adding python bindings for llama

//Considerations:
//Don't want to use pybind11 due to dependencies on MSVCC
//ZERO or MINIMAL changes as possible to main.cpp - do not move their function declarations here!
//Leave main.cpp UNTOUCHED, We want to be able to update the repo and pull any changes automatically.
//No dynamic memory allocation! Setup structs with FIXED (known) shapes and sizes for ALL output fields
//Python will ALWAYS provide the memory, we just write to it.

#include "main.cpp"

extern "C" {

    struct load_model_inputs
    {
        const int threads;
        const int max_context_length;
        const int batch_size;
        const char * model_filename;
    };
    struct generation_inputs
    {
        const int seed;                
        const char * prompt;
        const int max_length;
        const float temperature;
        const int top_k;
        const float top_p;
        const float rep_pen;
        const int rep_pen_range;    
    };
    struct generation_outputs
    {
        int status;
        char text[16384]; //16kb should be enough for any response
    };

    gpt_params api_params;
    gpt_vocab api_vocab;
    llama_model api_model;    
    int api_n_past = 0;
    std::vector<float> api_logits;

    bool load_model(const load_model_inputs inputs)
    {
        api_params.n_threads = inputs.threads;
        api_params.n_ctx = inputs.max_context_length;
        api_params.n_batch = inputs.batch_size;
        api_params.model = inputs.model_filename;

         if (!llama_model_load(api_params.model, api_model, api_vocab, api_params.n_ctx)) {  
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, api_params.model.c_str());
            return false;
        }

        return true;
    }

    generation_outputs generate(const generation_inputs inputs, generation_outputs output)
    {
        api_params.prompt = inputs.prompt;
        api_params.seed = inputs.seed;
        api_params.n_predict = inputs.max_length;
        api_params.top_k = inputs.top_k;
        api_params.top_p = inputs.top_p;
        api_params.temp = inputs.temperature;
        api_params.repeat_last_n = inputs.rep_pen_range;
        api_params.repeat_penalty = inputs.rep_pen;
      
        if (api_params.seed < 0)
        {
            api_params.seed = time(NULL);
        }
        
        api_params.prompt.insert(0, 1, ' ');
        // tokenize the prompt
        std::vector<gpt_vocab::id> embd_inp = ::llama_tokenize(api_vocab, api_params.prompt, true);
        api_params.n_predict = std::min(api_params.n_predict, api_model.hparams.n_ctx - (int)embd_inp.size());
        std::vector<gpt_vocab::id> embd;
        size_t mem_per_token = 0;
        llama_eval(api_model, api_params.n_threads, 0, {0, 1, 2, 3}, api_logits, mem_per_token);

        int last_n_size = api_params.repeat_last_n;
        std::vector<gpt_vocab::id> last_n_tokens(last_n_size);
        std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
        int remaining_tokens = api_params.n_predict;
        int input_consumed = 0;
        std::mt19937 api_rng(api_params.seed);

        std::string concat_output = "";
       
        while (remaining_tokens > 0)
        {
             gpt_vocab::id id = 0;
            // predict
            if (embd.size() > 0)
            {

                if (!llama_eval(api_model, api_params.n_threads, api_n_past, embd, api_logits, mem_per_token))
                {
                    fprintf(stderr, "Failed to predict\n");
                    _snprintf_s(output.text,sizeof(output.text),_TRUNCATE,"%s","");
                    output.status = 0;
                    return output;
                }
            }

            api_n_past += embd.size();
            embd.clear();

            if (embd_inp.size() <= input_consumed)
            {
                // out of user input, sample next token
                const float top_k = api_params.top_k;
                const float top_p = api_params.top_p;
                const float temp = api_params.temp;
                const float repeat_penalty = api_params.repeat_penalty;
                const int n_vocab = api_model.hparams.n_vocab;
                
                {
                    // set the logit of the eos token (2) to zero to avoid sampling it
                    api_logits[api_logits.size() - n_vocab + 2] = 0;
                    //set logits of opening square bracket to zero.
                    api_logits[api_logits.size() - n_vocab + 518] = 0;
                    api_logits[api_logits.size() - n_vocab + 29961] = 0;


                    id = llama_sample_top_p_top_k(api_vocab, api_logits.data() + (api_logits.size() - n_vocab), last_n_tokens, repeat_penalty, top_k, top_p, temp, api_rng);

                    last_n_tokens.erase(last_n_tokens.begin());
                    last_n_tokens.push_back(id);
                }

                // add it to the context
                embd.push_back(id);

                // decrement remaining sampling budget
                --remaining_tokens;

                concat_output += api_vocab.id_to_token[id].c_str();
            }
            else
            {
                // some user input remains from prompt or interaction, forward it to processing
                while (embd_inp.size() > input_consumed)
                {
                    embd.push_back(embd_inp[input_consumed]);
                    last_n_tokens.erase(last_n_tokens.begin());
                    last_n_tokens.push_back(embd_inp[input_consumed]);
                    ++input_consumed;
                    if (embd.size() > api_params.n_batch)
                    {
                        break;
                    }
                }
            }
            
        }

        printf("output: %s",concat_output.c_str());
        output.status = 1;
        _snprintf_s(output.text,sizeof(output.text),_TRUNCATE,"%s",concat_output.c_str());
        return output;
    }
}