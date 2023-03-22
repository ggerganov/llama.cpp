//This is Concedo's shitty adapter for adding python bindings for llama

//Considerations:
//Don't want to use pybind11 due to dependencies on MSVCC
//ZERO or MINIMAL changes as possible to main.cpp - do not move their function declarations here!
//Leave main.cpp UNTOUCHED, We want to be able to update the repo and pull any changes automatically.
//No dynamic memory allocation! Setup structs with FIXED (known) shapes and sizes for ALL output fields
//Python will ALWAYS provide the memory, we just write to it.

#include "main.cpp"
#include "extra.h"

extern "C" {

    struct load_model_inputs
    {
        const int threads;
        const int max_context_length;
        const int batch_size;
        const char * model_filename;
        const int n_parts_overwrite = -1;
    };
    struct generation_inputs
    {
        const int seed;                
        const char * prompt;
        const int max_context_length;
        const int max_length;
        const float temperature;
        const int top_k;
        const float top_p;
        const float rep_pen;
        const int rep_pen_range;
        const bool reset_state = true; //determines if we can continue off the previous prompt state
    };
    struct generation_outputs
    {
        int status = -1;
        char text[16384]; //16kb should be enough for any response
    };

    bool legacy_format = false;
    llama_context_params ctx_params;
    gpt_params params;
    int n_past = 0;
    llama_token old_embd_id = -1;
    int n_threads = 4;
    int n_batch = 8;
    std::string model;
    llama_context * ctx;
    std::vector<llama_token> last_n_tokens;

    bool load_model(const load_model_inputs inputs)
    {
        ctx_params = llama_context_default_params();

        n_threads = inputs.threads;       
        n_batch = inputs.batch_size;
        model = inputs.model_filename;        

        ctx_params.n_ctx      = inputs.max_context_length;
        ctx_params.n_parts    = inputs.n_parts_overwrite;
        ctx_params.seed       = -1;
        ctx_params.f16_kv     = true;
        ctx_params.logits_all = false;

        ctx = llama_init_from_file(model.c_str(), ctx_params);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, model.c_str());
            return false;
        }

        //return val: 0=fail, 1=newformat, 2=legacy
        int fileformat = check_file_format(model.c_str());        
        
        legacy_format = (fileformat==1?true:false);
        if(legacy_format)
        {
            printf("\n---\nWarning: Your model is using an OUTDATED format. Please reconvert it for better results!\n");
        }

        return true;
    }

    generation_outputs generate(const generation_inputs inputs, generation_outputs & output)
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

        bool reset_state = inputs.reset_state;
        if(n_past==0)
        {
            reset_state = true;
        }
      
        if(params.repeat_last_n<1)
        {
            params.repeat_last_n = 1;
        }
        if(params.top_k<1)
        {
            params.top_k = 300; //to disable top_k we actually need to increase this value to a very high number
        }
        if (params.seed <= 0)
        {
            params.seed = time(NULL);
        }

		if(reset_state)
		{
			params.prompt.insert(0, 1, ' ');
		}

	    // tokenize the prompt
 		std::vector<llama_token> embd_inp;
		if(legacy_format)
        {
            embd_inp = ::legacy_llama_tokenize(ctx, params.prompt, true);
        }else{
            embd_inp = ::llama_tokenize(ctx, params.prompt, true);
        }

 		//params.n_predict = std::min(params.n_predict, params.n_ctx - (int) embd_inp.size());
        //truncate to front of the prompt if its too long
        if (embd_inp.size() + params.n_predict > params.n_ctx) {
            int offset = embd_inp.size() - params.n_ctx + params.n_predict;
            embd_inp = std::vector<llama_token>(embd_inp.begin() + offset, embd_inp.end());
        }	   
   		std::vector<llama_token> embd;

		int last_n_size = params.repeat_last_n;
    	last_n_tokens.resize(last_n_size);

        //display usage
        // std::string tst = " ";
        // char * tst2 = (char*)tst.c_str();
        // gpt_print_usage(1,&tst2,params);

		if(reset_state)
        {
			const std::vector<llama_token> tmp = { 0, 1, 2, 3 };
	        llama_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);            
            std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
            n_past = 0;
        }
        else
        {
            //strip out the reset token (1) at the start of the embedding
            if(embd_inp.size()>0)
            {
                embd_inp.erase(embd_inp.begin());
            }
            if(old_embd_id!=-1)
            {
                embd.push_back(old_embd_id);
            }
        }
		
 		int remaining_tokens = params.n_predict;
		int input_consumed = 0;
    	std::mt19937 rng(params.seed);   
		std::string concat_output = "";  
    	
		bool startedsampling = false;
        printf("\nProcessing Prompt: ");

		while (remaining_tokens > 0) 
		{
			llama_token id = 0;
	        // predict
	        if (embd.size() > 0) 
			{
				printf("|");
                // for (auto i: embd) {                    
                //     std::cout << i << ',';
                // }
                // printf("\nnp:%d embd:%d",n_past,embd.size());
	            if (llama_eval(ctx, embd.data(), embd.size(), n_past, params.n_threads)) 
				{
	                fprintf(stderr, "Failed to predict\n");
                    snprintf(output.text, sizeof(output.text), "%s", "");
                    output.status = 0;
                    return output;
	            }
	        }

        	n_past += embd.size();
       		embd.clear();
        	if ((int) embd_inp.size() <= input_consumed) 
			{
	            // out of user input, sample next token
	            const float top_k          = params.top_k;
	            const float top_p          = params.top_p;
	            const float temp           = params.temp;
	            const float repeat_penalty = params.repeat_penalty;

            	if(!startedsampling)
                {
                    startedsampling = true;
                    printf("\nGenerating: ");
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
	            }

	            // add it to the context
				old_embd_id = id;
	            embd.push_back(id);


	            // decrement remaining sampling budget
	            --remaining_tokens;
                //printf("\nid:%d word:%s\n",id,llama_token_to_str(ctx, id));
				concat_output += llama_token_to_str(ctx, id);
        	} 
			else 
			{
	            // some user input remains from prompt or interaction, forward it to processing
	            while ((int) embd_inp.size() > input_consumed) 
				{
					old_embd_id = embd_inp[input_consumed];
	                embd.push_back(embd_inp[input_consumed]);
	                last_n_tokens.erase(last_n_tokens.begin());
	                last_n_tokens.push_back(embd_inp[input_consumed]);
	                ++input_consumed;
	                if ((int) embd.size() >= params.n_batch) 
					{
	                    break;
	                }
            	}
        	}

		}
       		
		output.status = 1;
        snprintf(output.text, sizeof(output.text), "%s", concat_output.c_str());
        return output;

    }
}