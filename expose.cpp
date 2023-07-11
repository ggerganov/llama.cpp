//This is Concedo's shitty adapter for adding python bindings for llama

//Considerations:
//Don't want to use pybind11 due to dependencies on MSVCC
//ZERO or MINIMAL changes as possible to main.cpp - do not move their function declarations here!
//Leave main.cpp UNTOUCHED, We want to be able to update the repo and pull any changes automatically.
//No dynamic memory allocation! Setup structs with FIXED (known) shapes and sizes for ALL output fields
//Python will ALWAYS provide the memory, we just write to it.

#include <cassert>
#include <cstring>
#include <fstream>
#include <regex>
#include <iostream>
#include <iterator>
#include <queue>
#include <string>
#include <math.h>

#include "expose.h"
#include "model_adapter.cpp"

extern "C"
{

    std::string platformenv, deviceenv;

    //return val: 0=fail, 1=(original ggml, alpaca), 2=(ggmf), 3=(ggjt)
    static FileFormat file_format = FileFormat::BADFORMAT;

    bool load_model(const load_model_inputs inputs)
    {
        std::string model = inputs.model_filename;
        lora_filename = inputs.lora_filename;
        lora_base = inputs.lora_base;

        int forceversion = inputs.forceversion;

        if(forceversion==0)
        {
            file_format = check_file_format(model.c_str());
        }
        else
        {
            printf("\nWARNING: FILE FORMAT FORCED TO VER %d\nIf incorrect, loading may fail or crash.\n",forceversion);
            file_format = (FileFormat)forceversion;
        }

        //first digit is whether configured, second is platform, third is devices
        int cl_parseinfo = inputs.clblast_info;

        std::string usingclblast = "GGML_OPENCL_CONFIGURED="+std::to_string(cl_parseinfo>0?1:0);
        putenv((char*)usingclblast.c_str());

        cl_parseinfo = cl_parseinfo%100; //keep last 2 digits
        int platform = cl_parseinfo/10;
        int devices = cl_parseinfo%10;
        platformenv = "GGML_OPENCL_PLATFORM="+std::to_string(platform);
        deviceenv = "GGML_OPENCL_DEVICE="+std::to_string(devices);
        putenv((char*)platformenv.c_str());
        putenv((char*)deviceenv.c_str());
        executable_path = inputs.executable_path;

        if(file_format==FileFormat::GPTJ_1 || file_format==FileFormat::GPTJ_2 || file_format==FileFormat::GPTJ_3 || file_format==FileFormat::GPTJ_4  || file_format==FileFormat::GPTJ_5)
        {
            printf("\n---\nIdentified as GPT-J model: (ver %d)\nAttempting to Load...\n---\n", file_format);
            ModelLoadResult lr = gpttype_load_model(inputs, file_format);
            if (lr == ModelLoadResult::RETRY_LOAD)
            {
                if(file_format==FileFormat::GPTJ_1)
                {
                    //if we tried 1 first, then try 3 and lastly 2
                    //otherwise if we tried 3 first, then try 2
                    file_format = FileFormat::GPTJ_4;
                    printf("\n---\nRetrying as GPT-J model: (ver %d)\nAttempting to Load...\n---\n", file_format);
                    lr = gpttype_load_model(inputs, file_format);
                }

                if (lr == ModelLoadResult::RETRY_LOAD)
                {
                    file_format = FileFormat::GPTJ_3;
                    printf("\n---\nRetrying as GPT-J model: (ver %d)\nAttempting to Load...\n---\n", file_format);
                    lr = gpttype_load_model(inputs, file_format);
                }

                //lastly try format 2
                if (lr == ModelLoadResult::RETRY_LOAD)
                {
                    file_format = FileFormat::GPTJ_2;
                    printf("\n---\nRetrying as GPT-J model: (ver %d)\nAttempting to Load...\n---\n", file_format);
                    lr = gpttype_load_model(inputs, file_format);
                }
            }

            if (lr == ModelLoadResult::FAIL || lr == ModelLoadResult::RETRY_LOAD)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
        else if(file_format==FileFormat::GPT2_1||file_format==FileFormat::GPT2_2||file_format==FileFormat::GPT2_3||file_format==FileFormat::GPT2_4)
        {
            printf("\n---\nIdentified as GPT-2 model: (ver %d)\nAttempting to Load...\n---\n", file_format);
            ModelLoadResult lr = gpttype_load_model(inputs, file_format);
            if (lr == ModelLoadResult::RETRY_LOAD)
            {
                file_format = FileFormat::GPT2_3;
                printf("\n---\nRetrying as GPT-2 model: (ver %d)\nAttempting to Load...\n---\n", file_format);
                lr = gpttype_load_model(inputs, file_format);
            }
            if (lr == ModelLoadResult::RETRY_LOAD)
            {
                file_format = FileFormat::GPT2_2;
                printf("\n---\nRetrying as GPT-2 model: (ver %d)\nAttempting to Load...\n---\n", file_format);
                lr = gpttype_load_model(inputs, file_format);
            }
            if (lr == ModelLoadResult::FAIL || lr == ModelLoadResult::RETRY_LOAD)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
        else if(file_format==FileFormat::RWKV_1 || file_format==FileFormat::RWKV_2)
        {
            printf("\n---\nIdentified as RWKV model: (ver %d)\nAttempting to Load...\n---\n", file_format);
            ModelLoadResult lr = gpttype_load_model(inputs, file_format);
            if (lr == ModelLoadResult::FAIL || lr == ModelLoadResult::RETRY_LOAD)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
        else if(file_format==FileFormat::NEOX_1 || file_format==FileFormat::NEOX_2 || file_format==FileFormat::NEOX_3 || file_format==FileFormat::NEOX_4 || file_format==FileFormat::NEOX_5 || file_format==FileFormat::NEOX_6 || file_format==FileFormat::NEOX_7)
        {
            printf("\n---\nIdentified as GPT-NEO-X model: (ver %d)\nAttempting to Load...\n---\n", file_format);
            ModelLoadResult lr = gpttype_load_model(inputs, file_format);
            if (lr == ModelLoadResult::RETRY_LOAD)
            {
                if(file_format==FileFormat::NEOX_2)
                {
                    file_format = FileFormat::NEOX_3;
                    printf("\n---\nRetrying as GPT-NEO-X model: (ver %d)\nAttempting to Load...\n---\n", file_format);
                    lr = gpttype_load_model(inputs, file_format);
                }
                else
                {
                    file_format = FileFormat::NEOX_5;
                    printf("\n---\nRetrying as GPT-NEO-X model: (ver %d)\nAttempting to Load...\n---\n", file_format);
                    lr = gpttype_load_model(inputs, file_format);
                }
            }
            if (lr == ModelLoadResult::RETRY_LOAD)
            {
                file_format = FileFormat::NEOX_1;
                printf("\n---\nRetrying as GPT-NEO-X model: (ver %d)\nAttempting to Load...\n---\n", file_format);
                lr = gpttype_load_model(inputs, file_format);
            }
            if (lr == ModelLoadResult::FAIL || lr == ModelLoadResult::RETRY_LOAD)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
        else if(file_format==FileFormat::MPT_1)
        {
            printf("\n---\nIdentified as MPT model: (ver %d)\nAttempting to Load...\n---\n", file_format);
            ModelLoadResult lr = gpttype_load_model(inputs, file_format);
            if (lr == ModelLoadResult::FAIL || lr == ModelLoadResult::RETRY_LOAD)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
        else
        {
            printf("\n---\nIdentified as LLAMA model: (ver %d)\nAttempting to Load...\n---\n", file_format);
            ModelLoadResult lr = gpttype_load_model(inputs, file_format);
            if (lr == ModelLoadResult::FAIL || lr == ModelLoadResult::RETRY_LOAD)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
    }

    generation_outputs generate(const generation_inputs inputs, generation_outputs &output)
    {
        return gpttype_generate(inputs, output);
    }

    const char* new_token(int idx) {
        if (generated_tokens.size() <= idx || idx < 0) return nullptr;

        return generated_tokens[idx].c_str();
    }

    int get_stream_count() {
        return generated_tokens.size();
    }

    bool has_finished() {
        return generation_finished;
    }

    float get_prompt_eval_time() {
        return prompt_eval_time;
    }

    float get_prompt_process_time() {
        return prompt_process_time;
    }

    const char* get_pending_output() {
       return gpttype_get_pending_output().c_str();
    }

    bool abort_generate() {
        return gpttype_generate_abort();
    }
}
