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

    //return val: 0=fail, 1=(original ggml, alpaca), 2=(ggmf), 3=(ggjt)
    static FileFormat file_format = FileFormat::BADFORMAT;

    bool load_model(const load_model_inputs inputs)
    {
        std::string model = inputs.model_filename;
        file_format = check_file_format(model.c_str());

        //first digit is platform, second is devices
        int platform = inputs.clblast_info/10;
        int devices = inputs.clblast_info%10;
        std::string platformenv = "KCPP_CLBLAST_PLATFORM="+std::to_string(platform);
        std::string deviceenv = "KCPP_CLBLAST_DEVICES="+std::to_string(devices);
        putenv(platformenv.c_str());
        putenv(deviceenv.c_str());

        if(file_format==FileFormat::GPTJ_1 || file_format==FileFormat::GPTJ_2 || file_format==FileFormat::GPTJ_3)
        {
            printf("\n---\nIdentified as GPT-J model: (ver %d)\nAttempting to Load...\n---\n", file_format);
            ModelLoadResult lr = gpttype_load_model(inputs, file_format);
            if (lr == ModelLoadResult::RETRY_LOAD)
            {
                file_format = FileFormat::GPTJ_2;
                printf("\n---\nRetrying as GPT-J model: (ver %d)\nAttempting to Load...\n---\n", file_format);
                lr = gpttype_load_model(inputs, file_format);
            }
            if (lr == ModelLoadResult::RETRY_LOAD)
            {
                file_format = FileFormat::GPTJ_3;
                printf("\n---\nRetrying as GPT-J model: (ver %d)\nAttempting to Load...\n---\n", file_format);
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
        else if(file_format==FileFormat::GPT2_1||file_format==FileFormat::GPT2_2)
        {
            printf("\n---\nIdentified as GPT-2 model: (ver %d)\nAttempting to Load...\n---\n", file_format);
            ModelLoadResult lr = gpttype_load_model(inputs, file_format);
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
        else
        {
            printf("\n---\nIdentified as LLAMA model: (ver %d)\nAttempting to Load...\n---\n", file_format);   
            return llama_load_model(inputs, file_format);
        }
    }

    generation_outputs generate(const generation_inputs inputs, generation_outputs &output)
    {
        if (file_format == FileFormat::GPTJ_1 || file_format == FileFormat::GPTJ_2 || file_format==FileFormat::GPTJ_3 
        || file_format==FileFormat::GPT2_1 || file_format==FileFormat::GPT2_2 )
        {
            return gpttype_generate(inputs, output);
        }
        else
        {
            return llama_generate(inputs, output);
        }       
    }
}