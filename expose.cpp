//This is Concedo's shitty adapter for adding python bindings for llama

//Considerations:
//Don't want to use pybind11 due to dependencies on MSVCC
//ZERO or MINIMAL changes as possible to main.cpp - do not move their function declarations here!
//Leave main.cpp UNTOUCHED, We want to be able to update the repo and pull any changes automatically.
//No dynamic memory allocation! Setup structs with FIXED (known) shapes and sizes for ALL output fields
//Python will ALWAYS provide the memory, we just write to it.

#include "model_adapter.h"
#include "expose.h"
#include "extra.h"

extern "C"
{

    //return val: 0=fail, 1=(original ggml, alpaca), 2=(ggmf), 3=(ggjt)
    static FileFormat file_format = FAIL;

    bool load_model(const load_model_inputs inputs)
    {
        std::string model = inputs.model_filename;
        file_format = check_file_format(model.c_str());
        printf("\nIdentified as LLAMA model: (ver %d)\n", file_format);
        
        return llama_load_model(inputs, file_format);
    }

    generation_outputs generate(const generation_inputs inputs, generation_outputs &output)
    {
        return llama_generate(inputs, output);
    }
}