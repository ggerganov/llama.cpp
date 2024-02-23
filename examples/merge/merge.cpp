#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <cmath>
#include <algorithm>


int32_t merge(
    const std::string & fname_inp1,
    const std::vector<float> scale1,
    const std::string & fname_inp2,
    const std::vector<float> scale2,
    const int n_layers,
    const std::string & fname_out) {
#if defined(__linux__) || defined(_WIN32)
    constexpr bool use_mmap = true;
#else
    constexpr bool use_mmap = false;
#endif

    llama_model_loader ml(fname_inp1, use_mmap, NULL);
    ml.init_mapping(false); // no prefetching?

    llama_model model;
    llm_load_arch(ml, model);
    llm_load_hparams(ml, model);

    struct gguf_context * ctx_out = gguf_init_empty();
    // copy the KV pairs from the input file
    gguf_set_kv(ctx_out, ml.ctx_gguf);
    
    // populate the original tensors so we get an initial meta data
    for (int i = 0; i < ml.n_tensors; ++i) {
        struct ggml_tensor * meta = ml.get_tensor_meta(i);
        gguf_add_tensor(ctx_out, meta);
    }

    std::ofstream fout(fname_out, std::ios::binary);
    fout.exceptions(std::ofstream::failbit); // fail fast on write errors

    const size_t meta_size = gguf_get_meta_size(ctx_out);

    LLAMA_LOG_INFO("%s: meta size = %zu bytes\n", __func__, meta_size);

    // placeholder for the meta data
    ::zeros(fout, meta_size);

    std::vector<no_init<uint8_t>> read_data;

    for (int i = 0; i < ml.n_tensors; ++i) {
        struct ggml_tensor * tensor = ml.get_tensor_meta(i);

        const std::string name = ggml_get_name(tensor);

        if (!ml.use_mmap) {
            if (read_data.size() < ggml_nbytes(tensor)) {
                read_data.resize(ggml_nbytes(tensor));
            }
            tensor->data = read_data.data();
        }
        ml.load_data_for(tensor);

        size_t new_size = ggml_nbytes(tensor);
        void * new_data = tensor->data;

        LLAMA_LOG_INFO("[%4d/%4d] %36s - [%s], type = %6s, ",
               ++idx, ml.n_tensors,
               ggml_get_name(tensor),
               llama_format_tensor_shape(tensor).c_str(),
               ggml_type_name(tensor->type));

        // update the gguf meta data as we go
        gguf_set_tensor_type(ctx_out, name.c_str(), new_type);
        gguf_set_tensor_data(ctx_out, name.c_str(), new_data, new_size);

        // write tensor data + padding
        fout.write((const char *) new_data, new_size);
        zeros(fout, GGML_PAD(new_size, GGUF_DEFAULT_ALIGNMENT) - new_size);
    }

    // go back to beginning of file and write the updated meta data
    {
        fout.seekp(0);
        std::vector<uint8_t> data(gguf_get_meta_size(ctx_out));
        gguf_get_meta_data(ctx_out, data.data());
        fout.write((const char *) data.data(), data.size());
    }

    fout.close();

    gguf_free(ctx_out);
}


// usage:
//  ./merge ./path/model_1 LAYERS_1 ./path/model_2 LAYERS_2
//
[[noreturn]]
static void usage(const char * executable) {
    printf("usage: %s ./path/model_1 LAYERS_1 ./path/model_2 LAYERS_2\n\n", executable);
    printf("  LAYERS must be in format: p0-p1,p2-p3,p4,... Example: 0-5,7,8-12\n");
    //printf("  Optionally, you can specify the scaling for a range of layers, for example: 0-5*0.5,6-7*1\n");
    printf("  The embedding layer of the first model will be used");
    exit(1);
}

int main(int argc, char ** argv) {
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);
    llama_backend_free();
}