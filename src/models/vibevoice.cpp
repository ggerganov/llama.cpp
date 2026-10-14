#include "models.h"

void llama_model_vibevoice::load_arch_tensors(llama_model_loader & ml) {
    llama_model_qwen2::load_arch_tensors(ml);

    // TODO: load connectors, diffusion head and acoustic/semantic tokenizers
    throw std::runtime_error("vibevoice: speech tensors are not supported yet");
}
