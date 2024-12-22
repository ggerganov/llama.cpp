#include "llama-adapter.h"

void llama_lora_adapter_free(struct llama_lora_adapter * adapter) {
    delete adapter;
}
