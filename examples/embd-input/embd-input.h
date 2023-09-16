#ifndef _EMBD_INPUT_H_
#define _EMBD_INPUT_H_ 1

#include "common.h"
#include "llama.h"

extern "C" {

typedef struct MyModel {
    llama_context* ctx;
    gpt_params params;
    int n_past = 0;
} MyModel;

struct MyModel* create_mymodel(int argc, char ** argv);

bool eval_float(void* model, float* input, int N);
bool eval_tokens(void* model, std::vector<llama_token> tokens);
bool eval_id(struct MyModel* mymodel, int id);
bool eval_string(struct MyModel* mymodel, const char* str);
const char * sampling(struct MyModel* mymodel);
llama_token sampling_id(struct MyModel* mymodel);
void free_mymodel(struct MyModel* mymodel);

}

#endif
