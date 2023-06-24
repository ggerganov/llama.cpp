#include "embd-input.h"
#include <stdlib.h>
#include <random>
#include <string.h>

int main(int argc, char** argv) {

    auto mymodel = create_mymodel(argc, argv);
    int N = 10;
    int max_tgt_len = 500;
    int n_embd = llama_n_embd(mymodel->ctx);

    // add random float embd to test evaluation
    float * data = new float[N*n_embd];
    std::default_random_engine e;
    std::uniform_real_distribution<float>  u(0,1);
    for (int i=0;i<N*n_embd;i++) {
        data[i] = u(e);
    }

    eval_string(mymodel, "user: what is the color of the flag of UN?");
    eval_float(mymodel, data, N);
    eval_string(mymodel, "assistant:");
    eval_string(mymodel, mymodel->params.prompt.c_str());
    const char* tmp;
    for (int i=0; i<max_tgt_len; i++) {
        tmp = sampling(mymodel);
        if (strcmp(tmp, "</s>")==0) break;
        printf("%s", tmp);
        fflush(stdout);
    }
    printf("\n");
    free_mymodel(mymodel);
    return 0;
}
