#include "embd_input.h"
#include <stdlib.h>
#include <random>

int main(int argc, char** argv) {

    auto mymodel = create_mymodel(argc, argv);
    int N = 10;
    int n_embd = llama_n_embd(mymodel->ctx);
    float* data = new float[N*n_embd];
    std::default_random_engine e;
    std::uniform_real_distribution<float>  u(0,1);
    for (int i=0;i<N*n_embd;i++) {
        data[i] = u(e);
    }

    eval_string(mymodel, "111");
    printf("eval float");
    eval_float(mymodel, data, N);
    printf("eval float end\n");
    eval_string(mymodel, mymodel->params.prompt.c_str());
    for (int i=0;i < 500; i++) {
        int id = sampling_id(mymodel);
        printf("%s", llama_token_to_str(mymodel->ctx, id));
        eval_id(mymodel, id);
    }
    printf("\n");
    return 0;
}
