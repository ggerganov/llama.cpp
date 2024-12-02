#include "omni.h"
#include <iostream>

using std::cout;
using std::endl;

int main(int argc, char **argv)
{

    omni_context_params ctx_params = omni_context_default_params();
    if (!omni_context_params_parse(argc, argv, ctx_params))
    {
        return 1;
    }

    omni_context *ctx_omni = omni_init_context(ctx_params);

    auto ret = omni_process_full(ctx_omni, ctx_params);
    cout << "[OMNI_PROCESS_FULL OUTPUT:] ";
    cout << ret << endl;

    omni_free(ctx_omni);

    return 0;
}
