#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "get_model.h"

char * get_model_or_exit(int argc, char *argv[]) {
    char * makelevel = getenv("MAKELEVEL");
    if (makelevel != nullptr && atoi(makelevel) > 0) {
        fprintf(stderr, "Detected being run in Make. Skipping this test.\n");
        exit(EXIT_SUCCESS);
    }

    char * model_path;
    if (argc > 1) {
        model_path = argv[1];
    } else {
        model_path = getenv("GG_RUN_CTEST_MODELFILE");
        if (!model_path || strlen(model_path) == 0) {
            fprintf(stderr, "error: no model file provided\n");
            exit(EXIT_FAILURE);
        }
    }

    return model_path;
}
