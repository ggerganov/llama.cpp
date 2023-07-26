#include "ggml.h"

#include <cstdio>
#include <string>

bool gguf_write(const std::string & fname) {


    return true;
}

bool gguf_read(const std::string & fname) {
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stdout, "usage: %s data.gguf r|w\n", argv[0]);
        return -1;
    }

    const std::string fname(argv[1]);
    const std::string mode(argv[2]);

    GGML_ASSERT((mode == "r" || mode == "w") && "mode must be r or w");

    if (mode == "w") {
        GGML_ASSERT(gguf_write(fname) && "failed to write gguf file");
    } else if (mode == "r") {
        GGML_ASSERT(gguf_read(fname)  && "failed to read gguf file");
    }

    return 0;
}
