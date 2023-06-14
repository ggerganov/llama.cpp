#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "build-info.h"
#include "ggml-tune.h"
#include "ggml.h"
#include "llama.h"

#define UNUSED(x) (void)(x)

static void print_build_tips(void) {
    const char *a = "LLAMA_NO_ACCELERATE";
    fprintf(stderr, "Tips on how to build with various backend vendors:\n\n");
    fprintf(stderr, "CUDA:       make clean; LLAMA_CUBLAS=1 make\n");
    fprintf(stderr, "CL:         make clean; LLAMA_CLBLAST=1 make\n");
    fprintf(stderr, "Accelerate: make clean; %s=  make\n", a);
    fprintf(stderr, "OpenBLAS:   make clean; %s=1 LLAMA_OPENBLAS=1 make\n", a);
    fprintf(stderr, "BLIS:       make clean; %s=1 LLAMA_BLIS=1 make\n", a);
    fprintf(stderr, "\n");
    fprintf(stderr, "NOTE: for CUDA/CL, use %s=1 to disable ACCELERATE\n", a);
}

static bool prompt_yes_no(const char *prompt) {
    char buf[2];
    while (true) {
        fprintf(stderr, "%s (Y|n)\n", prompt);
        buf[0] = 0;
        buf[1] = 0;
        int i = 0;
        int c = 0;

        while (c != '\n') {
            c = fgetc(stdin);
            buf[i % 2] = c;
            i++;
        }
        if (i == 1) {
            if (buf[0] == '\n') {
                return true;
            }
        } else if (i == 2) {
            if (buf[0] == 'Y' || buf[0] == 'y') {
                return true;
            }
            if (buf[0] == 'N' || buf[0] == 'n') {
                return false;
            }
        }
    }
}

static void cmd_analyze(struct ggml_mulmat_tune *tune);

static void usage(char *prog) {
    const char *usage_lines[] = {
        "usage: %s args",
        "",
        "bench [-m MODEL] [-t TYPE] [-f FILE] [-y]",
        "--model     MODEL    3B | 7B | 13B | 30B | 65B",
        "                     default 7B",
        "--ftype     FTYPE    ggml ftype:",
        "                     0:  all F32",
        "                     1:  mostly F16",
        "                     2:  mostly Q4_0",
        "                     3:  mostly Q4_1",
        "                     4:  mostly Q4_1, some F16",
        "                     7:  mostly Q8_0",
        "                     8:  mostly Q5_0",
        "                     9:  mostly Q5_1",
        "                     10: mostly Q2_K",
        "                     11: mostly Q3_K",
        "                     12: mostly Q4_K",
        "                     13: mostly Q5_K",
        "                     14: mostly Q6_K",
        "                     default 2 (mostly Q4_0)",
        "--m_num     M_NUM    number of M, the max M = 2^(M_NUM-1)",
        "                     requires between [6, 12]",
        "                     default 10",
        "--n_pass    PASS     number of passes to run",
        "                     default 1",
        "                     requires: between [1, 3]",
        "--n_threads NTH      bench with this number of threads",
        "                     requires: between [1, 16]",
        "                     default 1",
        "--file      FILE     data file to write",
        "                     default stdout",
        "-y                   always answer \"yes\" to all prompts",
    };

    int len = (int)(sizeof(usage_lines) / sizeof(char *));
    for (int i = 0; i < len; i++) {
        const char *line = usage_lines[i];
        if (i == 0) {
            fprintf(stderr, line, prog);
        } else {
            fprintf(stderr, "%s\n", line);
        }
    }

    printf("\n");
    print_build_tips();
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc == 2) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            usage(argv[0]);
            return 0;
        }
    }

    int arg_start = 1;

    const char *arg_model = NULL;
    const char *arg_ftype = NULL;
    const char *arg_m_num = NULL;
    const char *arg_n_threads = NULL;
    const char *arg_n_pass = NULL;
    const char *arg_file = NULL;
    bool always_yes = false;

    for (int i = arg_start; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0) {
            if (i + 1 < argc) {
                arg_model = argv[i + 1];
                ++i;
            }
        } else if (strcmp(argv[i], "--ftype") == 0) {
            if (i + 1 < argc) {
                arg_ftype = argv[i + 1];
                ++i;
            }
        } else if (strcmp(argv[i], "--m_num") == 0) {
            if (i + 1 < argc) {
                arg_m_num = argv[i + 1];
                ++i;
            }
        } else if (strcmp(argv[i], "--n_pass") == 0) {
            if (i + 1 < argc) {
                arg_n_pass = argv[i + 1];
                ++i;
            }
        } else if (strcmp(argv[i], "--n_threads") == 0) {
            if (i + 1 < argc) {
                arg_n_threads = argv[i + 1];
                ++i;
            }
        } else if (strcmp(argv[i], "--file") == 0) {
            if (i + 1 < argc) {
                arg_file = argv[i + 1];
                ++i;
            }
        } else if (strcmp(argv[i], "-y") == 0) {
            always_yes = true;
        } else {
            fprintf(stderr, "invalid arg: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    enum ggml_ftype ftype = GGML_FTYPE_MOSTLY_Q4_0;
    {
        if (arg_ftype != NULL) {
            int v = atoi(arg_ftype);
            ftype = (enum ggml_ftype)v;
        }

        if (ftype > GGML_FTYPE_MOSTLY_Q5_1) {
            fprintf(stderr, "k_quants type %d is not implemented\n", ftype);
            return 1;
        }
    }

    if (arg_file != NULL && !always_yes) {
        struct stat st;
        int rc = stat(arg_file, &st);
        UNUSED(st);
        if (rc == 0) { // prompt
            size_t len = strlen(arg_file) + 50;
            char *prompt = (char *)malloc(len);
            GGML_ASSERT(prompt);
            snprintf(prompt, len, "data file '%s' exists, override?", arg_file);

            if (!prompt_yes_no(prompt)) {
                printf("Aborted.\n");
                return 1;
            }
            free(prompt);
        }
    }

    int m_num = 10;
    {
        if (arg_m_num != NULL) {
            int v = atoi(arg_m_num);
            m_num = v;
        }

        if (m_num < 6 || m_num > 12) {
            fprintf(stderr, "invalid m_num: %d, expect between [6, 12]\n",
                    m_num);
            usage(argv[0]);
            return 1;
        }
    }

    int n_pass = 1;
    {
        if (arg_n_pass != NULL) {
            int v = atoi(arg_n_pass);
            n_pass = v;
        }
        if (n_pass < 1 || n_pass > GGML_MULMAT_MAX_PASS) {
            fprintf(stderr, "invalid n_pass: %d, expect between [1, %d]\n",
                    n_pass, GGML_MULMAT_MAX_PASS);
            usage(argv[0]);
            return 1;
        }
    }

    int n_threads = 1;
    {
        if (arg_n_threads != NULL) {
            int v = atoi(arg_n_threads);
            n_threads = v;
            if (n_threads < 1 || n_threads > 16) {
                fprintf(stderr,
                        "invalid n_threads: %d, expect between [1, 16]\n",
                        n_threads);
                usage(argv[0]);
                return 1;
            }
        }
    }

    const char *model_name = "7B";
    {
        if (arg_model != NULL) {
            model_name = arg_model;
        }
    }

    // Let init message print earlier.
    {
        struct ggml_init_params init_params = {
            /*.mem_size   =*/1,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/0,
        };
        struct ggml_context *ctx = ggml_init(init_params);
        GGML_ASSERT(ctx);
        ggml_free(ctx);
    }

    struct ggml_mulmat_tune tune;

    struct ggml_mulmat_tune_params params;
    memset(&params, 0, sizeof(struct ggml_mulmat_tune_params));

    ggml_mulmat_tune_model_init(&params.model, model_name, ftype);
    params.m_num = m_num;
    params.n_pass = n_pass;
    params.n_threads = n_threads;
    params.progress = true;
    params.output_console = true;
    params.fname = arg_file;

    bool ok = ggml_mulmat_tune_bench(&tune, &params);
    return ok ? 0 : 1;
}
