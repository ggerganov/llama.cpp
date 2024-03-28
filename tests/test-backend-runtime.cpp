#include <ggml-alloc.h>
#include <ggml-backend-impl.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

struct test_case {
    virtual const char* case_desc() = 0;
    virtual bool eval(ggml_backend_t backend) = 0;
    std::vector<float> get_random_float(size_t size) {
        std::vector<float> random_data;
        random_data.resize(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 128);
        for (size_t i = 0; i < size; i++) {
            random_data[i] = dis(gen);
        }
        return random_data;
    }

    void init_context(size_t tensor_count) {
        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * tensor_count,
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ctx = ggml_init(params);
    }

    ggml_tensor* new_tensor(ggml_backend_t backend, int dims, int64_t* ne) {
        ggml_tensor* tensor = ggml_new_tensor(ctx, GGML_TYPE_F32, dims, ne);
        ggml_backend_buffer_t buf =
            ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == NULL) {
            printf("failed to allocate tensors [%s] ",
                   ggml_backend_name(backend));
            ggml_free(ctx);
            return nullptr;
        }
        return tensor;
    }

    ggml_context* ctx;
};

struct test_tensor_get_set_cpy_async : public test_case {
    virtual const char* case_desc() { return "test_tensor_get_set_cpy_async"; }
    virtual bool eval(ggml_backend_t backend) {
        // init context
        init_context(2);

        // alloc tensor
        int64_t ne[] = {10, 10};
        ggml_tensor* tensor1 =
            new_tensor(backend, sizeof(ne) / sizeof(ne[0]), ne);
        ggml_tensor* tensor2 =
            new_tensor(backend, sizeof(ne) / sizeof(ne[0]), ne);

        // get random data
        int64_t elements = ggml_nelements(tensor1);
        std::vector<float> random_data = get_random_float(elements);
        std::vector<float> verify_data;
        verify_data.resize(elements);

        // upload and download data
        ggml_backend_tensor_set_async(backend, tensor1,
                                      (void*)random_data.data(), 0,
                                      ggml_nbytes(tensor1));
        ggml_backend_tensor_copy_async(backend, backend, tensor1, tensor2);
        ggml_backend_tensor_get_async(backend, tensor2,
                                      (void*)verify_data.data(), 0,
                                      ggml_nbytes(tensor2));
        ggml_backend_synchronize(backend);

        return (memcmp(random_data.data(), verify_data.data(),
                       sizeof(float) * elements) == 0);
    }
};

struct test_tensor_get_set_cpy : public test_case {
    virtual const char* case_desc() { return "test_tensor_get_set_cpy"; }
    virtual bool eval(ggml_backend_t backend) {
        // init context
        init_context(2);

        // alloc tensor
        int64_t ne[] = {10, 10};
        ggml_tensor* tensor1 =
            new_tensor(backend, sizeof(ne) / sizeof(ne[0]), ne);
        ggml_tensor* tensor2 =
            new_tensor(backend, sizeof(ne) / sizeof(ne[0]), ne);

        // get random data
        int64_t elements = ggml_nelements(tensor1);
        std::vector<float> random_data = get_random_float(elements);
        std::vector<float> verify_data;
        verify_data.resize(elements);

        // upload and download data
        ggml_backend_tensor_set(tensor1, (void*)random_data.data(), 0,
                                ggml_nbytes(tensor1));
        ggml_backend_tensor_copy(tensor1, tensor2);
        ggml_backend_tensor_get(tensor2, (void*)verify_data.data(), 0,
                                ggml_nbytes(tensor2));

        return (memcmp(random_data.data(), verify_data.data(),
                       sizeof(float) * elements) == 0);
    }
};

static bool test_backend(ggml_backend_t backend) {
    std::vector<std::unique_ptr<test_case>> test_cases;
    test_cases.emplace_back(new test_tensor_get_set_cpy_async());
    test_cases.emplace_back(new test_tensor_get_set_cpy());

    size_t n_ok = 0;
    for (auto& test : test_cases) {
        printf("  %s ", test->case_desc());
        if (test->eval(backend)) {
            n_ok++;
            printf("\033[1;32mOK\033[0m\n");
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
    }
    printf("  %zu/%zu tests passed\n", n_ok, test_cases.size());

    return n_ok == test_cases.size();
}

static void usage(char** argv) { printf("Usage: %s [-b backend]\n", argv[0]); }

int main(int argc, char** argv) {
    const char* backend = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                backend = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else {
            usage(argv);
            return 1;
        }
    }

    printf("Testing %zu backends\n\n", ggml_backend_reg_get_count());

    size_t n_ok = 0;
    for (size_t i = 0; i < ggml_backend_reg_get_count(); i++) {
        if (backend != NULL &&
            strcmp(backend, ggml_backend_reg_get_name(i)) != 0) {
            printf("  Skipping %s\n\n", ggml_backend_reg_get_name(i));
            n_ok++;
            continue;
        }

        ggml_backend_t backend = ggml_backend_reg_init_backend(i, NULL);

        GGML_ASSERT(backend != NULL);
        printf("  Backend name: %s\n", ggml_backend_name(backend));

        bool ok = test_backend(backend);

        printf("  Backend %s: ", ggml_backend_name(backend));
        if (ok) {
            printf("\033[1;32mOK\033[0m\n");
            n_ok++;
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }

        printf("\n");
    }

    return 0;
}