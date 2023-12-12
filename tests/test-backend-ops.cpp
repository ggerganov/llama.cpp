#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-backend-impl.h>
#include <algorithm>
#include <array>
#include <cfloat>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <vector>


static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    size_t size = ggml_nelements(tensor);
    std::vector<float> data(size);

#if 0
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<float> distribution(min, max);

    for (size_t i = 0; i < size; i++) {
        data[i] = distribution(generator);
    }
#endif
    auto init_thread = [&](size_t start, size_t end) {
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::uniform_real_distribution<float> distribution(min, max);

        for (size_t i = start; i < end; i++) {
            data[i] = distribution(generator);
        }
    };

    size_t n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (size_t i = 0; i < n_threads; i++) {
        size_t start =     i*size/n_threads;
        size_t end   = (i+1)*size/n_threads;
        threads.emplace_back(init_thread, start, end);
    }
    for (auto & t : threads) {
        t.join();
    }

    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, size * sizeof(float));
    } else if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16) {
        GGML_ASSERT(size % ggml_blck_size(tensor->type) == 0);
        std::vector<uint8_t> dataq(ggml_type_size(tensor->type)*size/ggml_blck_size(tensor->type));
        int64_t hist[16];
        ggml_quantize_chunk(tensor->type, data.data(), dataq.data(), 0, size, hist);
        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    } else {
        GGML_ASSERT(false);
    }
}

static std::vector<float> tensor_to_float(const ggml_tensor * t) {
    std::vector<float> tv;
    tv.reserve(ggml_nelements(t));

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    ggml_type_traits_t tt = ggml_internal_get_type_traits(t->type);
    size_t bs = ggml_blck_size(t->type);

    // access elements by index to avoid gaps in views
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0 += bs) {
                    size_t i = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0/bs*t->nb[0];
                    if (t->type == GGML_TYPE_F16) {
                        tv.push_back(ggml_fp16_to_fp32(*(ggml_fp16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_F32) {
                        tv.push_back(*(float *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I32) {
                        tv.push_back((float)*(int32_t *) &buf[i]);
                    } else if (ggml_is_quantized(t->type)) {
                        std::vector<float> vq(ggml_blck_size(t->type));
                        tt.to_float(&buf[i], vq.data(), ggml_blck_size(t->type));
                        tv.insert(tv.end(), vq.begin(), vq.end());
                    } else {
                        GGML_ASSERT(false);
                    }
                }
            }
        }
    }

    return tv;
}

/*
static double cosine_similarity(const float * v1, const float * v2, size_t n) {
    double dot = 0.0;
    double mag1 = 0.0;
    double mag2 = 0.0;

    for (size_t i = 0; i < n; i++) {
        if (std::isnan(v1[i]) || std::isnan(v2[i])) {
            return -1.0f;
        }
        if (std::isinf(v1[i]) && std::isinf(v2[i])) {
            continue;
        }
        dot  += v1[i]*v2[i];
        mag1 += v1[i]*v1[i];
        mag2 += v2[i]*v2[i];
    }

    return dot/sqrt(mag1*mag2);
}

static float distance(const float * v1, const float * v2, size_t n) {
    double d = 0.0;

    for (size_t i = 0; i < n; i++) {
        if (std::isnan(v1[i]) || std::isnan(v2[i])) {
            return INFINITY;
        }
        if (std::isinf(v1[i]) && std::isinf(v2[i])) {
            continue;
        }
        d += (v1[i] - v2[i])*(v1[i] - v2[i]);
    }

    return sqrt(d);
}

static float vec_len(const float * v, size_t n) {
    double d = 0.0;

    for (size_t i = 0; i < n; i++) {
        if (std::isnan(v[i])) {
            return INFINITY;
        }
        if (std::isinf(v[i])) {
            continue;
        }
        d += v[i]*v[i];
    }

    return sqrt(d);
}
*/

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < n; i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

// utils for printing the variables of the test cases
#define VAR_TO_STR(x) (#x "=" + var_to_str(x))

template<typename T>
static std::string var_to_str(const T & x) {
    return std::to_string(x);
}

template<typename T, size_t N>
static std::string var_to_str(const T (&x)[N]) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

template<typename T, size_t N>
static std::string var_to_str(const std::array<T, N> & x) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

//static std::string var_to_str(ggml_unary_op unary_op) {
//    return ggml_unary_op_name(unary_op);
//}

static std::string var_to_str(ggml_type type) {
    return ggml_type_name(type);
}

#define VARS_TO_STR1(a) VAR_TO_STR(a)
#define VARS_TO_STR2(a, b) VAR_TO_STR(a) + "," + VAR_TO_STR(b)
#define VARS_TO_STR3(a, b, c) VAR_TO_STR(a) + "," + VARS_TO_STR2(b, c)
#define VARS_TO_STR4(a, b, c, d) VAR_TO_STR(a) + "," + VARS_TO_STR3(b, c, d)
#define VARS_TO_STR5(a, b, c, d, e) VAR_TO_STR(a) + "," + VARS_TO_STR4(b, c, d, e)
#define VARS_TO_STR6(a, b, c, d, e, f) VAR_TO_STR(a) + "," + VARS_TO_STR5(b, c, d, e, f)
#define VARS_TO_STR7(a, b, c, d, e, f, g) VAR_TO_STR(a) + "," + VARS_TO_STR6(b, c, d, e, f, g)
#define VARS_TO_STR8(a, b, c, d, e, f, g, h) VAR_TO_STR(a) + "," + VARS_TO_STR7(b, c, d, e, f, g, h)
#define VARS_TO_STR9(a, b, c, d, e, f, g, h, i) VAR_TO_STR(a) + "," + VARS_TO_STR8(b, c, d, e, f, g, h, i)
#define VARS_TO_STR10(a, b, c, d, e, f, g, h, i, j) VAR_TO_STR(a) + "," + VARS_TO_STR9(b, c, d, e, f, g, h, i, j)
#define VARS_TO_STR11(a, b, c, d, e, f, g, h, i, j, k) VAR_TO_STR(a) + "," + VARS_TO_STR10(b, c, d, e, f, g, h, i, j, k)


// accept FLT_MAX as infinity
static bool isinf_or_max(float f) {
    return std::isinf(f) || f == FLT_MAX || f == -FLT_MAX;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

struct test_case {
    virtual ~test_case() {}

    virtual std::string op_desc(ggml_tensor * t) {
        return ggml_op_desc(t);
    }

    virtual std::string vars() {
        return "";
    }

    virtual ggml_tensor * build_graph(ggml_context * ctx) = 0;

    virtual double max_nmse_err() {
        return 1e-7;
    }

    virtual void initialize_tensors(ggml_context * ctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t);
        }
    }

    virtual size_t op_size(ggml_tensor * t) {
        size_t size = ggml_nbytes(t);
        // add source tensors
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (t->src[i] != NULL) {
                size += ggml_nbytes(t->src[i]);
            }
        }
        return size;
    }

    bool eval(ggml_backend_t backend1, ggml_backend_t backend2, const char * op_name) {
        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context * ctx = ggml_init(params);

        ggml_tensor * out = build_graph(ctx);

        if (op_name != nullptr && op_desc(out) != op_name) {
            //printf("  %s: skipping\n", op_desc(out).c_str());
            ggml_free(ctx);
            return true;
        }

        printf("  %s(%s): ", op_desc(out).c_str(), vars().c_str());
        fflush(stdout);

        // check if backends support op
        for (ggml_backend_t backend : {backend1, backend2}) {
            if (!ggml_backend_supports_op(backend, out)) {
                printf("not supported\n");
                ggml_free(ctx);
                return true;
            }
        }

        // allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend1);

        // build graph
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        // randomize tensors
        initialize_tensors(ctx);

        // compare
        struct callback_userdata {
            bool   ok;
            double max_err;
        };

        callback_userdata ud {
            true,
            max_nmse_err(),
        };

        auto callback = [](int index, ggml_tensor * t1, ggml_tensor * t2, void * user_data) -> bool {
            std::vector<float> f1 = tensor_to_float(t1);
            std::vector<float> f2 = tensor_to_float(t2);
            callback_userdata * ud = (callback_userdata *) user_data;

            for (size_t i = 0; i < f1.size(); i++) {
                // check for nans
                if (std::isnan(f1[i]) || std::isnan(f2[i])) {
                    printf("[%s] NaN at index %zu (%f %f) ", ggml_op_desc(t1), i, f1[i], f2[i]);
                    ud->ok = false;
                    return true;
                }
                // check for infs: both must be inf of the same sign, or both must be finite
                if (isinf_or_max(f1[i]) || isinf_or_max(f2[i])) {
                    if (isinf_or_max(f1[i]) && isinf_or_max(f2[i])) {
                        if (std::signbit(f1[i]) != std::signbit(f2[i])) {
                            printf("[%s] inf sign mismatch: %f %f ", ggml_op_desc(t1), f1[i], f2[i]);
                            ud->ok = false;
                            return true;
                        }
                    } else {
                        printf("[%s] inf mismatch: %f %f ", ggml_op_desc(t1), f1[i], f2[i]);
                        ud->ok = false;
                        return true;
                    }
                }
            }

            double err = nmse(f1.data(), f2.data(), f1.size());
            if (err > ud->max_err) {
                printf("[%s] NMSE = %f ", ggml_op_desc(t1), err);
                //for (int i = 0; i < f1.size(); i++) {
                //    printf("(%f, %f) ", f1[i], f2[i]);
                //}
                //printf("\n");
                ud->ok = false;
            }
            return true;

            GGML_UNUSED(index);
        };

        ggml_backend_compare_graph_backend(backend1, backend2, gf, callback, &ud);

        if (ud.ok) {
            printf("\033[1;32mOK\033[0m\n");
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }

        ggml_backend_buffer_free(buf);

        ggml_free(ctx);

        return ud.ok;
    }

    bool eval_perf(ggml_backend_t backend, const char * op_name) {
        static const size_t graph_nodes = 8192;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead_custom(graph_nodes, false),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context * ctx = ggml_init(params);

        ggml_tensor * out = build_graph(ctx);

        if (op_name != nullptr && op_desc(out) != op_name) {
            //printf("  %s: skipping\n", op_desc(out).c_str());
            ggml_free(ctx);
            return true;
        }

        int len = printf("  %s(%s): ", op_desc(out).c_str(), vars().c_str());
        fflush(stdout);

        // check if backends support op
        if (!ggml_backend_supports_op(backend, out)) {
            printf("not supported\n");
            ggml_free(ctx);
            return true;
        }

        // align while also leaving some margin for variations in parameters
        int align = 20;
        int last = (len + align - 1) / align * align;
        if (last - len < 5) {
            last += align;
        }
        last = std::max(last, 60);
        printf("%*s", last - len, "");

        // allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

        // randomize tensors
        initialize_tensors(ctx);

        // build graph
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_nodes, false);
        ggml_build_forward_expand(gf, out);

        // warmup run
        ggml_backend_graph_compute(backend, gf);

        // duplicate the op
        size_t target_size = ggml_backend_is_cpu(backend) ? 1ULL << 33 : 1ULL << 35; // 8 GB CPU, 32 GB GPU
        int n_runs = std::min((size_t)gf->size - gf->n_nodes, target_size / op_size(out)) + 1;
        for (int i = 1; i < n_runs; i++) {
            gf->nodes[gf->n_nodes++] = out;
        }

        // calculate memory
        size_t mem = n_runs * op_size(out);
        auto tensor_op_size = [](ggml_tensor * t) {
            size_t size = ggml_nbytes(t);
            // add source tensors
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (t->src[i] != NULL) {
                    size += ggml_nbytes(t->src[i]);
                }
            }
            return size;
        };
        for (int i = 0; i < gf->n_nodes; i++) {
            if (ggml_is_view_op(gf->nodes[i]->op) || gf->nodes[i] == out) {
                continue;
            }
            mem += tensor_op_size(gf->nodes[i]);
        }

        // run
        ggml_backend_synchronize(backend);

        int64_t start_time = ggml_time_us();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        int64_t end_time = ggml_time_us();
        double time_us = end_time - start_time;

        printf("    %5d runs - %8.2f us/run - %8zu kB/run - \033[1;34m%7.2f GB/s\033[0m\n",
            n_runs,
            time_us / n_runs,
            op_size(out) / 1024,
            mem / (time_us/1e6) / 1024.0 / 1024.0 / 1024.0);

        ggml_backend_buffer_free(buf);

        ggml_free(ctx);

        return true;
    }
};

// GGML_OP_UNARY
struct test_unary : public test_case {
    const ggml_unary_op op;
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_unary(ggml_unary_op op,
            ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {128, 10, 10, 10})
        : op(op), type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * in = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_unary(ctx, in, op);
        return out;
    }
};

// GGML_OP_GET_ROWS
struct test_get_rows : public test_case {
    const ggml_type type;
    const int n; // cols
    const int m; // rows
    const int r; // rows to get
    const int b; // batch size
    const bool v; // view (non-contiguous src1)

    std::string vars() override {
        return VARS_TO_STR6(type, n, m, r, b, v);
    }

    test_get_rows(ggml_type type = GGML_TYPE_F32, int n = 10, int m = 5, int r = 3, int b = 1, bool v = false)
        : type(type), n(n), m(m), r(r), b(b), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * in = ggml_new_tensor_3d(ctx, type, n, m, b);
        ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, r, b);
        if (v) {
            rows = ggml_view_2d(ctx, rows, r/2, b, rows->nb[1], 0);
        }
        ggml_tensor * out = ggml_get_rows(ctx, in, rows);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                // rows
                std::vector<int> data(r*b);
                for (int i = 0; i < r*b; i++) {
                    data[i] = rand() % m;
                }
                ggml_backend_tensor_set(t, data.data(), 0, r * b * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_REPEAT
struct test_repeat : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4> nr;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, nr);
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) * 2;
    }

    test_repeat(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            std::array<int, 4> nr = {2, 2, 2, 2})
        : type(type), ne(ne), nr(nr) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * target = ggml_new_tensor_4d(ctx, type, ne[0]*nr[0], ne[1]*nr[1], ne[2]*nr[2], ne[3]*nr[3]);
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_repeat(ctx, src, target);
        return out;
    }
};

// GGML_OP_DUP
struct test_dup : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_dup(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 1})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_dup(ctx, src);
        return out;
    }
};

// GGML_OP_CPY
struct test_cpy : public test_case {
    const ggml_type type_src;
    const ggml_type type_dst;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR3(type_src, type_dst, ne);
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) + ggml_nbytes(t->src[0]);
    }

    test_cpy(ggml_type type_src = GGML_TYPE_F32, ggml_type type_dst = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 1})
        : type_src(type_src), type_dst(type_dst), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type_src, 4, ne.data());
        ggml_tensor * dst = ggml_new_tensor(ctx, type_dst, 4, ne.data());
        ggml_tensor * out = ggml_cpy(ctx, src, dst);
        return out;
    }
};

// GGML_OP_CONT
struct test_cont : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_cont(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 1})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        src = ggml_transpose(ctx, src);
        ggml_tensor * out = ggml_cont(ctx, src);

        return out;
    }
};

// GGML_OP_ADD
// GGML_OP_MUL
// GGML_OP_DIV
struct test_bin_bcast : public test_case {
    using op_t = ggml_tensor * (*) (ggml_context *, ggml_tensor *, ggml_tensor *);
    op_t op;
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4> nr;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, nr);
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) * 3;
    }

    test_bin_bcast(op_t op, ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 1, 1},
            std::array<int, 4> nr = {1, 2, 1, 1})
        : op(op), type(type), ne(ne), nr(nr) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0]*nr[0], ne[1]*nr[1], ne[2]*nr[2], ne[3]*nr[3]);
        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = op(ctx, a, b);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (op == ggml_div) {
                // avoid division by zero
                init_tensor_uniform(t, 1.0f, 2.0f);
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_SCALE
struct test_scale : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_scale(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * scale = ggml_new_tensor_1d(ctx, type, 1);
        ggml_tensor * out = ggml_scale(ctx, a, scale);
        return out;
    }
};

// GGML_OP_NORM
struct test_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float eps;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, eps);
    }

    test_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 10, 10, 10},
            float eps = 1e-6f)
        : type(type), ne(ne), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_norm(ctx, a, eps);
        return out;
    }
};

// GGML_OP_RMS_NORM
struct test_rms_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float eps;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, eps);
    }

    test_rms_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 10, 10, 10},
            float eps = 1e-6f)
        : type(type), ne(ne), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_rms_norm(ctx, a, eps);
        return out;
    }
};

// GGML_OP_MUL_MAT
struct test_mul_mat : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const std::array<int64_t, 2> bs; // dims 3 and 4
    const std::array<int64_t, 2> nr; // repeat in dims 3 and 4

    std::string vars() override {
        return VARS_TO_STR7(type_a, type_b, m, n, k, bs, nr);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    size_t op_size(ggml_tensor * t) override {
        size_t a = ggml_nbytes(t->src[0]) * n * nr[0] * nr[1];
        size_t b = ggml_nbytes(t->src[1]) * m;
        size_t c  = ggml_nbytes(t);
        return a + b + c;

        GGML_UNUSED(t);
    }

    test_mul_mat(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int64_t m = 32, int64_t n = 32, int64_t k = 32,
            std::array<int64_t, 2> bs = {10, 10},
            std::array<int64_t, 2> nr = {2, 2})
        : type_a(type_a), type_b(type_b), m(m), n(n), k(k), bs(bs), nr(nr) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type_a, k, m, bs[0]      , bs[1]);
        ggml_tensor * b = ggml_new_tensor_4d(ctx, type_b, k, n, bs[0]*nr[0], bs[1]*nr[1]);
        ggml_tensor * out = ggml_mul_mat(ctx, a, b);
        return out;
    }
};

// GGML_OP_MUL_MAT_ID
struct test_mul_mat_id : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int n_mats;
    const int id;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const bool v; // view (non-contiguous ids)

    std::string vars() override {
        return VARS_TO_STR8(type_a, type_b, n_mats, id, m, n, k, v);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    size_t op_size(ggml_tensor * t) override {
        size_t a = ggml_nbytes(t->src[2]) * n;
        size_t b = ggml_nbytes(t->src[1]) * m;
        size_t c  = ggml_nbytes(t);
        return a + b + c;

        GGML_UNUSED(t);
    }

    test_mul_mat_id(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int n_mats = 2, int id = 0,
            int64_t m = 32, int64_t n = 32, int64_t k = 32, bool v = false)
        : type_a(type_a), type_b(type_b), n_mats(n_mats), id(id),
            m(m), n(n), k(k), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        std::vector<ggml_tensor *> mats;
        for (int i = 0; i < n_mats; i++) {
            ggml_tensor * a = ggml_new_tensor_2d(ctx, type_a, k, m);
            mats.push_back(a);
        }
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_mats, n);
        if (v) {
            ids = ggml_view_2d(ctx, ids, n_mats/2, ids->ne[1], ids->nb[1], 0);
        }
        ggml_tensor * b = ggml_new_tensor_2d(ctx, type_b, k, n);
        ggml_tensor * out = ggml_mul_mat_id(ctx, mats.data(), n_mats, ids, v ? id/2 : id, b);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                // ids
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<int32_t> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i % n_mats;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_SQR
struct test_sqr : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_sqr(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_sqr(ctx, a);
        return out;
    }
};

// GGML_OP_CLAMP
struct test_clamp : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float min;
    float max;

    std::string vars() override {
        return VARS_TO_STR4(type, ne, min, max);
    }

    test_clamp(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            float min = -0.5f, float max = 0.5f)
        : type(type), ne(ne), min(min), max(max) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_clamp(ctx, a, min, max);
        return out;
    }
};

// GGML_OP_DIAG_MASK_INF
struct test_diag_mask_inf : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int n_past;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, n_past);
    }

    test_diag_mask_inf(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            int n_past = 5)
        : type(type), ne(ne), n_past(n_past) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_diag_mask_inf(ctx, a, n_past);
        return out;
    }
};

// GGML_OP_SOFT_MAX
struct test_soft_max : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_soft_max(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_soft_max(ctx, a);
        return out;
    }
};

// GGML_OP_ROPE
struct test_rope : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    int n_dims;
    int mode;
    int n_ctx;

    std::string vars() override {
        return VARS_TO_STR5(type, ne, n_dims, mode, n_ctx);
    }

    test_rope(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 1},
            int n_dims = 10, int mode = 0, int n_ctx = 512)
        : type(type), ne(ne), n_dims(n_dims), mode(mode), n_ctx(n_ctx) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne[2]);
        ggml_tensor * out = ggml_rope(ctx, a, pos, n_dims, mode, n_ctx);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // pos
                std::vector<int> data(ne[2]);
                for (int i = 0; i < ne[2]; i++) {
                    data[i] = rand() % n_ctx;
                }
                ggml_backend_tensor_set(t, data.data(), 0, ne[2] * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_ALIBI
struct test_alibi : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    int n_past;
    int n_head;
    float bias_max;

    std::string vars() override {
        return VARS_TO_STR5(type, ne, n_past, n_head, bias_max);
    }

    test_alibi(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            int n_past = 512, int n_head = 10, float bias_max = 0.5f)
        : type(type), ne(ne), n_past(n_past), n_head(n_head), bias_max(bias_max) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_alibi(ctx, a, n_past, n_head, bias_max);
        return out;
    }
};

// GGML_OP_IM2COL
struct test_im2col : public test_case {
    const ggml_type type_input;
    const ggml_type type_kernel;
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;
    // stride
    const int s0;
    const int s1;
    // padding
    const int p0;
    const int p1;
    // dilatation
    const int d0;
    const int d1;
    // mode
    const bool is_2D;

    std::string vars() override {
        return VARS_TO_STR11(type_input, type_kernel, ne_input, ne_kernel, s0, s1, p0, p1, d0, d1, is_2D);
    }

    test_im2col(ggml_type type_input = GGML_TYPE_F32, ggml_type type_kernel = GGML_TYPE_F16,
            std::array<int64_t, 4> ne_input = {10, 10, 3, 1}, // [input_width, input_height, input_channels, 1]
            std::array<int64_t, 4> ne_kernel = {3, 3, 3, 1}, // [kernel_width, kernel_height, input_channels, 1]
            int s0 = 1, int s1 = 1,
            int p0 = 1, int p1 = 1,
            int d0 = 1, int d1 = 1,
            bool is_2D = true)
        : type_input(type_input), type_kernel(type_kernel), ne_input(ne_input), ne_kernel(ne_kernel), s0(s0), s1(s1), p0(p0), p1(p1), d0(d0), d1(d1), is_2D(is_2D) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, type_input, 4, ne_input.data());
        ggml_tensor * kernel = ggml_new_tensor(ctx, type_kernel, 4, ne_kernel.data());
        ggml_tensor * out = ggml_im2col(ctx, kernel, input, s0, s1, p0, p1, d0, d1, is_2D);
        return out;
    }
};

// GGML_OP_CONCAT
struct test_concat : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int64_t b_ne2;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, b_ne2);
    }

    test_concat(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            int64_t b_ne2 = 10)
        : type(type), ne(ne), b_ne2(b_ne2) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * b = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], b_ne2, ne[3]);
        ggml_tensor * out = ggml_concat(ctx, a, b);
        return out;
    }
};

// GGML_OP_ARGSORT
struct test_argsort : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    ggml_sort_order order;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, order);
    }

    test_argsort(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {16, 10, 10, 10},
            ggml_sort_order order = GGML_SORT_ASC)
        : type(type), ne(ne), order(order) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_argsort(ctx, a, order);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // indices
                std::vector<int> data(ggml_nelements(t));
                for (int i = 0; i < ggml_nelements(t); i++) {
                    data[i] = rand();
                }
                std::shuffle(data.begin(), data.end(), rng);
                ggml_backend_tensor_set(t, data.data(), 0, ne[0]*ne[1]*ne[2]*ne[3] * sizeof(int));
            } else if (t->type == GGML_TYPE_F32) {
                // initialize with unique values to avoid ties
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<float> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(float));
                }
            } else {
                GGML_ASSERT(false);
            }
        }
    }
};

// GGML_OP_SUM_ROWS
struct test_sum_rows : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_sum_rows(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_sum_rows(ctx, a);
        return out;
    }
};

// Mixtral MOE
struct test_moe : public test_case {
    const int n_experts;
    const int n_experts_per_tok;
    const int n_tokens;
    const int n_embd;
    const int n_ff;

    std::string op_desc(ggml_tensor * t) override {
        return "MOE";

        GGML_UNUSED(t);
    }

    std::string vars() override {
        return VARS_TO_STR5(n_experts, n_experts_per_tok, n_tokens, n_embd, n_ff);
    }

    test_moe(int n_experts = 8, int n_experts_per_tok = 2, int n_tokens = 1, int n_embd = 4096, int n_ff = 14336)
        : n_experts(n_experts), n_experts_per_tok(n_experts_per_tok), n_tokens(n_tokens), n_embd(n_embd), n_ff(n_ff) {
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * ffn_gate_inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_experts);

        std::vector<ggml_tensor *> ffn_up_exp(n_experts);
        std::vector<ggml_tensor *> ffn_gate_exp(n_experts);
        std::vector<ggml_tensor *> ffn_down_exp(n_experts);

        for (int i = 0; i < n_experts; ++i) {
            ffn_up_exp[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);
            ffn_gate_exp[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);
            ffn_down_exp[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff, n_embd);
        }

        ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);

        ggml_tensor * logits = ggml_mul_mat(ctx, ffn_gate_inp, cur);
        ggml_tensor * probs = ggml_soft_max_ext(ctx, logits, nullptr, 1.0f/sqrtf(n_embd));

        // select experts
        ggml_tensor * selected_experts = ggml_top_k(ctx, probs, n_experts_per_tok);

        ggml_tensor * weights = ggml_get_rows(ctx,
                ggml_reshape_3d(ctx, probs, 1, n_experts, n_tokens), selected_experts);

        weights = ggml_reshape_2d(ctx, weights, n_experts_per_tok, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx, weights);

        weights = ggml_div(ctx, weights, weights_sum);

        // compute expert outputs
        ggml_tensor * moe_out = nullptr;

        for (int i = 0; i < n_experts_per_tok; ++i) {
            ggml_tensor * cur_expert;

            ggml_tensor * cur_up = ggml_mul_mat_id(ctx, ffn_up_exp.data(), n_experts, selected_experts, i, cur);

            ggml_tensor * cur_gate = ggml_mul_mat_id(ctx, ffn_gate_exp.data(), n_experts, selected_experts, i, cur);

            cur_gate = ggml_silu(ctx, cur_gate);

            cur_expert = ggml_mul(ctx, cur_up, cur_gate);

            cur_expert = ggml_mul_mat_id(ctx, ffn_down_exp.data(), n_experts, selected_experts, i, cur_expert);

            cur_expert = ggml_mul(ctx, cur_expert,
                    ggml_view_2d(ctx, weights, 1, n_tokens, weights->nb[1], i*weights->nb[0]));

            if (i == 0) {
                moe_out = cur_expert;
            } else {
                moe_out = ggml_add(ctx, moe_out, cur_expert);
            }
        }

        cur = moe_out;

        return cur;
    }
};

enum test_mode {
    MODE_TEST,
    MODE_PERF,
};

static bool test_backend(ggml_backend_t backend, test_mode mode, const char * op_name) {
    std::vector<std::unique_ptr<test_case>> test_cases;

    const ggml_type all_types[] = {
        GGML_TYPE_F32, GGML_TYPE_F16,
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
        GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K
    };

    // unary ops
    for (int op = 0; op < GGML_UNARY_OP_COUNT; op++) {
        test_cases.emplace_back(new test_unary((ggml_unary_op) op));
    }

    test_cases.emplace_back(new test_get_rows(GGML_TYPE_F32, 1, 8, 2, 1, false));
    for (ggml_type type : all_types) {
        for (int b : {1, 7}) {
            for (bool v : {false, true}) {
                test_cases.emplace_back(new test_get_rows(type, 256, 5, 4, b, v));
            }
        }
    }

    test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 1, 1, 1}));
    test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 10, 10, 10}, {2, 1, 1, 1}));
    test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 2, 1, 1}));
    test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 1, 2, 1}));
    test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 1, 1, 2}));

    test_cases.emplace_back(new test_dup());

    for (ggml_type type : all_types) {
       test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, type, {256, 10, 10, 1}));
    }

    test_cases.emplace_back(new test_cont());

    auto add_test_bin_bcast = [&](ggml_type type, std::array<int64_t, 4> ne, std::array<int, 4> nr) {
        for (auto op : {ggml_add, ggml_mul, ggml_div}) {
            test_cases.emplace_back(new test_bin_bcast(op, type, ne, nr));
        }
    };

    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 8, 1}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 1, 1}, {32, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 320, 320}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 1, 1}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 1}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {2, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {1, 2, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {1, 1, 2, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {1, 1, 1, 2});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {1, 1, 2, 2});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {1, 2, 2, 2});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {2, 2, 2, 2});

    // stable diffusion
    add_test_bin_bcast(GGML_TYPE_F32, {1280, 1, 1, 1}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1280, 1, 1, 1}, {1, 16, 16, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1280, 16, 16, 1}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1280, 1, 1, 1}, {1, 256, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 1280, 1}, {16, 16, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 16, 1280, 1}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 1920, 1}, {16, 16, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 2560, 1}, {16, 16, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 1280, 1}, {32, 32, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 1920, 1}, {32, 32, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 640, 1}, {32, 32, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {5120, 1, 1, 1}, {1, 256, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {640, 1, 1, 1}, {1, 1, 1, 1});
    //add_test_bin_bcast(GGML_TYPE_F32, {3, 3, 2560, 1280}, {1, 1, 1, 1});
    //add_test_bin_bcast(GGML_TYPE_F32, {3, 3, 2560, 1280}, {2, 1, 1, 1});

    test_cases.emplace_back(new test_scale());

    for (float eps : {1e-6f, 1e-5f, 1e-3f, 1e-1f}) {
        test_cases.emplace_back(new test_norm(GGML_TYPE_F32, {64, 10, 10, 10}, eps));
        test_cases.emplace_back(new test_rms_norm(GGML_TYPE_F32, {64, 10, 10, 10}, eps));
    }

    for (ggml_type type_a : all_types) {
        for (ggml_type type_b : {GGML_TYPE_F32 /*, GGML_TYPE_F16 */}) {
            // FIXME: CPU crashes on f16xf16
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 1,  1}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {10,  1}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {10,  1}, {2, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {10, 10}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {10, 10}, {2, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {10, 10}, {1, 2}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {10, 10}, {2, 2}));

            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 1,  1}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {10,  1}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {10,  1}, {2, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {10, 10}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {10, 10}, {2, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {10, 10}, {1, 2}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {10, 10}, {2, 2}));
        }
    }

    for (ggml_type type_a : all_types) {
        for (ggml_type type_b : {GGML_TYPE_F32 /*, GGML_TYPE_F16 */}) {
            for (int n_mats : {2, 4, 8}) {
                for (int id = 0; id < n_mats; id++) {
                    for (bool v : {false, true}) {
                        test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, n_mats, id, 16, 16, 256, v));
                    }
                }
            }
        }
    }

    test_cases.emplace_back(new test_sqr());
    test_cases.emplace_back(new test_clamp());

    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, {10, 10,  1,  1}, 5));
    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, {10, 10, 10,  1}, 5));
    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, {10, 10, 10, 10}, 5));

    test_cases.emplace_back(new test_soft_max());

    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        test_cases.emplace_back(new test_rope(type, {128,  32, 10, 1}, 128, 0, 512)); // llama 7B
        test_cases.emplace_back(new test_rope(type, {128,  40, 10, 1}, 128, 0, 512)); // llama 13B
        test_cases.emplace_back(new test_rope(type, {128,  52, 10, 1}, 128, 0, 512)); // llama 30B
        test_cases.emplace_back(new test_rope(type, {128,  64, 10, 1}, 128, 0, 512)); // llama 65B
        test_cases.emplace_back(new test_rope(type, { 64,   1, 10, 1},  64, 2, 512)); // neox (falcon 7B)
        test_cases.emplace_back(new test_rope(type, { 64,  71, 10, 1},  64, 2, 512)); // neox (falcon 7B)
        test_cases.emplace_back(new test_rope(type, { 64,   8, 10, 1},  64, 2, 512)); // neox (falcon 40B)
        test_cases.emplace_back(new test_rope(type, { 64, 128, 10, 1},  64, 2, 512)); // neox (falcon 40B)
        test_cases.emplace_back(new test_rope(type, { 80,  32, 10, 1},  20, 2, 512)); // neox (stablelm)
    }

    test_cases.emplace_back(new test_alibi());
    test_cases.emplace_back(new test_im2col());
    test_cases.emplace_back(new test_concat());

    for (ggml_sort_order order : {GGML_SORT_ASC, GGML_SORT_DESC}) {
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {8, 1, 1, 1}, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {16, 10, 10, 10}, order));
    }

    test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, {10, 10, 10, 10}));
    test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, {2, 1, 1, 1}));

#if !defined(__SANITIZE_THREAD__)
    // FIXME: these tests use too much memory with thread sanitizer
    test_cases.emplace_back(new test_moe(8, 2, 1, 4096, 14336));
    //test_cases.emplace_back(new test_moe(8, 2, 8, 4096, 14336));
#endif

    // run tests
    if (mode == MODE_TEST) {
        ggml_backend_t backend_cpu = ggml_backend_cpu_init();

        size_t n_ok = 0;
        for (auto & test : test_cases) {
            if (test->eval(backend, backend_cpu, op_name)) {
                n_ok++;
            }
        }
        printf("  %zu/%zu tests passed\n", n_ok, test_cases.size());

        ggml_backend_free(backend_cpu);

        return n_ok == test_cases.size();
    }

    if (mode == MODE_PERF) {
        for (auto & test : test_cases) {
            test->eval_perf(backend, op_name);
        }
        return true;
    }

    GGML_ASSERT(false);
    return false;
}

static void usage(char ** argv) {
    printf("Usage: %s [mode] [-o op] [-b backend]\n", argv[0]);
    printf("  valid modes are: test (compare with CPU backend for correctness) or perf (performance evaluation)\n");
    printf("  op names are as given by ggml_op_desc()\n");
}

int main(int argc, char ** argv) {
    test_mode mode = MODE_TEST;
    const char * op_name = NULL;
    const char * backend = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "test") == 0) {
            mode = MODE_TEST;
        } else if (strcmp(argv[i], "perf") == 0) {
            mode = MODE_PERF;
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                op_name = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
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

    // enumerate backends
    printf("Testing %zu backends\n\n", ggml_backend_reg_get_count());

    size_t n_ok = 0;

    for (size_t i = 0; i < ggml_backend_reg_get_count(); i++) {
        printf("Backend %zu/%zu (%s)\n", i + 1, ggml_backend_reg_get_count(), ggml_backend_reg_get_name(i));

        if (backend != NULL && strcmp(backend, ggml_backend_reg_get_name(i)) != 0) {
            printf("  Skipping\n");
            n_ok++;
            continue;
        }

        ggml_backend_t backend = ggml_backend_reg_init_backend(i, NULL);
        GGML_ASSERT(backend != NULL);
        printf("  Backend name: %s\n", ggml_backend_name(backend));

        bool ok = test_backend(backend, mode, op_name);

        printf("  Backend %s: ", ggml_backend_name(backend));
        if (ok) {
            printf("\033[1;32mOK\033[0m\n");
            n_ok++;
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }

        printf("\n");

        ggml_backend_free(backend);
    }

    printf("%zu/%zu backends passed\n", n_ok, ggml_backend_reg_get_count());

    if (n_ok != ggml_backend_reg_get_count()) {
        printf("\033[1;31mFAIL\033[0m\n");
        return 1;
    }

    printf("\033[1;32mOK\033[0m\n");
    return 0;
}
