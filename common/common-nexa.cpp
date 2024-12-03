#include "common-nexa.h"

#include <thread>
#include <vector>
#include <string.h>
#include <functional>

#include "ggml.h"
// #include "src/ggml-impl.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "common.h"
#include <cmath>
#include <numeric>

void print_ggml_tensor(const char *name, const struct ggml_tensor *tensor, bool use_backend, int precision) {
    std::vector<float> data(ggml_nelements(tensor));
    if (use_backend) {
        ggml_backend_tensor_get(tensor, data.data(), 0, ggml_nbytes(tensor));
    } else {
        memcpy(data.data(), ggml_get_data_f32(tensor), ggml_nbytes(tensor));
    }

    std::vector<int64_t> shape;
    for (int i = 0; i < GGML_MAX_DIMS && tensor->ne[i] > 1; ++i) shape.push_back(tensor->ne[i]);

    print_ggml_tensor_shape(name, tensor);

    size_t offset = 0;
    std::function<void(size_t, size_t &)> print_recursive = [&](size_t dim, size_t &offset) {
        if (dim == shape.size()) {
            printf("%.*f", precision, data[offset++]);
        } else {
            printf("[ ");
            for (int64_t i = 0; i < shape[dim]; ++i) {
                if (i > 0) printf(dim == shape.size() - 1 ? ", " : ",\n%*s", static_cast<int>(dim) + 1, "");
                print_recursive(dim + 1, offset);
            }
            printf("]");
        }
    };
    print_recursive(0, offset);
    printf("\n");
}

void print_ggml_tensor_shape(const char *name, const struct ggml_tensor *tensor) {
    printf("%s: [ ", name);
    for (int i = 0; i < GGML_MAX_DIMS && tensor->ne[i] > 1; ++i) printf("%d ", static_cast<int>(tensor->ne[i]));
    printf("]\n");
}

bool load_hparams_and_tensors_from_gguf(const std::string &fname, NexaBaseModel &model, bool verbose)
{

    // Initialize GGUF context
    ggml_context *meta = nullptr;
    gguf_init_params params = {true, &meta};
    gguf_context *ctx_gguf = gguf_init_from_file(fname.c_str(), params);

    // Check if GGUF context initialization was successful
    if (!ctx_gguf)
        return fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__), false;

    // Get the number of tensors in the GGUF file
    const int n_tensors = gguf_get_n_tensors(ctx_gguf);
    const int n_tensors_in_model = model.tensor_names.size();
    if (n_tensors_in_model > n_tensors)
    {
        fprintf(stderr, "%s: model tensor_names size (%d) is greater than the number of tensors in the GGUF file (%d)\n", __func__, n_tensors_in_model, n_tensors);
        gguf_free(ctx_gguf);
        return false;
    }

    // Load hyperparameters
    for (const auto &name : model.hparam_names)
    {
        int key = gguf_find_key(ctx_gguf, name.c_str());
        if (key != -1)
        {
            model.hparams[name] = gguf_get_val_i32(ctx_gguf, key);
            if (verbose)
                fprintf(stderr, "%s: loaded hparam '%s' = %d\n", __func__, name.c_str(), std::get<int32_t>(model.hparams[name]));
        }
        else
            return fprintf(stderr, "%s: failed to load hparam '%s'\n", __func__, name.c_str()), gguf_free(ctx_gguf), false;
    }

    // Initialize GGML context for tensor data
    model.ctx_data = ggml_init({(n_tensors_in_model + 1) * ggml_tensor_overhead(), nullptr, true});
    if (!model.ctx_data)
        return fprintf(stderr, "%s: ggml_init() failed\n", __func__), gguf_free(ctx_gguf), false;

    // Open the GGUF file for reading tensor data
    std::ifstream fin(fname, std::ios::binary);
    if (!fin)
        return fprintf(stderr, "%s: cannot open model file for loading tensors\n", __func__), gguf_free(ctx_gguf), false;

    // Create tensor structures in the GGML context
    for (const auto &name : model.tensor_names)
    {
        ggml_tensor *t = ggml_dup_tensor(model.ctx_data, ggml_get_tensor(meta, name.c_str()));
        ggml_set_name(t, name.c_str());
    }

    // Allocate memory for tensors using the specified backend
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx_data, model.backend);

    // Load tensors from the GGUF file
    for (const auto &name : model.tensor_names)
    {
        char *tensor_name = const_cast<char *>(name.c_str());
        if (verbose)
            fprintf(stderr, "%s: loading tensor '%s'\n", __func__, tensor_name);

        ggml_tensor *cur = ggml_get_tensor(model.ctx_data, tensor_name);
        if (!cur)
            return fprintf(stderr, "%s: failed to get tensor %s\n", __func__, tensor_name), gguf_free(ctx_gguf), false;

        int tensor_idx = gguf_find_tensor(ctx_gguf, tensor_name);
        fin.seekg(gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, tensor_idx), std::ios::beg);

        int num_bytes = ggml_nbytes(cur);
        if (ggml_backend_buffer_is_host(model.buffer))
        {
            fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
        }
        else
        {
            std::vector<uint8_t> read_buf(num_bytes);
            fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
            ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
        }

        // mapping tensor name to tensor pointer
        model.tensors[name] = cur;
        if (verbose)
        {
            fprintf(stderr, "%s: mapped tensor ", __func__);
            print_ggml_tensor_shape(tensor_name, cur);
        }
    }

    ggml_free(meta);
    gguf_free(ctx_gguf);
    return true;
}

//
// NexaBaseModel
//

// initialize from gguf file
bool NexaBaseModel::load_from_gguf(const std::string &fname)
{
    init_backend();

    bool verbose = false;
#ifdef NEXA_DEBUG
    verbose = true;
#endif

    if (!load_hparams_and_tensors_from_gguf(fname, *this, verbose))
    {
        NEXA_LOG("failed to load params and tensors");
        return false;
    }

    reserve_memory();

    return true;
}

// Initialize the backend based on available hardware
void NexaBaseModel::init_backend()
{
#ifdef GGML_USE_CUDA
    NEXA_LOG("using CUDA backend");
    backend = ggml_backend_cuda_init(0); // Initialize CUDA on device 0
#endif

#ifdef GGML_USE_METAL
    NEXA_LOG("using Metal backend");
    backend = ggml_backend_metal_init(); // Initialize Metal backend
#endif

    // Fallback to CPU backend if no GPU is available
    if (!backend)
    {
        backend = ggml_backend_cpu_init();
        fprintf(stderr, "%s: using CPU backend\n", __func__);
    }
}

// measure mem requirement and allocate
void NexaBaseModel::reserve_memory()
{
    compute_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    struct ggml_cgraph *gf = build_graph();
    ggml_gallocr_reserve(compute_alloc, gf);
    size_t compute_memory_buffer_size = ggml_gallocr_get_buffer_size(compute_alloc, 0);
    NEXA_LOG("compute allocated memory: %.2f MB", compute_memory_buffer_size / 1024.0 / 1024.0);
}

// set the number of threads
void NexaBaseModel::set_n_threads(int n_threads)
{
    if (n_threads <= 0)
    {
        // if n_threads is not set, use the number of cores
        n_threads = std::thread::hardware_concurrency();
    }

    // Set backend options
    if (ggml_backend_is_cpu(backend))
    {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }

// #ifdef GGML_USE_METAL
//     if (ggml_backend_is_metal(backend))
//     {
//         ggml_backend_metal_set_n_cb(backend, n_threads);
//     }
// #endif
}

// Free allocated memory
void NexaBaseModel::free()
{
    ggml_gallocr_free(compute_alloc);
    ggml_free(ctx_data);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
}

void print_ggml_tensor_stats(const char *name, const struct ggml_tensor *tensor, bool use_backend) {
    std::vector<float> data(ggml_nelements(tensor));
    if (use_backend) {
        ggml_backend_tensor_get(tensor, data.data(), 0, ggml_nbytes(tensor));
    } else {
        memcpy(data.data(), ggml_get_data_f32(tensor), ggml_nbytes(tensor));
    }

    if (data.empty()) {
        printf("%s: Empty tensor\n", name);
        return;
    }

    // Calculate mean
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / data.size();

    // Calculate variance using two-pass algorithm for better numerical stability
    double sq_sum = 0.0;
    for (const auto &val : data) {
        double diff = val - mean;
        sq_sum += diff * diff;
    }
    double variance = sq_sum / data.size();

    // Print statistics
    printf("%s:\n", name);
    printf("  Shape: [");
    for (int i = 0; i < GGML_MAX_DIMS && tensor->ne[i] > 1; ++i) {
        printf("%d%s", static_cast<int>(tensor->ne[i]), (i < GGML_MAX_DIMS - 1 && tensor->ne[i+1] > 1) ? ", " : "");
    }
    printf("]\n");
    printf("  Mean: %.6f\n", mean);
    printf("  Variance: %.6f\n", variance);
    printf("  Standard Deviation: %.6f\n", std::sqrt(variance));
}

void print_all_tensor_names(struct gguf_context *ctx) {
    int n_tensors = gguf_get_n_tensors(ctx);
    printf("Number of tensors: %d\n", n_tensors);

    const char *separator = "";
    printf("Tensors: ");
    for (int i = 0; i < n_tensors; ++i) {
        const char *tensor_name = gguf_get_tensor_name(ctx, i);
        printf("%s%s", separator, tensor_name);
        separator = ", ";  // Set separator after the first tensor
    }
    printf("\n");
}

struct ggml_tensor * checked_get_tensor(struct ggml_context * ctx, const char * name) {
    struct ggml_tensor * tensor = ggml_get_tensor(ctx, name);
    // print_ggml_tensor_stats(name, tensor, false);
    if (!tensor) {
        fprintf(stderr, "%s: tensor '%s' not found\n", __func__, name);
        throw std::runtime_error("ggml_get_tensor() failed");
    }
    return tensor;
}

// //
// // original ggml functions
// //
//
// struct ggml_tensor * ggml_graph_node(struct ggml_cgraph * cgraph, int i) {
//     if (i < 0) {
//         GGML_ASSERT(cgraph->n_nodes + i >= 0);
//         return cgraph->nodes[cgraph->n_nodes + i];
//     }
//
//     GGML_ASSERT(i < cgraph->n_nodes);
//     return cgraph->nodes[i];
// }
