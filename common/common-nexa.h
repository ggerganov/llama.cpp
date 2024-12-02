#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <vector>
#include <variant>
#include <cmath>

// Replace the cxxabi.h include and NEXA_CLASS_NAME definition with cross-platform version
#ifdef _MSC_VER
    // Windows/MSVC version
    #include <typeinfo>
    #define NEXA_CLASS_NAME (typeid(*this).name())
#else
    // Unix/GCC/Clang version
    #include <cxxabi.h>
    #define NEXA_CLASS_NAME (abi::__cxa_demangle(typeid(*this).name(), nullptr, nullptr, nullptr))
#endif

#define NEXA_LOG(fmt, ...) fprintf(stderr, "%s::%s: " fmt "\n", NEXA_CLASS_NAME, __func__, ##__VA_ARGS__)

// Prints the content of a ggml_tensor with specified precision. Can use the backend if available.
void print_ggml_tensor(const char *name, const struct ggml_tensor *tensor, bool use_backend, int precision = 4);

// Prints the shape (dimensions) of a ggml_tensor without printing its contents.
void print_ggml_tensor_shape(const char *name, const struct ggml_tensor *tensor);

// Prints the statistics (mean, min, max, std) of a ggml_tensor. Can use the backend if available.
void print_ggml_tensor_stats(const char *name, const struct ggml_tensor *tensor, bool use_backend);

// Print all tensor names in the provided GGUF context.
void print_all_tensor_names(struct gguf_context *ctx);

// get tensor, print stats and check for null
struct ggml_tensor * checked_get_tensor(struct ggml_context * ctx, const char * name);

// Base class for all Nexa models

struct NexaBaseModel
{
    std::vector<std::string> hparam_names;
    std::map<std::string, std::variant<int32_t, float_t>> hparams; // hyperparameters, dict value can be either int32_t or float_t

    std::vector<std::string> tensor_names;
    std::map<std::string, struct ggml_tensor *> tensors; // std::variant is a type-safe union that can hold either: (1) int32_t (32-bit integer) (2) float_t (floating-point number)

    struct ggml_context *ctx_data; // GGML context for tensor management

    ggml_backend_buffer_t buffer; // Backend buffer to store tensor data

    ggml_backend_t backend = NULL;       // Backend for computation (CPU, CUDA, METAL)
    ggml_gallocr_t compute_alloc = NULL; // Memory allocator for computation

    // constructor & destructor
    NexaBaseModel() {}
    ~NexaBaseModel()
    {
        free();
        // NEXA_LOG("allocated resources freed");
    }

    // Initialize the backend based on available hardware
    void init_backend();

    // measure mem requirement and allocate
    void reserve_memory();

    // initialize from gguf file
    bool load_from_gguf(const std::string &fname);

    // build the computation graph
    // this is a pure virtual function that must be implemented by the derived class
    virtual ggml_cgraph *build_graph() = 0;

    // set the number of threads
    void set_n_threads(int n_threads);

    // Free allocated memory
    void free();
};

bool load_hparams_and_tensors_from_gguf(const std::string &fname, NexaBaseModel &model, bool verbose = false);

struct ggml_tensor * ggml_graph_node(struct ggml_cgraph * cgraph, int i);
