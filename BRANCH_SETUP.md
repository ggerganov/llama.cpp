# Setup this branch

## Create a lora adpter bin file

0. `mkdir models/open-llama` and download [Open-llama  (all files)](https://huggingface.co/openlm-research/open_llama_3b_v2/tree/main) in the folder `./models/open-llama`

2. `mkdir data && touch data/hot-lora.txt` and write a couple of words in it.

3. Run:
    ```bash
    # Convert base model to gguf
    python3 convert-hf-to-gguf.py models/open-llama/
    # Quantize base model
    ./quantize ./models/open-llama/ggml-model-f16.gguf ./models/open-llama/ggml-model-q8_0.gguf Q8_0
    # Obtain Lora adapter
    ./finetune  --model-base models/open-llama/ggml-model-q8_0.gguf \
    --checkpoint-in models/open-llama/chk-lora-ggml-model-q8_0-hot-lora-LATEST.gguf \
    --checkpoint-out models/open-llama/chk-lora-ggml-model-q8_0-hot-lora-ITERATION.gguf \
    --lora-out models/open-llama/lora-ggml-model-q8_0-hot-lora-ITERATION.bin \
    --train-data "data/hot-lora.txt" \
    --save-every 1 \
    --threads 1 \
    --adam-iter 1 \
    --batch 1 \
    --ctx 16 \
    --use-checkpointing
    ```

## Run main with adapter

Run main with base model and lora adapter to hot-swap
```bash
./main -m ./models/open-llama/ggml-model-f16.gguf \
--hot-lora models/open-llama/lora-ggml-model-q8_0-hot-lora-LATEST.bin \
-ngl 0 \
-n 128
```

Working but `ggml_metal_get_buffer: error: tensor 'blk.16.attn_v.weight.loraB' buffer is nil`

With `ngl > 0` the code breaks. Probably because the Lora tensors try to interact with the base tensors (as in `lora_mul_mat`), but the lora tensors are not moved to the gpu buffer of the base tensors.

# Logic




# Current status

- Only one Lora adapter can be passed. 
- Applying only adapter to Q, K, V matrices to keep the code contained (fintuning trained lora tensors for all linear layers)
- GPU not supported




# Tutorial

```cpp
#include "llama.h"

#include "unicode.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_RPC
#  include "ggml-rpc.h"
#endif

#ifdef GGML_USE_CUDA
#  include "ggml-cuda.h"
#elif defined(GGML_USE_VULKAN)
#  include "ggml-vulkan.h"
#elif defined(GGML_USE_SYCL)
#  include "ggml-sycl.h"
#elif defined(GGML_USE_KOMPUTE)
#   include "ggml-kompute.h"
#endif

#ifdef GGML_USE_METAL
#  include "ggml-metal.h"
#endif

// TODO: replace with ggml API call
#define QK_K 256

#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/mman.h>
            #include <fcntl.h>
        #endif
        #if defined(_POSIX_MEMLOCK_RANGE)
            #include <sys/resource.h>
        #endif
    #endif
#endif

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cfloat>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <forward_list>
#include <fstream>
#include <functional>
#include <future>
#include <initializer_list>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include "ggml-metal.h"

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#ifdef __GNUC__
#ifdef __MINGW32__
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define LLAMA_ATTRIBUTE_FORMAT(...)
#endif

#define LLAMA_MAX_NODES   8192
#define LLAMA_MAX_EXPERTS 160

  
int main() {
    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
        /*.no_alloc   =*/ true,
    };

    // The library allows the user to define a certain function using the available tensor operations. This function
    // definition is represented internally via a computation graph. Each tensor operation in the function definition
    // corresponds to a node in the graph. Having the computation graph defined, the user can choose to compute the
    // function's value and/or its gradient with respect to the input variables. Optionally, the function can be optimized
    // using one of the available optimization algorithms.
    //
    // For example, here we define the function: f(x) = a*x^2 + b    

    // memory allocation happens here
    // Create context allogating memory
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_param(ctx, x); // x is an input variable

    struct ggml_tensor * a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * x2 = ggml_mul(ctx, x, x);
    struct ggml_tensor * f  = ggml_add(ctx, ggml_mul(ctx, a, x2), b);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    // ggml_backend_alloc_ctx_tensors_from_buft(ctx,  ggml_backend_metal_buffer_type());
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_metal_buffer_type());
            if (buf == nullptr) {
                throw std::runtime_error("unable to allocate backend buffer");
            }
    ggml_used_mem(ctx);

    // llama_default_buffer_type_offload(model, layer_gpu); used in llama.cpp
    // How to check which buffer is the context allocated, 
    // can look at single tensors? option, check in inited in base model

    // Try this
    // You can simplify all of this for testing, and if you are using CPU only, and just run with -ngl 0 
    // and allocate everything in a CPU buffer by using 
    //  ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    // or run with -ngl 99 and use a Metal buffer type instead with 
    //  ggml_backend_metal_buffer_type()
    // It will still run if you allocate the tensors in the wrong buffer type as long as you use ggml-backend 
    // to allocate the tensors, it will just be slower.

    // Notice that the function definition above does not involve any actual computation. The computation is performed only
    // when the user explicitly requests it. For example, to compute the function's value at x = 2.0:


    ggml_build_forward_expand(gf, f);

    // set the input variable and parameter values
    ggml_set_f32(x, 2.0f);
    ggml_set_f32(a, 3.0f);
    ggml_set_f32(b, 4.0f);

    ggml_graph_compute_with_ctx(ctx, gf, 1);

    printf("f = %f\n", ggml_get_f32_1d(f, 0));

    // The actual computation is performed in the ggml_graph_compute() function.
    //
    // The ggml_new_tensor_...() functions create new tensors. They are allocated in the memory buffer provided to the
    // ggml_init() function. You have to be careful not to exceed the memory buffer size. Therefore, you have to know
    // in advance how much memory you need for your computation. Alternatively, you can allocate a large enough memory
    // and after defining the computation graph, call the ggml_used_mem() function to find out how much memory was
    // actually needed.
    //
    // The ggml_set_param() function marks a tensor as an input variable. This is used by the automatic
    // differentiation and optimization algorithms.
    //
    // The described approach allows to define the function graph once and then compute its forward or backward graphs
    // multiple times. All computations will use the same memory buffer allocated in the ggml_init() function. This way
    // the user can avoid the memory allocation overhead at runtime.
    //
    // The library supports multi-dimensional tensors - up to 4 dimensions. The FP16 and FP32 data types are first class
    // citizens, but in theory the library can be extended to support FP8 and integer data types.
    //
    // Each tensor operation produces a new tensor. Initially the library was envisioned to support only the use of unary
    // and binary operations. Most of the available operations fall into one of these two categories. With time, it became
    // clear that the library needs to support more complex operations. The way to support these operations is not clear
    // yet, but a few examples are demonstrated in the following operations:
    //
    //   - ggml_permute()
    //   - ggml_conv_1d_1s()
    //   - ggml_conv_1d_2s()
    //
    // For each tensor operator, the library implements a forward and backward computation function. The forward function
    // computes the output tensor value given the input tensor values. The backward function computes the adjoint of the
    // input tensors given the adjoint of the output tensor. For a detailed explanation of what this means, take a
    // calculus class, or watch the following video:
    //
    //   What is Automatic Differentiation?
    //   https://www.youtube.com/watch?v=wG_nF1awSSY

    // ## Tensor data (struct ggml_tensor)
    //
    // The tensors are stored in memory via the ggml_tensor struct. The structure provides information about the size of
    // the tensor, the data type, and the memory buffer where the tensor data is stored. Additionally, it contains
    // pointers to the "source" tensors - i.e. the tensors that were used to compute the current tensor. For example:
    
    struct ggml_tensor * c = ggml_add(ctx, a, b);

    assert(c->src[0] == a);
    assert(c->src[1] == b);

    // The multi-dimensional tensors are stored in row-major order. The ggml_tensor struct contains fields for the
    // number of elements in each dimension ("ne") as well as the number of bytes ("nb", a.k.a. stride). This allows
    // to store tensors that are not contiguous in memory, which is useful for operations such as transposition and
    // permutation. All tensor operations have to take the stride into account and not assume that the tensor is
    // contiguous in memory.
    
    // The data of the tensor is accessed via the "data" pointer. For example:

    const int nx = 2;
    const int ny = 3;

    struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nx, ny);

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            *(float *) ((char *) A->data + y*A->nb[1] + x*A->nb[0]) = x + y;
        }
    }

    //
    // Alternatively, there are helper functions, such as ggml_get_f32_1d() and ggml_set_f32_1d() that can be used.
    //

  }
  ```