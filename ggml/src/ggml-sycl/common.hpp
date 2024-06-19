//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_COMMON_HPP
#define GGML_SYCL_COMMON_HPP

#include <fstream>
#include <iostream>

#include "dpct/helper.hpp"
#include "ggml-sycl.h"
#include "presets.hpp"

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

void* ggml_sycl_host_malloc(size_t size);
void ggml_sycl_host_free(void* ptr);

static int g_ggml_sycl_debug = 0;
#define GGML_SYCL_DEBUG(...)        \
  do {                              \
    if (g_ggml_sycl_debug)          \
      fprintf(stderr, __VA_ARGS__); \
  } while (0)

#define CHECK_TRY_ERROR(expr)                                            \
  [&]() {                                                                \
    try {                                                                \
      expr;                                                              \
      return dpct::success;                                              \
    } catch (std::exception const& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::endl;                                            \
      return dpct::default_error;                                        \
    }                                                                    \
  }()

// #define DEBUG_SYCL_MALLOC

static int g_work_group_size = -1;
// typedef sycl::half ggml_fp16_t;

#define __SYCL_ARCH__ DPCT_COMPATIBILITY_TEMP
#define VER_4VEC 610 // todo for hardward optimize.
#define VER_GEN9 700 // todo for hardward optimize.
#define VER_GEN12 1000000 // todo for hardward optimize.
#define VER_GEN13 (VER_GEN12 + 1030) // todo for hardward optimize.

#define GGML_SYCL_MAX_NODES 8192 // TODO: adapt to hardwares

// define for XMX in Intel GPU
// TODO: currently, it's not used for XMX really.
#if !defined(GGML_SYCL_FORCE_MMQ)
    #define SYCL_USE_XMX
#endif

// max batch size to use MMQ kernels when tensor cores are available
#define MMQ_MAX_BATCH_SIZE 32

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

// dmmv = dequantize_mul_mat_vec
#ifndef GGML_SYCL_DMMV_X
#define GGML_SYCL_DMMV_X 32
#endif
#ifndef GGML_SYCL_MMV_Y
#define GGML_SYCL_MMV_Y 1
#endif

typedef sycl::queue *queue_ptr;

enum ggml_sycl_backend_gpu_mode {
  SYCL_UNSET_GPU_MODE = -1,
  SYCL_SINGLE_GPU_MODE = 0,
  SYCL_MUL_GPU_MODE
};

enum ggml_sycl_backend_device_filter {
  SYCL_DEVICE_FILTER_ALL = 0,
  SYCL_DEVICE_FILTER_TOP_LEVEL_ZERO
};

static_assert(sizeof(sycl::half) == sizeof(ggml_fp16_t), "wrong fp16 size");

static void crash() {
  int* ptr = NULL;
  *ptr = 0;
}

[[noreturn]] static void ggml_sycl_error(
    const char* stmt,
    const char* func,
    const char* file,
    const int line,
    const char* msg) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, msg);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, line);
  GGML_ASSERT(!"SYCL error");
}

#define SYCL_CHECK(err)                     \
  do {                                      \
    auto err_ = (err);                      \
    if (err_ != 0)                          \
      ggml_sycl_error(                      \
          #err,                             \
          __func__,                         \
          __FILE__,                         \
          __LINE__,                         \
          "Meet error in this line code!"); \
  } while (0)

#if DPCT_COMPAT_RT_VERSION >= 11100
#define GGML_SYCL_ASSUME(x) __builtin_assume(x)
#else
#define GGML_SYCL_ASSUME(x)
#endif // DPCT_COMPAT_RT_VERSION >= 11100

#ifdef GGML_SYCL_F16
typedef sycl::half dfloat; // dequantize float
typedef sycl::half2 dfloat2;
#else
typedef float dfloat; // dequantize float
typedef sycl::float2 dfloat2;
#endif // GGML_SYCL_F16

#define MMVQ_MAX_BATCH_SIZE  8

static const int8_t kvalues_iq4nl[16]={-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static int g_all_sycl_device_count = -1;
static bool g_ggml_backend_sycl_buffer_type_initialized = false;

static ggml_sycl_backend_gpu_mode g_ggml_sycl_backend_gpu_mode =
    SYCL_UNSET_GPU_MODE;

static void* g_scratch_buffer = nullptr;
static size_t g_scratch_size = 0; // disabled by default
static size_t g_scratch_offset = 0;

[[noreturn]] static inline void bad_arch(const sycl::stream& stream_ct1) {
  stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::exit(1);

  (void)bad_arch; // suppress unused function warning
}

int get_current_device_id();

inline dpct::err0 ggml_sycl_set_device(const int device_id) try {

  int current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d, current_device_id=%d\n", device_id, current_device_id);
  if (device_id == current_device_id) {
    return 0;
  }

  return CHECK_TRY_ERROR(dpct::select_device(device_id));

} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  crash();
  std::exit(1);
}

class sycl_device_mgr {
  public:
    std::vector<int> device_ids;
    std::vector<sycl::device> devices;
    std::vector<int> max_compute_units;
    std::vector<int> work_group_sizes;
    sycl::queue *first_queue;
    std::vector<sycl::queue *> queues;
    std::vector<sycl::context> ctxs;
    std::string device_list = "";

    sycl_device_mgr(ggml_sycl_backend_device_filter device_filter) {
        if (device_filter == SYCL_DEVICE_FILTER_TOP_LEVEL_ZERO) {
            detect_sycl_gpu_list_with_max_cu();
            create_context_for_group_gpus();
        } else {
            detect_all_sycl_device_list();
            create_context_queue_for_devices();
        }
        get_allow_devices();
    }

    /*
    Bind all gpus in same host with same context, for better performance in
    device-to-device copy in the future.
    */
    void create_context_for_group_gpus() {
        sycl::context ctx = sycl::context(devices);
        assert(device_ids.size() > 0);
        first_queue = dpct::get_current_device().create_queue(ctx, devices[0]);
        sycl::context ctx0 = first_queue->get_context();
        for (int i = 0; i < device_ids.size(); i++) {
            ctxs.push_back(ctx0);
            dpct::select_device(device_ids[i]);
            queues.push_back(
                dpct::get_current_device().create_queue(ctx0, devices[i]));
        }
    }

    void create_context_queue_for_devices() {
        for (int i = 0; i < device_ids.size(); i++) {
            sycl::context ctx = sycl::context(devices[i]);
            ctxs.push_back(ctx);
            dpct::select_device(device_ids[i]);
            queues.push_back(
                dpct::get_current_device().create_queue(ctx, devices[i]));
        }
    }

    void get_allow_devices() {
        device_list = "";
        for (size_t i = 0; i < device_ids.size(); ++i) {
            device_list += std::to_string(device_ids[i]);
            device_list += ",";
        }
        if (device_list.length() > 1) {
            device_list.pop_back();
        }
    }

    bool is_allowed_device(int device_id) {
        return std::find(device_ids.begin(), device_ids.end(), device_id) != device_ids.end();
    }

    void detect_all_sycl_device_list() try {
        int device_count = dpct::dev_mgr::instance().device_count();

        for (int id = 0; id < device_count; id++) {
            sycl::device device = dpct::dev_mgr::instance().get_device(id);
            device_ids.push_back(id);
            devices.push_back(device);
            dpct::device_info prop;
            dpct::get_device_info(prop, device);
            work_group_sizes.push_back(prop.get_max_work_group_size());
            max_compute_units.push_back(prop.get_max_compute_units());
        }
        return;
    } catch (sycl::exception const &exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                  << ", line:" << __LINE__ << std::endl;
        std::exit(1);
    }

    /*
    Use all GPUs with same top max compute units
    */
    void detect_sycl_gpu_list_with_max_cu() try {
        int device_count = dpct::dev_mgr::instance().device_count();
        int local_max_compute_units = 0;
        for (int id = 0; id < device_count; id++) {
            sycl::device device = dpct::dev_mgr::instance().get_device(id);
            if (!device.is_gpu())
                continue;
            dpct::device_info prop;
            dpct::get_device_info(prop, device);
            if (local_max_compute_units < prop.get_max_compute_units())
                local_max_compute_units = prop.get_max_compute_units();
        }

        for (int id = 0; id < device_count; id++) {
            sycl::device device = dpct::dev_mgr::instance().get_device(id);
            if (!device.is_gpu())
                continue;
            dpct::device_info prop;
            dpct::get_device_info(prop, device);
            if (local_max_compute_units == prop.get_max_compute_units() &&
                is_ext_oneapi_device(device)) {
                device_ids.push_back(id);
                devices.push_back(device);
                work_group_sizes.push_back(prop.get_max_work_group_size());
                max_compute_units.push_back(prop.get_max_compute_units());
            }
        }
        return;
    } catch (sycl::exception const &exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                  << ", line:" << __LINE__ << std::endl;
        std::exit(1);
    }

    int get_device_count() { return (int)device_ids.size(); }

    bool is_ext_oneapi_device(const sycl::device &dev) {
        sycl::backend dev_backend = dev.get_backend();
        if (dev_backend == sycl::backend::ext_oneapi_level_zero ||
            dev_backend == sycl::backend::ext_oneapi_cuda ||
            dev_backend == sycl::backend::ext_oneapi_hip)
            return true;
        return false;
    }
};

struct ggml_sycl_device_info {
    int device_count;
    int main_gpu_id = -1;
    ggml_sycl_backend_gpu_mode use_gpu_mode = SYCL_MUL_GPU_MODE;
    struct sycl_device_info {
        int cc; // compute capability
        // int     nsm;                // number of streaming multiprocessors
        // size_t  smpb;               // max. shared memory per block
        bool vmm; // virtual memory support
        size_t total_vram;
    };

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<float, GGML_SYCL_MAX_DEVICES> default_tensor_split = {};

    sycl_device_mgr *local_sycl_device_mgr = NULL;

    void print_gpu_device_list() {
        GGML_ASSERT(local_sycl_device_mgr);

        char *hint = NULL;
        if (use_gpu_mode == SYCL_MUL_GPU_MODE) {
            hint = "detect %d SYCL GPUs: [%s] with top Max compute units:%d\n";
            fprintf(stderr, hint, local_sycl_device_mgr->get_device_count(),
                    local_sycl_device_mgr->device_list.c_str(),
                    local_sycl_device_mgr->max_compute_units[main_gpu_id]);
        } else {
            hint = "use main device [%d] with Max compute units:%d\n";
            fprintf(stderr, hint, main_gpu_id,
                    local_sycl_device_mgr->max_compute_units[main_gpu_id]);
        }
    }

    int work_group_size(int device_id) {
        GGML_ASSERT(local_sycl_device_mgr);
        return local_sycl_device_mgr->work_group_sizes[device_id];
    }

    void refresh_device(ggml_sycl_backend_gpu_mode gpu_model,
                        int p_main_gpu_id = 0) {
        main_gpu_id = p_main_gpu_id;
        use_gpu_mode = gpu_model;
        if (!local_sycl_device_mgr)
            delete local_sycl_device_mgr;

        if (use_gpu_mode == SYCL_MUL_GPU_MODE) {
            local_sycl_device_mgr =
                new sycl_device_mgr(SYCL_DEVICE_FILTER_TOP_LEVEL_ZERO);
        } else {
            GGML_ASSERT(main_gpu_id >= 0);
            local_sycl_device_mgr = new sycl_device_mgr(SYCL_DEVICE_FILTER_ALL);
        }

        device_count = local_sycl_device_mgr->get_device_count();

        int64_t total_vram = 0;

        for (int i = 0; i < device_count; ++i) {
            devices[i].vmm = 0;
            dpct::device_info prop;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                prop, dpct::dev_mgr::instance().get_device(i))));

            default_tensor_split[i] = total_vram;
            total_vram += prop.get_global_mem_size();

            devices[i].cc =
                100 * prop.get_major_version() + 10 * prop.get_minor_version();
        }

        for (int id = 0; id < device_count; ++id) {
            default_tensor_split[id] /= total_vram;
        }

        g_work_group_size = work_group_size(main_gpu_id);
        print_gpu_device_list();
    }

};

struct ggml_sycl_pool {
    virtual ~ggml_sycl_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

template<typename T>
struct ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    explicit ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T * get() {
        return ptr;
    }

    ggml_sycl_pool_alloc() = default;
    ggml_sycl_pool_alloc(const ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = delete;
    ggml_sycl_pool_alloc& operator=(const ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc& operator=(ggml_sycl_pool_alloc &&) = delete;
};

// backend interface

struct ggml_tensor_extra_gpu {
  void* data_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
};

struct ggml_backend_sycl_context {
    int device;
    std::string name;

    queue_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_sycl_context(struct ggml_sycl_device_info &sycl_device_info, int device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
            qptrs[device][0] = sycl_device_info.local_sycl_device_mgr->queues[device];
    }

    queue_ptr stream(int device, int stream) {
        assert(qptrs[device][0] != nullptr);
        return qptrs[device][0];
    }

    queue_ptr stream() {
        return stream(device, 0);
    }

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    static std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, int device);

    ggml_sycl_pool & pool(int device) {
        if (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        return *pools[device];
    }

    ggml_sycl_pool & pool() {
        return pool(device);
    }
};

// common host functions

static inline int get_work_group_size(const sycl::device& device) {
    dpct::device_info prop;
    dpct::get_device_info(prop, device);
    return prop.get_max_work_group_size();
}


// common device functions

static __dpct_inline__ float warp_reduce_sum(float x,
    const sycl::nd_item<3>& item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        /*
        DPCT1096:98: The right-most dimension of the work-group used in the SYCL
        kernel that calls this function may be less than "32". The function
        "dpct::permute_sub_group_by_xor" may return an unexpected result on the
        CPU device. Modify the size of the work-group to ensure that the value
        of the right-most dimension is a multiple of "32".
        */
        x += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask);
    }
    return x;
}

static __dpct_inline__ sycl::float2
warp_reduce_sum(sycl::float2 a, const sycl::nd_item<3>& item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        a.x() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.x(),
            mask);
        a.y() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.y(),
            mask);
    }
    return a;
}

static __dpct_inline__ float warp_reduce_max(float x,
    const sycl::nd_item<3>& item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        /*
        DPCT1096:97: The right-most dimension of the work-group used in the SYCL
        kernel that calls this function may be less than "32". The function
        "dpct::permute_sub_group_by_xor" may return an unexpected result on the
        CPU device. Modify the size of the work-group to ensure that the value
        of the right-most dimension is a multiple of "32".
        */
        x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
            item_ct1.get_sub_group(), x, mask));
    }
    return x;
}

#endif // GGML_SYCL_COMMON_HPP
