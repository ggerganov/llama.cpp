#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include <cstring>
#include <vector>

// Backend registry

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef GGML_USE_SYCL
#include "ggml-sycl.h"
#endif

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef GGML_USE_BLAS
#include "ggml-blas.h"
#endif

#ifdef GGML_USE_TINYBLAS
#include "ggml-tinyblas.h"
#endif

#ifdef GGML_USE_RPC
#include "ggml-rpc.h"
#endif

#ifdef GGML_USE_AMX
#  include "ggml-amx.h"
#endif

#ifdef GGML_USE_CANN
#include "ggml-cann.h"
#endif

#ifdef GGML_USE_KOMPUTE
#include "ggml-kompute.h"
#endif

struct ggml_backend_registry {
    std::vector<ggml_backend_reg_t> backends;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_registry() {
#ifdef GGML_USE_CUDA
        register_backend(ggml_backend_cuda_reg());
#endif
#ifdef GGML_USE_METAL
        register_backend(ggml_backend_metal_reg());
#endif
#ifdef GGML_USE_SYCL
        register_backend(ggml_backend_sycl_reg());
#endif
#ifdef GGML_USE_VULKAN
        register_backend(ggml_backend_vk_reg());
#endif
#ifdef GGML_USE_CANN
        register_backend(ggml_backend_cann_reg());
#endif
#ifdef GGML_USE_BLAS
        register_backend(ggml_backend_blas_reg());
#endif
#ifdef GGML_USE_TINYBLAS
        register_backend(ggml_backend_tinyblas_reg());
#endif
#ifdef GGML_USE_RPC
        register_backend(ggml_backend_rpc_reg());
#endif
#ifdef GGML_USE_AMX
        register_backend(ggml_backend_amx_reg());
#endif
#ifdef GGML_USE_KOMPUTE
        register_backend(ggml_backend_kompute_reg());
#endif

        register_backend(ggml_backend_cpu_reg());
    }

    void register_backend(ggml_backend_reg_t reg) {
        if (!reg) {
            return;
        }

        GGML_LOG_INFO("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
//#ifndef NDEBUG
//        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
//            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
//#endif
        backends.push_back(reg);
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    void register_device(ggml_backend_dev_t device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
        devices.push_back(device);
    }
};

static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
    return reg;
}

// Internal API
void ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

void ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
size_t ggml_backend_reg_count() {
    return get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_reg_count());
    return get_reg().backends[index];
}

ggml_backend_reg_t ggml_backend_reg_by_name(const char * name) {
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (std::strcmp(ggml_backend_reg_name(reg), name) == 0) {
            return reg;
        }
    }
    return NULL;
}

// Device enumeration
size_t ggml_backend_dev_count() {
    return get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_dev_count());
    return get_reg().devices[index];
}

ggml_backend_dev_t ggml_backend_dev_by_name(const char * name) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (strcmp(ggml_backend_dev_name(dev), name) == 0) {
            return dev;
        }
    }
    return NULL;
}

ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == type) {
            return dev;
        }
    }
    return NULL;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    if (!dev) {
        return NULL;
    }
    return ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    if (!dev) {
        return NULL;
    }
    return ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (!dev) {
        return NULL;
    }
    return ggml_backend_dev_init(dev, NULL);
}
