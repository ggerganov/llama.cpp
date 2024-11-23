#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include <algorithm>
#include <cstring>
#include <vector>

// Backend registry
#ifdef GGML_USE_CPU
#include "ggml-cpu.h"
#endif

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

struct ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    void * handle;
};

struct ggml_backend_registry {
    std::vector<ggml_backend_reg_entry> backends;
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
#ifdef GGML_USE_RPC
        register_backend(ggml_backend_rpc_reg());
#endif
#ifdef GGML_USE_AMX
        register_backend(ggml_backend_amx_reg());
#endif
#ifdef GGML_USE_KOMPUTE
        register_backend(ggml_backend_kompute_reg());
#endif
#ifdef GGML_USE_CPU
        register_backend(ggml_backend_cpu_reg());
#endif
    }

    ~ggml_backend_registry() {
        while (!backends.empty()) {
            ggml_backend_unload(backends.back().reg);
        }
    }

    void register_backend(ggml_backend_reg_t reg, void * handle = nullptr) {
        if (!reg) {
            return;
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
#endif
        backends.push_back({ reg, handle });
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
    return get_reg().backends[index].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(const char * name) {
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (std::strcmp(ggml_backend_reg_name(reg), name) == 0) {
            return reg;
        }
    }
    return nullptr;
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
    return nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == type) {
            return dev;
        }
    }
    return nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, nullptr);
}

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

typedef ggml_backend_reg_t (*ggml_backend_init_t)(void);

ggml_backend_reg_t ggml_backend_load(const char * path) {
#ifdef _WIN32
    HMODULE handle = LoadLibraryA(path);
    if (!handle) {
        GGML_LOG_ERROR("%s: failed to load %s: %lu\n", __func__, path, GetLastError());
        return nullptr;
    }
    ggml_backend_init_t backend_init = (ggml_backend_init_t) GetProcAddress(handle, "ggml_backend_init");
    if (!backend_init) {
        GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s: %lu\n", __func__, path, GetLastError());
        FreeLibrary(handle);
        return nullptr;
    }
#else
    void * handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path, dlerror());
        return nullptr;
    }
    auto * backend_init = (ggml_backend_init_t) dlsym(handle, "ggml_backend_init");
    if (!backend_init) {
        GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s: %s\n", __func__, path, dlerror());
        dlclose(handle);
        return nullptr;
    }
#endif
    ggml_backend_reg_t reg = backend_init();
    if (!reg) {
        GGML_LOG_ERROR("%s: failed to initialize backend from %s\n", __func__, path);
        dlclose(handle);
        return nullptr;
    }
    GGML_LOG_DEBUG("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path);
    get_reg().register_backend(reg, handle);
    return reg;
}

void ggml_backend_unload(ggml_backend_reg_t reg) {
    auto it = std::find_if(get_reg().backends.begin(), get_reg().backends.end(),
                           [reg](ggml_backend_reg_entry entry) { return entry.reg == reg; });

    if (it == get_reg().backends.end()) {
        GGML_LOG_ERROR("%s: backend not found\n", __func__);
        return;
    }

    GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));

    // remove devices
    get_reg().devices.erase(
        std::remove_if(get_reg().devices.begin(), get_reg().devices.end(),
                       [reg](ggml_backend_dev_t dev) { return ggml_backend_dev_backend_reg(dev) == reg; }),
        get_reg().devices.end());

    // unload library
    if (it->handle) {
#ifdef _WIN32
        FreeLibrary((HMODULE) it->handle);
#else
        dlclose(it->handle);
#endif
    }

    // remove backend
    get_reg().backends.erase(it);
}

void ggml_backend_load_all() {
#ifdef _WIN32
    #define GGML_BACKEND_PATH(backend) "ggml-" backend ".dll"
#elif defined(__APPLE__)
    // path is hardcoded to the cmake build directory for now
    // FIXME: should also search default system paths
    #define GGML_BACKEND_PATH(backend) "build/ggml/src/ggml-" backend "/libggml-" backend ".dylib"
#else
    #define GGML_BACKEND_PATH(backend) "build/ggml/src/ggml-" backend "/libggml-" backend ".so"
#endif

    ggml_backend_load(GGML_BACKEND_PATH("amx"));
    ggml_backend_load(GGML_BACKEND_PATH("blas"));
    ggml_backend_load(GGML_BACKEND_PATH("cann"));
    ggml_backend_load(GGML_BACKEND_PATH("cuda"));
    ggml_backend_load(GGML_BACKEND_PATH("hip"));
    ggml_backend_load(GGML_BACKEND_PATH("kompute"));
    ggml_backend_load(GGML_BACKEND_PATH("metal"));
    ggml_backend_load(GGML_BACKEND_PATH("rpc"));
    ggml_backend_load(GGML_BACKEND_PATH("sycl"));
    ggml_backend_load(GGML_BACKEND_PATH("vulkan"));
    ggml_backend_load(GGML_BACKEND_PATH("musa"));
    ggml_backend_load(GGML_BACKEND_PATH("cpu"));
}
