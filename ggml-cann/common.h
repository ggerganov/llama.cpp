#pragma once

#include <acl/acl.h>

#include <cstdio>
#include <iostream>
#include <string>

#include "../ggml-cann.h"
#include "../ggml.h"

#define MATRIX_ROW_PADDING 512
#define GGML_CANN_MAX_STREAMS 8

[[noreturn]] void ggml_cann_error(const char* stmt, const char* func,
                                  const char* file, int line, const char* msg);

// Error handling macro
#define ACL_CHECK_GEN(stmt, success, error_fn)                                \
    do {                                                                      \
        int err_code = (stmt);                                                \
        if (err_code != (success)) {                                          \
            ggml_cann_error(#stmt, __func__, __FILE__, __LINE__, error_fn()); \
        }                                                                     \
    } while (0);

#define ACL_CHECK(stmt) ACL_CHECK_GEN(stmt, 0, aclGetRecentErrMsg)

struct ggml_cann_device_info {
    int32_t device_count;

    // TODO: add more device info later.
    // struct cann_device_info {
    //     int     cc;                 // compute capability
    //     size_t  smpb;               // max. shared memory per block
    //     bool    vmm;                // virtual memory support
    //     size_t  vmm_granularity;    // granularity of virtual memory
    //     size_t  total_vram;
    // };

    // cann_device_info devices[GGML_CANN_MAX_DEVICES] = {};
};

const ggml_cann_device_info& ggml_cann_info();

void ggml_cann_set_device(int32_t device);
int32_t ggml_cann_get_device();

struct ggml_backend_cann_context {
    int32_t device;
    std::string name;

    aclrtStream streams[GGML_CANN_MAX_DEVICES][GGML_CANN_MAX_STREAMS] = {
        {nullptr}};

    explicit ggml_backend_cann_context(int device)
        : device(device), name(GGML_CANN_NAME + std::to_string(device)) {}

    ~ggml_backend_cann_context() {
        for (int i = 0; i < GGML_CANN_MAX_DEVICES; ++i) {
            for (int j = 0; j < GGML_CANN_MAX_STREAMS; ++j) {
                if (streams[i][j] != nullptr) {
                    ACL_CHECK(aclrtDestroyStream(streams[i][j]));
                }
            }
        }
    }

    aclrtStream stream(int device, int stream) {
        if (streams[device][stream] == nullptr) {
            ggml_cann_set_device(device);
            ACL_CHECK(aclrtCreateStreamWithConfig(&streams[device][stream], 0,
                                                  ACL_STREAM_FAST_LAUNCH));
        }
        return streams[device][stream];
    }

    aclrtStream stream() { return stream(device, 0); }
};
