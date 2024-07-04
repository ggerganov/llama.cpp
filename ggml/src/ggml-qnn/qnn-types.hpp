
#pragma once

#include "QnnCommon.h"
#include "QnnInterface.h"
#include "QnnTypes.h"
#include "Saver/QnnSaver.h"
#include "System/QnnSystemInterface.h"

namespace qnn {
// =================================================================================================
//
// helper data type / data structure / macros / functions of
// Qualcomm QNN(Qualcomm Neural Network, aka Qualcomm AI Engine Direct) SDK
// ref:https://github.com/pytorch/executorch/tree/main/backends/qualcomm
// =================================================================================================
enum sdk_profile_level { profile_off = 0, profile_basic = 1, profile_detail = 2 };

enum qcom_htp_arch {
    NONE = 0,
    V68 = 68,
    V69 = 69,
    V73 = 73,
    V75 = 75,
};

enum qcom_chipset {
    UNKNOWN_SM = 0,
    SM8450 = 36, // v69
    SM8475 = 42, // v69
    SM8550 = 43, // v73
    SM8650 = 57, // v75
};

struct qcom_socinfo {
    uint32_t soc_model;
    size_t htp_arch;
    size_t vtcm_size_in_mb;
};

using pfn_rpc_mem_init = void (*)(void);
using pfn_rpc_mem_deinit = void (*)(void);
using pfn_rpc_mem_alloc = void *(*)(int, uint32_t, int);
using pfn_rpc_mem_free = void (*)(void *);
using pfn_rpc_mem_to_fd = int (*)(void *);

using pfn_qnnsaver_initialize = decltype(QnnSaver_initialize);
using pfn_qnninterface_getproviders = decltype(QnnInterface_getProviders);
using pfn_qnnsysteminterface_getproviders = decltype(QnnSystemInterface_getProviders);
} // namespace qnn

#define QNN_VER_PTR(x) (&((x).v1)) // TODO: remove this macro after we have a separate header for QNN

#define RPCMEM_DEFAULT_FLAGS 1
#define RPCMEM_HEAP_ID_SYSTEM 25
