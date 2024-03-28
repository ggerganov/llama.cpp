#include <aclnn/aclnn_base.h>

#include "common.h"

// Broadcast
aclDataType type_mapping(ggml_type type);

aclTensor* create_acl_tensor(const ggml_tensor* tensor,
                             const int64_t* bcast_ne = nullptr,
                             int64_t* bcast_stride = nullptr,
                             int64_t bcast_dims = 0);

bool need_bcast(const ggml_tensor* t0, const ggml_tensor* t1);

int64_t get_bcast_shape(const ggml_tensor* src0, const ggml_tensor* src1,
                        int64_t* bcast_ne_src0, int64_t* bcast_ne_src1,
                        int64_t* bcast_stride_src0, int64_t* bcast_stride_src1);

// Bcast macro to avoid duplicate code.
#define BCAST_SHAPE(src0, src1)                                       \
    int64_t bcast_ne_##src0[GGML_MAX_DIMS * 2];                       \
    int64_t bcast_ne_##src1[GGML_MAX_DIMS * 2];                       \
    int64_t bcast_stride_##src0[GGML_MAX_DIMS * 2];                   \
    int64_t bcast_stride_##src1[GGML_MAX_DIMS * 2];                   \
    int64_t bcast_dims =                                              \
        get_bcast_shape(src0, src1, bcast_ne_##src0, bcast_ne_##src1, \
                        bcast_stride_##src0, bcast_stride_##src1);

#define BCAST_PARAM(src) bcast_ne_##src, bcast_stride_##src, bcast_dims
