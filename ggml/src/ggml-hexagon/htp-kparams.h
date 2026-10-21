#ifndef HTP_KPARAMS_H
#define HTP_KPARAMS_H

#include "htp-common.h"

#include "htp/allreduce-ops.h"
#include "htp/argsort-ops.h"
#include "htp/binary-ops.h"
#include "htp/concat-ops.h"
#include "htp/cpy-ops.h"
#include "htp/cumsum-ops.h"
#include "htp/diag-ops.h"
#include "htp/fill-ops.h"
#include "htp/flash-attn-ops.h"
#include "htp/gated-delta-net-ops.h"
#include "htp/get-rows-ops.h"
#include "htp/glu-ops.h"
#include "htp/im2col-ops.h"
#include "htp/matmul-ops.h"
#include "htp/pad-ops.h"
#include "htp/pool-ops.h"
#include "htp/repeat-ops.h"
#include "htp/roll-ops.h"
#include "htp/rope-ops.h"
#include "htp/set-rows-ops.h"
#include "htp/softmax-ops.h"
#include "htp/solve-tri-ops.h"
#include "htp/ssm-conv.h"
#include "htp/sum-rows-ops.h"
#include "htp/unary-ops.h"

bool        htp_rows_stride(const int64_t * ne, const size_t * nb, size_t * stride);
bool        htp_matmul_can_collapse(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);
ggml_tensor htp_tensor_collapse_rows(const struct ggml_tensor * t);

bool htp_precompute_allreduce_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    bool has_add,
    bool is_row_bcast,
    bool is_shard_ok,
    struct htp_allreduce_kernel_params * kparams
);

bool htp_precompute_flash_attn_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_fa_kernel_params * kparams
);

bool htp_precompute_gated_delta_net_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_gdn_kernel_params * kparams
);

bool htp_precompute_matmul_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_mm_kernel_params * kparams
);

bool htp_precompute_fused_matmul_add_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * src2,
    const struct ggml_tensor * dst,
    struct htp_mm_kernel_params * kparams
);

bool htp_precompute_fused_mmnx_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * act,
    int32_t n_weights,
    struct htp_mm_kernel_params * kparams
);

bool htp_precompute_fused_mmidnx_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * act,
    const struct ggml_tensor * dst,
    int32_t n_weights,
    struct htp_mm_kernel_params * kparams
);

bool htp_precompute_binary_params(
    const struct ggml_hexagon_session * sess,
    uint32_t op,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_binary_kernel_params * kparams,
    const struct ggml_tensor * src2 = nullptr
);

bool htp_precompute_unary_params(
    const struct ggml_hexagon_session * sess,
    uint32_t op,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_unary_kernel_params * kparams
);

bool htp_precompute_get_rows_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_get_rows_kernel_params * kparams
);

bool htp_precompute_set_rows_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_set_rows_kernel_params * kparams
);

bool htp_precompute_softmax_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_softmax_kernel_params * kparams
);

bool htp_precompute_rope_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_rope_kernel_params * kparams
);

bool htp_precompute_sum_rows_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_sum_rows_kernel_params * kparams
);

bool htp_precompute_glu_params(
    const struct ggml_hexagon_session * sess,
    uint32_t op,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_glu_kernel_params * kparams
);

bool htp_precompute_repeat_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_repeat_kernel_params * kparams
);

bool htp_precompute_ssm_conv_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_ssm_conv_kernel_params * kparams
);

bool htp_precompute_sort_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    bool is_top_k,
    struct htp_sort_kernel_params * kparams
);

bool htp_precompute_pool_2d_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * dst,
    struct htp_pool_2d_kernel_params * kparams,
    bool is_pool_1d
);

bool htp_precompute_concat_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_concat_kernel_params * kparams
);

bool htp_precompute_cpy_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_copy_kernel_params * kparams
);

bool htp_precompute_im2col_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_im2col_kernel_params * kparams
);

bool htp_precompute_im2col_3d_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_im2col_kernel_params * kparams
);

bool htp_precompute_pad_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_pad_kernel_params * kparams
);

bool htp_precompute_cumsum_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_cumsum_kernel_params * kparams
);

bool htp_precompute_fill_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_fill_kernel_params * kparams
);

bool htp_precompute_roll_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_roll_kernel_params * kparams
);

bool htp_precompute_diag_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_diag_kernel_params * kparams
);

bool htp_precompute_solve_tri_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_solve_tri_kernel_params * kparams
);

bool htp_precompute_tri_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op
);

#endif // HTP_KPARAMS_H
