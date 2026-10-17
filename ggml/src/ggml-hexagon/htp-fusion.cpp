#include "htp-fusion.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstring>

static inline bool htp_same_tensor_data(const ggml_tensor * a, const ggml_tensor * b) {
    return a && b && (a == b || a->data == b->data);
}

static inline bool htp_same_shape_and_contiguity(const ggml_tensor * a, const ggml_tensor * b) {
    return ggml_are_same_shape(a, b) && (ggml_is_contiguous(a) == ggml_is_contiguous(b));
}

static inline bool htp_match_consumer(
    const htp_opnode & prev,
    const htp_opnode & cur,
    bool commutative,
    const ggml_tensor ** other_src = nullptr
) {
    if (prev.outputs.size() != 1) {
        return false;
    }
    const ggml_tensor * prev_out = prev.dst();
    if (!htp_tensor_is_fuseable(prev_out)) {
        return false;
    }
    if (htp_same_tensor_data(cur.src0(), prev_out)) {
        if (other_src) *other_src = cur.src1();
        return true;
    }
    if (commutative && htp_same_tensor_data(cur.src1(), prev_out)) {
        if (other_src) *other_src = cur.src0();
        return true;
    }
    return false;
}

static bool htp_matmul_add_can_collapse(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * src2,
    const struct ggml_tensor * dst
) {
    if (!htp_matmul_can_collapse(src0, src1, dst)) {
        return false;
    }
    if (!src2) {
        return true;
    }
    const int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    if (src2_nrows == 1) {
        return true;
    }
    size_t s2;
    return src2->nb[0] == ggml_type_size(src2->type) &&
           ggml_are_same_shape(src2, dst) &&
           htp_rows_stride(src2->ne, src2->nb, &s2);
}

static void htp_precompute_fused_matmul_add_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * src2,
    const struct ggml_tensor * dst,
    struct htp_mm_kernel_params * kparams
) {
    if (htp_matmul_add_can_collapse(src0, src1, src2, dst)) {
        const ggml_tensor src1_collapsed = htp_tensor_collapse_rows(src1);
        const ggml_tensor dst_collapsed  = htp_tensor_collapse_rows(dst);
        const ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
                                           ? htp_tensor_collapse_rows(src2)
                                           : (src2 ? *src2 : ggml_tensor{});
        const struct ggml_tensor * p_src2 = src2 ? &src2_collapsed : nullptr;
        const size_t src2_size = p_src2 ? hex_round_up(ggml_nbytes(p_src2), 128) : 0;
        htp_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_size, kparams);
        kparams->collapse = 1;
        return;
    }
    const size_t src2_size = src2 ? hex_round_up(ggml_nbytes(src2), 128) : 0;
    htp_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_size, kparams);
}

static void htp_precompute_fused_mmnx_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * act,
    int32_t n_weights,
    struct htp_mm_kernel_params * kparams
) {
    const uint32_t n_threads   = htp_session_n_threads(sess);
    const uint32_t n_hmx       = htp_session_n_hmx(sess);
    const size_t   vtcm_budget = htp_session_vtcm_size(sess);

    memset(kparams, 0, sizeof(*kparams));
    kparams->n_threads = n_threads;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = act->ne[0];
    const int ne11 = act->ne[1];
    const int ne12 = act->ne[2];
    const int ne13 = act->ne[3];

    const bool is_tiled    = htp_tensor_is_tiled(src0);
    const int  wtype       = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    const int  ne00_padded = is_tiled ? hex_round_up(ne00, 32) : ne00;
    const int  ne11_padded = hex_round_up(ne11, 32);
    const int  ne01_tiled  = hex_round_up(ne01, 32);
    const bool is_batched  = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    const bool hmx_enabled = (n_hmx > 0) && (opt_mm_select >= 2);
    if (hmx_enabled && htp_matmul_is_hmx_eligible(src0, act, nullptr, false, is_batched)) {
        if (htp_precompute_hmx_mm_params(sess, src0, act, nullptr, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, false, is_batched, 0, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            goto finalize;
        }
    }

    if (!is_tiled || ne10 % 32 != 0) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        return;
    }

    {
        const int    act_nrows     = ne11 * ne12 * ne13;
        const size_t act_row_size  = htp_mm_tiled_act_row_size(wtype, ne10);
        const size_t src0_row_size = src0->nb[1];

        const uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        uint32_t best_n_prefetch = 2;
        for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            struct htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, n_threads,
                0, src0_row_size, act_row_size, 0, d, false, true
            );
            if (L.total_bytes <= vtcm_budget) {
                best_n_prefetch = d;
                break;
            }
        }

        struct htp_mm_hvx_vtcm_layout L;
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, n_threads,
            0, src0_row_size, act_row_size, 0, best_n_prefetch, false, true
        );

        if (opt_mm_select >= 1 && L.total_bytes <= vtcm_budget) {
            kparams->kernel_type       = HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->tile_size         = htp_tensor_get_tile_size(wtype);
            kparams->aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);
            kparams->act_row_size      = act_row_size;
            kparams->vtcm_src0_size    = L.src0_bytes;
            kparams->vtcm_act_size     = L.act_bytes;
            kparams->vtcm_bias_size    = 0;
            kparams->vtcm_dst_size     = L.dst_bytes;
            kparams->vtcm_size         = L.total_bytes;
            kparams->n_prefetch        = best_n_prefetch;
            kparams->n_weights         = n_weights;
        } else {
            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            return;
        }
    }

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_values(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_values(ne11);
    kparams->div_r2       = init_fastdiv_values(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_values(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_values(ne12);
}

static void htp_precompute_fused_mmidnx_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * act,
    const struct ggml_tensor * dst,
    int32_t n_weights,
    struct htp_mm_kernel_params * kparams
) {
    htp_precompute_matmul_params_impl(sess, src0, act, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
}

static bool mm_is_hmx_eligible(const ggml_tensor * t) {
    if (opt_nhmx == 0) return false;

    const ggml_tensor * src0 = t->src[0];
    const ggml_tensor * src1 = t->src[1];

    if (htp_matmul_can_collapse(src0, src1, t)) {
        const ggml_tensor src1_c = htp_tensor_collapse_rows(src1);
        return htp_matmul_is_hmx_eligible(src0, &src1_c, t, false, false);
    }

    const bool is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    const bool is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    return htp_matmul_is_hmx_eligible(src0, src1, t, is_matmul_id, is_batched);
}

static bool is_supported_mul_mat_nx_kernel(const ggml_tensor * src0, const struct htp_mm_kernel_params * kparams) {
    if (kparams->n_hmx) {
        return kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }
    return htp_tensor_is_tiled(src0) && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW);
}

static bool is_supported_mul_mat_id_nx_kernel(const ggml_tensor * src0, const struct htp_mm_kernel_params * kparams) {
    if (kparams->n_hmx) {
        return kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }
    return htp_tensor_is_tiled(src0) &&
           (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK);
}

static bool is_mergeable_mul_mat(const ggml_tensor * t) {
    if (t->op != GGML_OP_MUL_MAT) return false;

    const ggml_tensor * src0 = t->src[0];
    const ggml_tensor * src1 = t->src[1];
    if (src0->ne[2] != 1 || src0->ne[3] != 1) return false;

    if (mm_is_hmx_eligible(t)) {
        return htp_is_hmx_weight_type(src0->type);
    }
    return (src1->type == GGML_TYPE_F32) && htp_tensor_is_tiled(src0);
}

static bool is_mergeable_mul_mat_id(const ggml_tensor * t) {
    if (t->op != GGML_OP_MUL_MAT_ID) return false;

    const ggml_tensor * src0 = t->src[0];
    if (mm_is_hmx_eligible(t)) {
        return htp_is_hmx_weight_type(src0->type);
    }
    return htp_tensor_is_tiled(src0);
}

// --- Fusion rules ---

static bool fuse_allreduce_add(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (opt_ar_select != 2 || prev.opcode != HTP_OP_ALLREDUCE) return false;

    const auto * ar_kparams = (const struct htp_allreduce_kernel_params *) prev.kernel_params;
    const uint32_t rank    = (uint32_t) ar_kparams->rank;
    const uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;

    const ggml_tensor * ar_local = prev.inputs[rank];
    if (!htp_tensor_is_fuseable(ar_local)) return false;

    const ggml_tensor * add_dst    = cur.dst();
    const ggml_tensor * res_tensor = nullptr;
    if (htp_same_tensor_data(cur.src0(), ar_local)) {
        res_tensor = cur.src1();
    } else if (htp_same_tensor_data(cur.src1(), ar_local)) {
        res_tensor = cur.src0();
    } else {
        return false;
    }

    if (ar_local->type != res_tensor->type) return false;

    const bool is_same_shape = ggml_are_same_shape(ar_local, res_tensor);
    const bool is_row_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] &&
                                                  res_tensor->ne[1] == 1 && res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);
    if (!is_same_shape && !is_row_bcast) return false;

    if (is_same_shape) {
        if (ar_local->nb[1] != res_tensor->nb[1] || ar_local->nb[2] != res_tensor->nb[2] || ar_local->nb[3] != res_tensor->nb[3]) {
            return false;
        }
        if (ggml_is_contiguous(ar_local) != ggml_is_contiguous(res_tensor)) return false;
    }
    if (ggml_is_contiguous(ar_local) != ggml_is_contiguous(add_dst)) return false;

    const bool is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && add_dst->data == ar_local->data;

    struct htp_allreduce_kernel_params kparams;
    if (!htp_precompute_allreduce_params(sess, add_dst, rank, n_ranks, true, is_row_bcast, is_shard_ok, &kparams)) {
        HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", htp_session_name(sess));
        return false;
    }

    const bool scatter_ok = kparams.n_dsts > 1;
    kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

    for (uint32_t r = 0; r < n_ranks; r++) {
        const ggml_tensor * ar_src = prev.inputs[r];
        if (!htp_tensors_overlap(add_dst, ar_src)) continue;
        if (scatter_ok && r == rank) continue;
        HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", htp_session_name(sess), r);
        return false;
    }

    prev.opcode = HTP_OP_ALLREDUCE_ADD;
    prev.name   = "ALLREDUCE+ADD";
    prev.inputs.push_back(res_tensor);
    if (kparams.mode != HTP_ALLREDUCE_SHARDED_FANOUT) {
        prev.outputs = { add_dst };
    }
    prev.fused.push_back(cur.node);
    memcpy(prev.kernel_params, &kparams, sizeof(kparams));
    return true;
}

static bool fuse_rms_norm_mul(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_RMS_NORM) return false;

    const ggml_tensor * weight = nullptr;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ true, &weight)) return false;

    const ggml_tensor * src0 = prev.src0();
    const ggml_tensor * dst  = cur.dst();
    if (src0->ne[0] != weight->ne[0] || !htp_same_shape_and_contiguity(src0, dst)) return false;

    const bool is_row_bcast  = (weight->ne[1] == 1 && weight->ne[2] == 1 && weight->ne[3] == 1);
    const bool is_same_shape = ggml_are_same_shape(src0, weight);
    if (!is_row_bcast && !is_same_shape) return false;

    struct htp_unary_kernel_params kparams;
    htp_precompute_unary_params(sess, HTP_OP_RMS_NORM_MUL, src0, weight, dst, &kparams);
    if ((size_t) kparams.vtcm_size > htp_session_vtcm_size(sess)) {
        HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                    htp_session_name(sess), kparams.vtcm_size, htp_session_vtcm_size(sess));
        return false;
    }

    prev.opcode  = HTP_OP_RMS_NORM_MUL;
    prev.name   += "+MUL";
    prev.inputs  = { src0, weight };
    prev.outputs = { dst };
    prev.fused.push_back(cur.node);
    memcpy(prev.kernel_params, &kparams, sizeof(kparams));
    return true;
}

static bool fuse_rms_norm_scale(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    GGML_UNUSED(sess);
    if (prev.opcode != HTP_OP_RMS_NORM && prev.opcode != HTP_OP_RMS_NORM_MUL) return false;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ false)) return false;

    const ggml_tensor * src0 = prev.src0();
    const ggml_tensor * dst  = cur.dst();
    if (src0->type != GGML_TYPE_F32 || !htp_same_shape_and_contiguity(src0, dst)) return false;

    float scale = 0.0f;
    float bias  = 0.0f;
    memcpy(&scale, (const float *) cur.params + 0, sizeof(float));
    memcpy(&bias,  (const float *) cur.params + 1, sizeof(float));
    if (bias != 0.0f || scale == 0.0f) return false;

    float prev_scale = 0.0f;
    memcpy(&prev_scale, &prev.params[1], sizeof(float));
    if (prev_scale != 0.0f) {
        scale *= prev_scale;
    }
    memcpy(&prev.params[1], &scale, sizeof(float));

    prev.name      += "+SCALE";
    prev.outputs[0] = dst;
    prev.fused.push_back(cur.node);
    return true;
}

static bool fuse_mul_mat_add(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_MUL_MAT) return false;

    const ggml_tensor * src2 = nullptr;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ true, &src2)) return false;
    if (src2->type != GGML_TYPE_F32) return false;

    const ggml_tensor * src0 = prev.src0();
    const ggml_tensor * src1 = prev.src1();
    const ggml_tensor * dst  = cur.dst();
    if (!ggml_are_same_shape(prev.dst(), dst)) return false;

    const auto * orig_kparams = (const struct htp_mm_kernel_params *) prev.kernel_params;
    struct htp_mm_kernel_params kparams;
    htp_precompute_fused_matmul_add_params(sess, src0, src1, src2, dst, &kparams);
    if (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || orig_kparams->collapse != kparams.collapse) {
        return false;
    }

    const int src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
    if (kparams.n_hmx == 0 && src1_nrows != 1) return false;

    const size_t vtcm_size = htp_session_vtcm_size(sess);
    if ((size_t) kparams.vtcm_size > vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)\n",
                    htp_session_name(sess), kparams.vtcm_size, vtcm_size);
        return false;
    }

    if (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
        if (kparams.m_chunk < orig_kparams->m_chunk ||
            kparams.n_chunk < orig_kparams->n_chunk ||
            kparams.n_act_threads < orig_kparams->n_act_threads) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                        htp_session_name(sess), orig_kparams->m_chunk, kparams.m_chunk,
                        orig_kparams->n_chunk, kparams.n_chunk,
                        orig_kparams->n_act_threads, kparams.n_act_threads);
            return false;
        }
    }

    prev.opcode  = HTP_OP_MUL_MAT_ADD;
    prev.name    = "MUL_MAT+ADD";
    prev.inputs  = { src0, src1, src2 };
    prev.outputs = { dst };
    prev.fused.push_back(cur.node);
    memcpy(prev.kernel_params, &kparams, sizeof(kparams));
    return true;
}

static bool fuse_mul_mat_nx(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_MUL_MAT && prev.opcode != HTP_OP_MUL_MAT_NX) return false;
    if (!is_mergeable_mul_mat(prev.node) || !is_mergeable_mul_mat(cur.node)) return false;

    const uint32_t curr_n = (uint32_t) prev.outputs.size();
    if (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) return false;

    const ggml_tensor * w0   = prev.inputs[0];
    const ggml_tensor * x    = prev.inputs[curr_n];
    const ggml_tensor * w_in = cur.src0();
    const ggml_tensor * x_in = cur.src1();
    const ggml_tensor * d_in = cur.dst();

    if (x_in != x || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] ||
        htp_tensor_is_tiled(w_in) != htp_tensor_is_tiled(w0) ||
        mm_is_hmx_eligible(prev.node) != mm_is_hmx_eligible(cur.node)) {
        return false;
    }

    const auto * orig_kparams = (const struct htp_mm_kernel_params *) prev.kernel_params;
    const bool collapse = orig_kparams->collapse && htp_matmul_can_collapse(w_in, x, d_in);
    if (orig_kparams->collapse && !collapse) return false;

    struct htp_mm_kernel_params kparams;
    if (collapse) {
        const ggml_tensor x_collapsed = htp_tensor_collapse_rows(x);
        htp_precompute_fused_mmnx_params(sess, w0, &x_collapsed, curr_n + 1, &kparams);
        kparams.collapse = 1;
    } else {
        htp_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
    }
    if (!is_supported_mul_mat_nx_kernel(w0, &kparams)) return false;

    const size_t vtcm_size = htp_session_vtcm_size(sess);
    if ((size_t) kparams.vtcm_size > vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                    htp_session_name(sess), kparams.vtcm_size, vtcm_size);
        return false;
    }

    prev.opcode = HTP_OP_MUL_MAT_NX;
    prev.name   = "MUL_MAT_NX";
    prev.inputs.insert(prev.inputs.begin() + curr_n, w_in);
    prev.outputs.push_back(d_in);
    prev.fused.push_back(cur.node);
    memcpy(prev.kernel_params, &kparams, sizeof(kparams));
    return true;
}

static bool fuse_mul_mat_id_nx(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_MUL_MAT_ID && prev.opcode != HTP_OP_MUL_MAT_ID_NX) return false;
    if (!is_mergeable_mul_mat_id(prev.node) || !is_mergeable_mul_mat_id(cur.node)) return false;

    const uint32_t curr_n = (uint32_t) prev.outputs.size();
    if (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) return false;

    const ggml_tensor * w0     = prev.inputs[0];
    const ggml_tensor * x      = prev.inputs[curr_n];
    const ggml_tensor * ids    = prev.inputs[curr_n + 1];
    const ggml_tensor * w_in   = cur.src0();
    const ggml_tensor * x_in   = cur.src1();
    const ggml_tensor * ids_in = cur.node->src[2];
    const ggml_tensor * d_in   = cur.dst();

    if (x_in != x || ids_in != ids || w_in->type != w0->type ||
        w_in->ne[0] != w0->ne[0] || w_in->ne[2] != w0->ne[2] ||
        htp_tensor_is_tiled(w_in) != htp_tensor_is_tiled(w0) ||
        mm_is_hmx_eligible(prev.node) != mm_is_hmx_eligible(cur.node)) {
        return false;
    }

    struct htp_mm_kernel_params kparams;
    htp_precompute_fused_mmidnx_params(sess, w0, x, d_in, curr_n + 1, &kparams);
    if (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) return false;

    const size_t vtcm_size = htp_session_vtcm_size(sess);
    if ((size_t) kparams.vtcm_size > vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                    htp_session_name(sess), kparams.vtcm_size, vtcm_size);
        return false;
    }

    prev.opcode = HTP_OP_MUL_MAT_ID_NX;
    prev.name   = "MUL_MAT_ID_NX";
    prev.inputs.insert(prev.inputs.begin() + curr_n, w_in);
    prev.outputs.push_back(d_in);
    prev.fused.push_back(cur.node);
    memcpy(prev.kernel_params, &kparams, sizeof(kparams));
    return true;
}

static bool fuse_gdn_cpy(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    GGML_UNUSED(sess);
    if (prev.opcode != HTP_OP_GATED_DELTA_NET || prev.outputs.size() != 1) return false;

    const ggml_tensor * gdn_out  = prev.dst();
    const ggml_tensor * cpy_node = cur.node;
    const ggml_tensor * cpy_src  = cur.src0();
    const ggml_tensor * cpy_dst  = cur.dst();

    if (!cpy_src || !cpy_dst || !cpy_dst->data) return false;
    if (gdn_out->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) return false;
    if ((gdn_out->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_node->flags & GGML_TENSOR_FLAG_OUTPUT)) return false;

    const ggml_tensor * v = prev.node->src[2];
    if (!v) return false;

    const int64_t S_v       = v->ne[0];
    const int64_t H         = v->ne[1];
    const int64_t n_tokens  = v->ne[2];
    const int64_t n_seqs    = v->ne[3];
    const int64_t K         = ggml_get_op_params_i32(prev.node, 0);
    const size_t  tail_off  = (size_t) S_v * H * n_tokens * n_seqs * sizeof(float);
    const int64_t D         = S_v * S_v * H;
    const int64_t n_written = std::min<int64_t>(n_tokens, K);

    if (cpy_src->op != GGML_OP_VIEW || !htp_same_tensor_data(cpy_src->view_src, gdn_out) ||
        cpy_src->view_offs != tail_off || !ggml_is_contiguous(cpy_src)) {
        return false;
    }

    if (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->nb[0] != sizeof(float)) return false;
    if (n_seqs > 1 && cpy_dst->nb[1] != (size_t) D * sizeof(float)) return false;
    if (n_written > 1 && (cpy_dst->ne[2] != n_written || cpy_dst->nb[2] != (size_t) D * n_seqs * sizeof(float))) {
        return false;
    }

    prev.name += "+CPY";
    prev.outputs.push_back(cpy_dst);
    prev.fused.push_back(cur.node);
    return true;
}

static bool fuse_repeat_get_rows(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_REPEAT) return false;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ false)) return false;

    const ggml_tensor * repeat_out  = prev.dst();
    const ggml_tensor * repeat_src0 = prev.src0();
    const ggml_tensor * gr_src1     = cur.src1();
    const ggml_tensor * gr_dst      = cur.dst();

    if (repeat_src0->ne[0] != repeat_out->ne[0] || repeat_src0->ne[1] != repeat_out->ne[1]) return false;
    if (repeat_src0->type != repeat_out->type || !ggml_is_contiguous_rows(repeat_src0)) return false;

    struct htp_get_rows_kernel_params kparams;
    htp_precompute_get_rows_params(sess, repeat_src0, gr_src1, gr_dst, &kparams);
    if (kparams.vtcm_size == 0 || (size_t) kparams.vtcm_size > htp_session_vtcm_size(sess)) return false;

    prev.opcode  = HTP_OP_GET_ROWS;
    prev.name    = "REPEAT+GET_ROWS";
    prev.inputs  = { repeat_src0, gr_src1 };
    prev.outputs = { gr_dst };
    prev.fused.push_back(cur.node);
    memcpy(prev.kernel_params, &kparams, sizeof(kparams));
    return true;
}

static bool fuse_unary_mul_glu(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    htp_op_code glu_op = HTP_OP_INVALID;
    switch (prev.opcode) {
        case HTP_OP_UNARY_SILU:       glu_op = HTP_OP_GLU_SWIGLU;      break;
        case HTP_OP_UNARY_GELU:       glu_op = HTP_OP_GLU_GEGLU;       break;
        case HTP_OP_UNARY_GELU_QUICK: glu_op = HTP_OP_GLU_GEGLU_QUICK; break;
        case HTP_OP_UNARY_GELU_ERF:   glu_op = HTP_OP_GLU_GEGLU_ERF;   break;
        default: return false;
    }

    const ggml_tensor * up = nullptr;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ true, &up)) return false;

    const ggml_tensor * gate = prev.src0();
    const ggml_tensor * dst  = cur.dst();
    if (gate->type != GGML_TYPE_F32 && gate->type != GGML_TYPE_F16) return false;
    if ((glu_op == HTP_OP_GLU_GEGLU_QUICK || glu_op == HTP_OP_GLU_GEGLU_ERF) && gate->type != GGML_TYPE_F32) return false;
    if (up->type != gate->type || dst->type != gate->type) return false;
    if (!ggml_are_same_shape(gate, up) || !ggml_are_same_shape(gate, dst)) return false;
    if (!ggml_is_contiguous_1(gate) || !ggml_is_contiguous_1(up) || !ggml_is_contiguous(dst)) return false;

    const size_t row_size_aligned = hex_round_up(dst->ne[0] * ggml_type_size(dst->type), 128);
    if (6 * row_size_aligned * htp_session_n_threads(sess) > htp_session_vtcm_size(sess)) return false;

    prev.opcode  = glu_op;
    prev.name   += "+MUL";
    prev.inputs  = { gate, up };
    prev.outputs = { dst };
    prev.fused.push_back(cur.node);
    memset(prev.params,        0, sizeof(prev.params));
    memset(prev.kernel_params, 0, sizeof(prev.kernel_params));
    return true;
}

static bool fuse_mul_mul_scalar(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_MUL || prev.inputs.size() != 2) return false;

    const auto * k0 = (const struct htp_binary_kernel_params *) prev.kernel_params;
    const auto * k1 = (const struct htp_binary_kernel_params *) cur.kernel_params;
    if (k0->kernel_type != HTP_BINARY_KERNEL_SCALAR || k1->kernel_type != HTP_BINARY_KERNEL_SCALAR) return false;
    if (k0->src1_strided || k1->src1_strided) return false;

    const ggml_tensor * src2 = nullptr;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ false, &src2)) return false;

    const ggml_tensor * src0 = prev.src0();
    const ggml_tensor * src1 = prev.src1();
    const ggml_tensor * dst  = cur.dst();
    if (!ggml_are_same_shape(src1, src2) || !htp_same_shape_and_contiguity(src0, dst)) return false;

    const size_t vtcm_size = htp_session_vtcm_size(sess);
    struct htp_binary_kernel_params kparams = *k0;
    kparams.src2_size = kparams.src1_size;
    if (kparams.src1_size > 65536 || kparams.src1_size + kparams.src2_size > vtcm_size) return false;

    struct htp_binary_vtcm_layout L;
    htp_binary_vtcm_layout_build(&L, &kparams, vtcm_size);
    if (L.rows_per_buffer == 0 || L.total_bytes > vtcm_size) return false;

    kparams.rows_per_buffer = L.rows_per_buffer;
    kparams.vtcm_size       = L.total_bytes;

    prev.name += "+MUL";
    prev.inputs.push_back(src2);
    prev.outputs[0] = dst;
    prev.fused.push_back(cur.node);
    memcpy(prev.kernel_params, &kparams, sizeof(kparams));
    return true;
}

static bool fuse_binary_nx(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != cur.opcode) return false;

    const uint32_t curr_n = (uint32_t) prev.inputs.size();
    if (curr_n < 2 || curr_n >= HTP_OP_MAX_INPUTS) return false;

    const auto * k0 = (const struct htp_binary_kernel_params *) prev.kernel_params;
    const auto * k1 = (const struct htp_binary_kernel_params *) cur.kernel_params;
    const bool k0_ok = (k0->kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE) ||
                       (k0->kernel_type == HTP_BINARY_KERNEL_CHUNKED && !k0->is_scalar);
    const bool k1_ok = (k1->kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE) ||
                       (k1->kernel_type == HTP_BINARY_KERNEL_CHUNKED && !k1->is_scalar);
    if (!k0_ok || !k1_ok) return false;

    const bool is_comm = (cur.opcode == HTP_OP_ADD || cur.opcode == HTP_OP_MUL);
    const ggml_tensor * new_src = nullptr;
    if (!htp_match_consumer(prev, cur, is_comm, &new_src)) return false;

    const ggml_tensor * src0 = prev.inputs[0];
    const ggml_tensor * dst  = cur.dst();
    if (!ggml_are_same_shape(src0, prev.inputs[1]) ||
        !ggml_are_same_shape(src0, new_src) ||
        !htp_same_shape_and_contiguity(prev.dst(), dst)) {
        return false;
    }
    if (ggml_is_permuted(new_src) || ggml_is_permuted(dst)) return false;
    if (!htp_binary_strides_safe(src0, new_src, dst, ggml_type_size(src0->type))) return false;

    for (const auto * in : prev.inputs) {
        if (htp_tensors_overlap(dst, in) &&
            (dst->data != in->data || dst->nb[1] != in->nb[1] || dst->nb[2] != in->nb[2] || dst->nb[3] != in->nb[3])) {
            return false;
        }
    }

    struct htp_binary_kernel_params kparams = *k0;
    kparams.n_srcs = curr_n + 1;
    if (k0->kernel_type != HTP_BINARY_KERNEL_CHUNKED || k1->kernel_type != HTP_BINARY_KERNEL_CHUNKED) {
        if (kparams.kernel_type != HTP_BINARY_KERNEL_SAME_SHAPE) {
            const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
            memset(&kparams, 0, sizeof(kparams));
            kparams.kernel_type           = HTP_BINARY_KERNEL_SAME_SHAPE;
            kparams.n_threads             = (std::min)(htp_session_n_threads(sess), src0_nrows);
            kparams.n_srcs                = curr_n + 1;
            kparams.src0_row_size_aligned = k0->src0_row_size_aligned;
            kparams.src1_row_size_aligned = k0->src0_row_size_aligned;
            kparams.dst_row_size_aligned  = k0->dst_row_size_aligned;
        }

        auto check_split = [&](const ggml_tensor * t) {
            if (src0->ne[2] > 1 && t->nb[2] != t->ne[1] * t->nb[1]) kparams.split_at_ne01 = 1;
            if (src0->ne[3] > 1 && t->nb[3] != t->ne[2] * t->nb[2]) kparams.split_at_ne02 = 1;
        };
        for (const auto * in : prev.inputs) check_split(in);
        check_split(new_src);
        check_split(dst);
    }

    const size_t vtcm_size = htp_session_vtcm_size(sess);
    struct htp_binary_vtcm_layout L;
    htp_binary_vtcm_layout_build(&L, &kparams, vtcm_size);
    if (L.rows_per_buffer == 0 || L.total_bytes == 0 || L.total_bytes > vtcm_size) return false;

    kparams.rows_per_buffer = L.rows_per_buffer;
    kparams.vtcm_size       = L.total_bytes;

    switch (cur.opcode) {
        case HTP_OP_ADD: prev.name = "ADD_NX"; break;
        case HTP_OP_SUB: prev.name = "SUB_NX"; break;
        case HTP_OP_MUL: prev.name = "MUL_NX"; break;
        case HTP_OP_DIV: prev.name = "DIV_NX"; break;
        default: break;
    }
    prev.inputs.push_back(new_src);
    prev.outputs[0] = dst;
    prev.fused.push_back(cur.node);
    memcpy(prev.kernel_params, &kparams, sizeof(kparams));
    return true;
}

void htp_fusion_tag_graph(struct ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes; i++) {
        auto * extra = (htp_tensor_extra *) graph->nodes[i]->extra;
        if (!extra) continue;

        extra->flags &= ~HTP_TENSOR_FUSEABLE;

        const ggml_op op = graph->nodes[i]->op;
        if ((op == GGML_OP_RMS_NORM || op == GGML_OP_SCALE || op == GGML_OP_REPEAT ||
             op == GGML_OP_UNARY    || op == GGML_OP_ADD   || op == GGML_OP_SUB    ||
             op == GGML_OP_MUL      || op == GGML_OP_DIV   || op == GGML_OP_MUL_MAT_ID ||
             (op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(graph->nodes[i], 1) != GGML_HINT_SRC0_IS_HADAMARD)) &&
            ggml_node_has_n_uses(graph, i, 1)) {
            extra->flags |= HTP_TENSOR_FUSEABLE;
        }
    }
}

bool htp_try_fuse(const struct ggml_hexagon_session * sess, struct htp_opnode & prev, const struct htp_opnode & cur) {
    if (!opt_opfusion) return false;

    switch (cur.opcode) {
        case HTP_OP_ADD:
            return (htp_is_fusion_enabled(HTP_FUSE_ALLREDUCE_ADD) && fuse_allreduce_add(sess, prev, cur)) ||
                   (htp_is_fusion_enabled(HTP_FUSE_MUL_MAT_ADD)   && fuse_mul_mat_add(sess, prev, cur))   ||
                   (htp_is_fusion_enabled(HTP_FUSE_BINARY_NX)     && fuse_binary_nx(sess, prev, cur));
        case HTP_OP_MUL:
            return (htp_is_fusion_enabled(HTP_FUSE_RMS_NORM_MUL)   && fuse_rms_norm_mul(sess, prev, cur))   ||
                   (htp_is_fusion_enabled(HTP_FUSE_UNARY_MUL_GLU)  && fuse_unary_mul_glu(sess, prev, cur))  ||
                   (htp_is_fusion_enabled(HTP_FUSE_MUL_MUL_SCALAR) && fuse_mul_mul_scalar(sess, prev, cur)) ||
                   (htp_is_fusion_enabled(HTP_FUSE_BINARY_NX)      && fuse_binary_nx(sess, prev, cur));
        case HTP_OP_SUB:
        case HTP_OP_DIV:
            return htp_is_fusion_enabled(HTP_FUSE_BINARY_NX) && fuse_binary_nx(sess, prev, cur);
        case HTP_OP_SCALE:
            return htp_is_fusion_enabled(HTP_FUSE_RMS_NORM_SCALE) && fuse_rms_norm_scale(sess, prev, cur);
        case HTP_OP_GET_ROWS:
            return htp_is_fusion_enabled(HTP_FUSE_REPEAT_GET_ROWS) && fuse_repeat_get_rows(sess, prev, cur);
        case HTP_OP_CPY:
            return htp_is_fusion_enabled(HTP_FUSE_GDN_CPY) && fuse_gdn_cpy(sess, prev, cur);
        case HTP_OP_MUL_MAT:
            return htp_is_fusion_enabled(HTP_FUSE_MUL_MAT_NX) && fuse_mul_mat_nx(sess, prev, cur);
        case HTP_OP_MUL_MAT_ID:
            return htp_is_fusion_enabled(HTP_FUSE_MUL_MAT_ID_NX) && fuse_mul_mat_id_nx(sess, prev, cur);
        default:
            return false;
    }
}
