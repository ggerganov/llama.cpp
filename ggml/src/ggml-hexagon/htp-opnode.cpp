#include "htp-opnode.h"

#include "htp/allreduce-ops.h"
#include "htp/argsort-ops.h"
#include "htp/binary-ops.h"
#include "htp/cumsum-ops.h"
#include "htp/diag-ops.h"
#include "htp/fill-ops.h"
#include "htp/flash-attn-ops.h"
#include "htp/gated-delta-net-ops.h"
#include "htp/get-rows-ops.h"
#include "htp/glu-ops.h"
#include "htp/matmul-ops.h"
#include "htp/pad-ops.h"
#include "htp/repeat-ops.h"
#include "htp/roll-ops.h"
#include "htp/softmax-ops.h"
#include "htp/solve-tri-ops.h"
#include "htp/ssm-conv.h"
#include "htp/sum-rows-ops.h"
#include "htp/unary-ops.h"

#include <algorithm>
#include <stdio.h>

std::string htp_opnode::name() const {
    switch (opcode) {
        case HTP_OP_MDEV_GROUP:    return "MDEV_GROUP";
        case HTP_OP_CPY_FENCE:     return "CPY+FENCE";
        case HTP_OP_FENCE:         return (node && node->op_params[1] == 0) ? "FENCE_WAIT" : "FENCE_SIGNAL";
        case HTP_OP_ALLREDUCE:     return "ALLREDUCE";
        case HTP_OP_ALLREDUCE_ADD: return "ALLREDUCE+ADD";
        case HTP_OP_HADAMARD:      return "MUL_MAT_HADAMARD";
        case HTP_OP_MUL_MAT_NX:    return "MUL_MAT_NX";
        case HTP_OP_MUL_MAT_ID_NX: return "MUL_MAT_ID_NX";
        case HTP_OP_ADD:
        case HTP_OP_SUB:
        case HTP_OP_MUL:
        case HTP_OP_DIV:
            if (!fused.empty()) {
                const auto * kp = (const struct htp_binary_kernel_params *) kernel_params;
                if (kp->kernel_type != HTP_BINARY_KERNEL_SCALAR) {
                    switch (opcode) {
                        case HTP_OP_ADD: return "ADD_NX";
                        case HTP_OP_SUB: return "SUB_NX";
                        case HTP_OP_MUL: return "MUL_NX";
                        case HTP_OP_DIV: return "DIV_NX";
                        default: break;
                    }
                }
            }
            break;
        default:
            break;
    }

    if (!node) {
        return "";
    }
    if (node->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(node, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        return "MUL_MAT_HADAMARD";
    }

    std::string res = ggml_op_desc(node);
    for (const auto * t : fused) {
        res += "+";
        res += ggml_op_desc(t);
    }
    return res;
}

int htp_opformat::format_tensor_dims(char * str, size_t max_size, const struct ggml_tensor * t) {
    if (!t) {
        return snprintf(str, max_size, "NONE");
    }
    if (t->ne[2] == 1 && t->ne[3] == 1) {
        return snprintf(str, max_size, "%d:%d", (int) t->ne[0], (int) t->ne[1]);
    } else {
        return snprintf(str, max_size, "%d:%d:%d:%d", (int) t->ne[0], (int) t->ne[1], (int) t->ne[2], (int) t->ne[3]);
    }
}

void htp_opformat::format_op_dims(char * str, size_t max_size, const htp_opnode & node) {
    char * p = str;
    char * p_end = str + max_size;
    const auto & inputs = node.get_inputs();

    if (!inputs.empty()) {
        p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[0]), (size_t)(p_end - p));

        for (size_t i = 1; i < inputs.size(); i++) {
            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
            }
            if (p < p_end) {
                p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[i]), (size_t)(p_end - p));
            }
        }

        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
        }
    }

    char self[64];
    format_tensor_dims(self, sizeof(self), node.dst());
    if (p < p_end) {
        p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
    }
}

int htp_opformat::format_tensor_strides(char * str, size_t max_size, const struct ggml_tensor * t) {
    if (!t) {
        return snprintf(str, max_size, "NONE");
    }
    const char * c = ggml_is_contiguous(t) ? "" : "!";

    if (t->ne[2] == 1 && t->ne[3] == 1) {
        return snprintf(str, max_size, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
    } else {
        return snprintf(str, max_size, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
    }
}

void htp_opformat::format_op_strides(char * str, size_t max_size, const htp_opnode & node) {
    char * p = str;
    char * p_end = str + max_size;
    const auto & inputs = node.get_inputs();

    if (!inputs.empty()) {
        p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[0]), (size_t)(p_end - p));

        for (size_t i = 1; i < inputs.size(); i++) {
            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
            }
            if (p < p_end) {
                p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[i]), (size_t)(p_end - p));
            }
        }

        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
        }
    }

    char self[64];
    format_tensor_strides(self, sizeof(self), node.dst());
    if (p < p_end) {
        p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
    }
}

void htp_opformat::format_op_types(char * str, size_t max_size, const htp_opnode & node) {
    char * p = str;
    char * p_end = str + max_size;
    const auto & inputs = node.get_inputs();

    if (!inputs.empty()) {
        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_end - p));
        }

        for (size_t i = 1; i < inputs.size(); i++) {
            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
            }
            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_end - p));
            }
        }

        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
        }
    }

    if (p < p_end) {
        p += std::min((size_t)snprintf(p, p_end - p, "%s", ggml_type_name(node.dst()->type)), (size_t)(p_end - p));
    }
}

const char * htp_opformat::tensor_buff_name(const struct ggml_tensor * t) {
    if (t && t->buffer) {
        return ggml_backend_buffer_name(t->buffer);
    }
    return "NONE";
}

void htp_opformat::format_op_buffs(char * str, size_t max_size, const htp_opnode & node) {
    char * p = str;
    char * p_end = str + max_size;
    const auto & inputs = node.get_inputs();

    if (!inputs.empty()) {
        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_end - p));
        }

        for (size_t i = 1; i < inputs.size(); i++) {
            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
            }
            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_end - p));
            }
        }

        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
        }
    }

    if (p < p_end) {
        p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(node.dst())), (size_t)(p_end - p));
    }
}

void htp_opformat::format_op_names(char * str, size_t max_size, const htp_opnode & node) {
    char * p = str;
    char * p_end = str + max_size;
    const auto & inputs = node.get_inputs();

    if (!inputs.empty()) {
        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_end - p));
        }

        for (size_t i = 1; i < inputs.size(); i++) {
            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
            }
            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_end - p));
            }
        }

        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
        }
    }

    if (p < p_end) {
        p += std::min((size_t)snprintf(p, p_end - p, "%s", node.dst()->name), (size_t)(p_end - p));
    }
}

void htp_opformat::format_kernel_params(char * str, size_t max_size, const htp_opnode & node) {
    if (node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID ||
        node.opcode == HTP_OP_MUL_MAT_NX || node.opcode == HTP_OP_MUL_MAT_ID_NX ||
        node.opcode == HTP_OP_MUL_MAT_ADD) {
        const auto * kparams = (const struct htp_mm_kernel_params *) node.kernel_params;
        const char * path = "unknown";
        int32_t type = kparams->kernel_type;
        if (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
            path = (kparams->tile_size > 0) ? "hmx-tiled" : "hmx-flat";
        } else if (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                   type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
            path = "hvx-flat";
        } else if (type == HTP_MM_KERNEL_HVX_QUANT_ROW || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
            path = "hvx-tiled";
        }
        snprintf(str, max_size, "%s vtcm %d", path, (int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_FLASH_ATTN_EXT) {
        const auto * kparams = (const struct htp_fa_kernel_params *) node.kernel_params;
        const char * path = "unknown";
        int32_t type = kparams->kernel_type;
        if (type == HTP_FA_KERNEL_HMX) {
            path = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
        } else if (type == HTP_FA_KERNEL_HVX) {
            path = "hvx";
        }
        snprintf(str, max_size, "%s vtcm %d", path, (int) kparams->vtcm_size);
    } else if (htp_op_is_unary(node.opcode)) {
        const auto * kparams = (const struct htp_unary_kernel_params *) node.kernel_params;
        snprintf(str, max_size, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_MDEV_GROUP && node.node) {
        snprintf(str, max_size, "idx %d count %d", (int) node.node->op_params[0], (int) node.dst()->ne[1]);
    } else if ((node.opcode == HTP_OP_FENCE || node.opcode == HTP_OP_CPY_FENCE) && node.node) {
        snprintf(str, max_size, "seq 0x%x", (uint32_t) node.node->op_params[0]);
    } else if (node.opcode == HTP_OP_ALLREDUCE && node.node) {
        snprintf(str, max_size, "seq 0x%x -> 0x%x", (uint32_t) node.node->op_params[0], (uint32_t) node.node->op_params[1]);
    } else if (node.opcode == HTP_OP_SSM_CONV) {
        const auto * kparams = (const struct htp_ssm_conv_kernel_params *) node.kernel_params;
        snprintf(str, max_size, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_SOFTMAX) {
        const auto * kparams = (const struct htp_softmax_kernel_params *) node.kernel_params;
        snprintf(str, max_size, "k%d nth %d vtcm %d", (int) kparams->kernel_type, (int) kparams->n_threads, (int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_GATED_DELTA_NET) {
        const auto * kparams = (const struct htp_gdn_kernel_params *) node.kernel_params;
        const char * path = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
        snprintf(str, max_size, "%s-%s vtcm %u", path, kparams->kda ? "kda" : "scalar",
                 (unsigned int) (kparams->vtcm_size ? kparams->vtcm_size : kparams->vtcm_per_thread * kparams->n_threads));
    } else if (node.opcode == HTP_OP_MUL || node.opcode == HTP_OP_ADD || node.opcode == HTP_OP_ADD_ID ||
               node.opcode == HTP_OP_SUB || node.opcode == HTP_OP_DIV) {
        const auto * kparams = (const struct htp_binary_kernel_params *) node.kernel_params;
        snprintf(str, max_size, "vtcm %u", (unsigned int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_ARGSORT || node.opcode == HTP_OP_TOP_K) {
        const auto * kparams = (const struct htp_sort_kernel_params *) node.kernel_params;
        snprintf(str, max_size, "%s nth %d nchk %d chk %d vtcm %d",
                 node.opcode == HTP_OP_TOP_K ? "top_k" : "argsort",
                 (int) kparams->n_threads, (int) kparams->n_chunks,
                 (int) kparams->chunk_elems, (int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_GET_ROWS) {
        const auto * kparams = (const struct htp_get_rows_kernel_params *) node.kernel_params;
        const char * ktype_str = "unknown";
        switch (kparams->kernel_type) {
            case HTP_GET_ROWS_KERNEL_SAMETYPE: ktype_str = "sametype"; break;
            case HTP_GET_ROWS_KERNEL_TILED:    ktype_str = "tiled"; break;
            case HTP_GET_ROWS_KERNEL_FLAT:     ktype_str = "flat"; break;
        }
        snprintf(str, max_size, "%s%s vtcm %u", ktype_str, kparams->n_threads > 1 ? "-multi" : "", (unsigned int) kparams->vtcm_size);
    } else if (htp_op_is_glu(node.opcode)) {
        const auto * kparams = (const struct htp_glu_kernel_params *) node.kernel_params;
        snprintf(str, max_size, "nth %d vtcm %u", (int) kparams->n_threads, (unsigned int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_PAD) {
        const auto * kparams = (const struct htp_pad_kernel_params *) node.kernel_params;
        snprintf(str, max_size, "k%d nth %d vtcm %u", (int) kparams->kernel_type, (int) kparams->n_threads, (unsigned int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_CUMSUM) {
        const auto * kparams = (const struct htp_cumsum_kernel_params *) node.kernel_params;
        snprintf(str, max_size, "%s nth %d vtcm %u", kparams->use_dma ? "dma" : "hvx", (int) kparams->n_threads, (unsigned int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_FILL) {
        const auto * kparams = (const struct htp_fill_kernel_params *) node.kernel_params;
        snprintf(str, max_size, "%s nth %d", kparams->kernel_type == HTP_FILL_KERNEL_CONTIGUOUS ? "contig" : "strided", (int) kparams->n_threads);
    } else if (node.opcode == HTP_OP_ROLL) {
        const auto * kparams = (const struct htp_roll_kernel_params *) node.kernel_params;
        snprintf(str, max_size, "k%d nth %d", (int) kparams->kernel_type, (int) kparams->n_threads);
    } else if (node.opcode == HTP_OP_DIAG) {
        const auto * kparams = (const struct htp_diag_kernel_params *) node.kernel_params;
        snprintf(str, max_size, "%s nth %d vtcm %u", kparams->use_dma ? "dma" : "hvx", (int) kparams->n_threads, (unsigned int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_SOLVE_TRI) {
        const auto * kparams = (const struct htp_solve_tri_kernel_params *) node.kernel_params;
        snprintf(str, max_size, "%s nth %d", kparams->batched ? "batched" : "chunked", (int) kparams->n_threads);
    } else {
        snprintf(str, max_size, "----");
    }
}

void htp_opformat::format(const htp_opnode & node) {
    format_op_dims(dims, sizeof(dims), node);
    format_op_strides(strides, sizeof(strides), node);
    format_op_types(types, sizeof(types), node);
    format_op_buffs(buffs, sizeof(buffs), node);
    format_op_names(names, sizeof(names), node);
    format_kernel_params(kparams, sizeof(kparams), node);
}
