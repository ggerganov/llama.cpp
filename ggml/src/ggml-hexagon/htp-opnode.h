#ifndef HTP_OPNODE_H
#define HTP_OPNODE_H

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "htp/htp-ops.h"

#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#define HTP_OP_MAX_FUSED 16

// local std::inplace_vector replacement
template <typename T, size_t MAX_N>
struct htp_opvec {
    T      data[MAX_N] {};
    size_t n = 0;

    htp_opvec() = default;
    htp_opvec(std::initializer_list<T> init) {
        GGML_ASSERT(init.size() <= MAX_N);
        for (const auto & v : init) {
            data[n++] = v;
        }
    }

    size_t    size()  const { return n; }
    bool      empty() const { return n == 0; }
    void      clear()       { n = 0; }
    T *       begin()       { return data; }
    T *       end()         { return data + n; }
    const T * begin() const { return data; }
    const T * end()   const { return data + n; }
    T &       back()        { return data[n - 1]; }
    const T & back()  const { return data[n - 1]; }
    T &       operator[](size_t i)       { return data[i]; }
    const T & operator[](size_t i) const { return data[i]; }

    void push_back(T val) {
        GGML_ASSERT(n < MAX_N);
        data[n++] = val;
    }

    void insert(T * pos, T val) {
        GGML_ASSERT(n < MAX_N);
        for (T * p = data + n; p > pos; --p) {
            *p = *(p - 1);
        }
        *pos = val;
        n++;
    }
};

struct htp_opnode {
    ggml_tensor * node   { nullptr };
    htp_op_code   opcode { HTP_OP_INVALID };
    int32_t       params[HTP_OP_MAX_PARAMS] {0};
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<std::shared_ptr<ggml_tensor>>          dummy;

    htp_opvec<ggml_tensor *,   HTP_OP_MAX_FUSED>       fused;
    htp_opvec<const ggml_tensor *, HTP_OP_MAX_FUSED>   inputs;
    htp_opvec<const ggml_tensor *, HTP_OP_MAX_OUTPUTS> outputs;

    int n_active_src(const ggml_tensor * t) const {
        if (!t) return 0;
        for (int i = GGML_MAX_SRC - 1; i >= 0; i--) {
            if (t->src[i]) {
                return i + 1;
            }
        }
        return 0;
    }

    void init(ggml_tensor * node) {
        static_assert(sizeof(this->params) == sizeof(this->node->op_params), "params size mismatch");

        this->node = node;
        if (!node) return;

        memcpy(this->params, this->node->op_params, sizeof(this->params));

        // Build inputs (preserving optional nullptrs)
        if (this->node->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(this->node, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
            this->inputs.push_back(this->node->src[1]);
        } else {
            int n_inputs = n_active_src(this->node);
            for (int i = 0; i < n_inputs; i++) {
                this->inputs.push_back(this->node->src[i]);
            }
        }

        // Build outputs
        this->outputs.push_back(this->dst());
    }

    htp_opnode(htp_op_code opcode = HTP_OP_INVALID, ggml_tensor * node = nullptr) : opcode(opcode) {
        init(node);
    }

    const ggml_tensor * src0() const { return node->src[0]; }
    const ggml_tensor * src1() const { return node->src[1]; }
    const ggml_tensor * dst()  const { return outputs.empty() ? node : outputs.back(); }

    ggml_tensor * add_dummy(const ggml_tensor & t) {
        dummy.push_back(std::make_shared<ggml_tensor>(t));
        return dummy.back().get();
    }

    const htp_opvec<const ggml_tensor *, HTP_OP_MAX_FUSED> & get_inputs() const {
        return inputs;
    }

    const htp_opvec<const ggml_tensor *, HTP_OP_MAX_OUTPUTS> & get_outputs() const {
        return outputs;
    }

    std::string name() const;
};

struct htp_opformat {
    char strides[64 * GGML_MAX_SRC];
    char dims[64 * GGML_MAX_SRC];
    char types[16 * GGML_MAX_SRC];
    char buffs[64 * GGML_MAX_SRC];
    char names[64 * GGML_MAX_SRC];
    char kparams[128];

    int          format_tensor_dims(char * str, size_t max_size, const struct ggml_tensor * t);
    void         format_op_dims(char * str, size_t max_size, const htp_opnode & node);
    int          format_tensor_strides(char * str, size_t max_size, const struct ggml_tensor * t);
    void         format_op_strides(char * str, size_t max_size, const htp_opnode & node);
    void         format_op_types(char * str, size_t max_size, const htp_opnode & node);
    const char * tensor_buff_name(const struct ggml_tensor * t);
    void         format_op_buffs(char * str, size_t max_size, const htp_opnode & node);
    void         format_op_names(char * str, size_t max_size, const htp_opnode & node);
    void         format_kernel_params(char * str, size_t max_size, const htp_opnode & node);
    void         format(const htp_opnode & node);

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        names[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(const htp_opnode & node) { format(node); }
};

#endif // HTP_OPNODE_H
