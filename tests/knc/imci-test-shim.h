#ifndef GGML_KNC_IMCI_TEST_SHIM_H
#define GGML_KNC_IMCI_TEST_SHIM_H
/* Test-only source-instruction semantics. No native ISA or timing claim.
 * Instruction definitions: Intel Xeon Phi ISA manual 327364-001 (2012).
 * https://www.intel.com/content/dam/develop/external/us/en/documents/327364001en.pdf
 * The translator rejects unsupported instructions and operand forms.
 */
#ifndef GGML_KNC_TEST_EMULATION
#    error "IMCI source model is restricted to explicit host correctness tests"
#endif
#if defined(__k1om__) || defined(__KNC__) || defined(__MIC__)
#    error "The host instruction model must never enter a native KNC build"
#endif
#ifdef __FAST_MATH__
#    error "Instruction semantics and numerical oracles must retain strict FP"
#endif
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef union {
    uint32_t u[16];
    float    f[16];
} model_vector;

enum { _MM_HINT_T0 = 5, _MM_HINT_T1 = 6 };
#ifdef GGML_KNC_TEST_PREFETCH_TRACE
void ggml_knc_test_prefetch(const void * p, int hint);
#else
static inline void ggml_knc_test_prefetch(const void * p, int hint) {
    (void) p;
    (void) hint;
}
#endif

#define MODEL_HELPER static __attribute__((noinline, unused))

MODEL_HELPER uint64_t model_load_u64(uintptr_t address) {
    uint64_t value;
    memcpy(&value, (const void *) address, sizeof(value));
    return value;
}

MODEL_HELPER void model_load(model_vector * out, uintptr_t address) {
    assert((address & 63u) == 0);
    memcpy(out, (const void *) address, 64);
}

MODEL_HELPER void model_store(uintptr_t address, const model_vector * value) {
    assert((address & 63u) == 0);
    memcpy((void *) address, value, 64);
}

MODEL_HELPER void model_load_u8(model_vector * out, uintptr_t address) {
    const uint8_t * p = (const uint8_t *) address;
    unsigned        i;
    assert((address & 15u) == 0);
    for (i = 0; i < 16; ++i) {
        out->u[i] = p[i];
    }
}

MODEL_HELPER void model_load_i8_f32(model_vector * out, uintptr_t address) {
    const uint8_t * p = (const uint8_t *) address;
    unsigned        i;
    assert((address & 15u) == 0);
    for (i = 0; i < 16; ++i) {
        out->f[i] = (float) (p[i] < 128 ? (int) p[i] : (int) p[i] - 256);
    }
}

MODEL_HELPER void model_broadcast(model_vector * out, uintptr_t address) {
    uint32_t value;
    unsigned i;
    assert((address & 3u) == 0);
    memcpy(&value, (const void *) address, sizeof(value));
    for (i = 0; i < 16; ++i) {
        out->u[i] = value;
    }
}

MODEL_HELPER void model_broadcast_i32_f32(model_vector * out, uintptr_t address) {
    int32_t  value;
    unsigned i;
    assert((address & 3u) == 0);
    memcpy(&value, (const void *) address, sizeof(value));
    for (i = 0; i < 16; ++i) {
        out->f[i] = (float) value;
    }
}

MODEL_HELPER void model_i32_f32(model_vector * out, const model_vector * in) {
    unsigned i;
    for (i = 0; i < 16; ++i) {
        int32_t value;
        memcpy(&value, &in->u[i], sizeof(value));
        out->f[i] = (float) value;
    }
}

MODEL_HELPER void model_f32_i32(model_vector * out, const model_vector * in) {
    unsigned i;
    for (i = 0; i < 16; ++i) {
        const float value = in->f[i];
        /* This kernel converts only exact integer-valued FP32 sums < 2^24.
         * Reject a violated proof rather than hide it with a rounding rule. */
        assert(isfinite(value) && (double) value >= INT32_MIN && (double) value <= INT32_MAX);
        assert(truncf(value) == value && fabsf(value) < 16777216.0f);
        out->u[i] = (uint32_t) (int32_t) value;
    }
}

#undef MODEL_HELPER
#endif
