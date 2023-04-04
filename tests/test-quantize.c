#include "ggml.h"
#undef NDEBUG
#include <assert.h>
#include <math.h>

int main(void) {
    #define QK 32
    float src[QK];
    uint8_t dst[24];
    int64_t hist[16];

    for (int i = 0; i < QK; i++) {
        src[i] = (float)(i + 1);
    }

    size_t size = ggml_quantize_q4_0(src, dst, QK, QK, hist);
    assert(size == 20);
    float max_result = ((float *)dst)[0];
    float max_expected = src[31] / ((1 << 3) - 1);
    assert(max_result == max_expected);
    for (int i = 0; i < QK; i++) {
        uint8_t q4_result = (i % 2) ? (dst[sizeof(float) + i/2] >> 4) : (dst[sizeof(float) + i/2] & 0xF);
        uint8_t q4_expected = roundf(src[i] / max_expected) + 8;
        assert(q4_result == q4_expected);
    }

    size = ggml_quantize_q4_1(src, dst, QK, QK, hist);
    assert(size == 24);
    float delta_result = ((float *)dst)[0];
    float delta_expected = (src[31] - src[0]) / ((1 << 4) - 1);
    assert(delta_result == delta_expected);
    float min_result = ((float *)dst)[1];
    float min_expected = src[0];
    assert(min_result == min_expected);
    for (int i = 0; i < QK; i++) {
        uint8_t q4_result = (i % 2) ? (dst[sizeof(float)*2 + i/2] >> 4) : (dst[sizeof(float)*2 + i/2] & 0xF);
        uint8_t q4_expected = roundf((src[i] - min_expected) / delta_expected);
        assert(q4_result == q4_expected);
    }

    return 0;
}
