// Shared helpers of the analog compute shaders.
//
// The random numbers are the ones of the CPU kernels (analog-kernels.cpp): the same 64 bit
// splitmix finalizer and the same 32 bit weights, so a given array configuration produces the very
// same noise stream no matter which backend runs the kernel. This is what makes the shader output
// comparable to the CPU output element by element instead of only statistically.

#ifndef ANALOG_COMMON_GLSL
#define ANALOG_COMMON_GLSL

// elements per level-snapping block, the block size of the formats and of the peripheral
// converters of typical array designs
#define ANALOG_BLOCK 32

// readout noise samples, both the table size and the two mixing constants have to match
// ggml_analog_kernel_noise_table() and analog_noise_index()
#define ANALOG_NOISE_TABLE 8192
#define ANALOG_NOISE_A 0x9E3779B1u
#define ANALOG_NOISE_B 0x85EBCA6Bu

uint64_t analog_hash(uint64_t x) {
    x += 0x9E3779B97F4A7C15UL;
    x = (x ^ (x >> 30))*0xBF58476D1CE4E5B9UL;
    x = (x ^ (x >> 27))*0x94D049BB133111EBUL;

    return x ^ (x >> 31);
}

// approximated standard normal: normalized sum of 4 uniform random numbers (Irwin-Hall)
float analog_gauss(uint64_t h) {
    uint u = 0u;

    for (int i = 0; i < 4; ++i) {
        h = analog_hash(h);
        u += uint(h >> 40); // 24 uniform bits
    }

    const float s = float(u)/float(1u << 24);

    return (s - 2.0f)*1.7320508f;
}

// one sample of the readout noise table, generated the same way as on the CPU instead of being
// uploaded as a buffer
float analog_noise_at(uint64_t idx) {
    const uint64_t h = analog_hash(idx*0x9E3779B97F4A7C15UL + 12345UL);

    const float u = float(uint( h        & 0xFFFFFFUL))/float(1u << 24);
    const float v = float(uint((h >> 24) & 0xFFFFFFUL))/float(1u << 24);

    // u + v - 1 is triangular with variance 1/6
    return 2.4494898f*(u + v - 1.0f);
}

// the noise sample that belongs to a (row, column) pair of the array
float analog_noise_of(uint row, uint col) {
    const uint idx = (row*ANALOG_NOISE_A + (col + 1u)*ANALOG_NOISE_B) & (ANALOG_NOISE_TABLE - 1u);

    return analog_noise_at(uint64_t(idx));
}

#endif
