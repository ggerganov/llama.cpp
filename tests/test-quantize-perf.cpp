// Benchmark quantization specific functions on synthetic data

#include "ggml.h"

#undef NDEBUG
#include <algorithm>
#include <assert.h>
#include <functional>
#include <inttypes.h>
#include <math.h>
#include <memory>
#include <stdio.h>
#include <string>
#include <vector>

#define MAX_ALIGNMENT 64
#define QK 32
#define WARMUP 5
#define ITERATIONS 10

#define L1_SIZE      32*128
#define L2_SIZE     32*2048
#define L3_SIZE    32*20480
#define MEM_SIZE 32*2048000

struct quantize_perf_params {
    std::vector<std::string> include_types;
    std::vector<size_t> test_sizes;
    size_t alignment_offset = 0;
    bool op_quantize_row_q_reference = false;
    bool op_quantize_row_q = false;
    bool op_dequantize_row_q = false;
    bool op_quantize_row_q_dot = false;
    bool op_vec_dot_q = false;
};


#if defined(__x86_64__) || defined(__i386__)

#include <x86intrin.h>
inline int64_t cpu_cycles() {
// Rough way to detect new-ish CPUs
#ifdef __POPCNT__
    unsigned int dummy;
    return __rdtscp(&dummy);
#else
    return __rdtsc();
#endif
}

#else

#define cpu_cycles() 0

#endif


// Generate synthetic data
void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

float gigabytes_per_second(size_t bytes, int64_t usecs) {
    return bytes / (float) usecs * 1000000 / (1024*1024*1024);
}

void * align_with_offset(void * ptr, int offset) {
    size_t dummy_size = MAX_ALIGNMENT * 4;
    return (char *) std::align(MAX_ALIGNMENT, MAX_ALIGNMENT, ptr, dummy_size) + offset;
}

void benchmark_function(size_t size, size_t q_size, std::function<size_t(void)> function) {
    int64_t min_time_us = INT64_MAX;
    int64_t total_time_us = 0;
    int64_t min_time_cycles = INT64_MAX;
    int64_t total_time_cycles = 0;

    for (int i = 0; i < WARMUP; i++) {
        function();
    }


    for (int i = 0; i < ITERATIONS; i++) {
        const int64_t start_time = ggml_time_us();
        const int64_t start_cycles = cpu_cycles();

        function();

        const int64_t end_cycles = cpu_cycles();
        const int64_t end_time = ggml_time_us();

        total_time_cycles += end_cycles - start_cycles;
        min_time_cycles = std::min(min_time_cycles, end_cycles - start_cycles);
        total_time_us += end_time - start_time;
        min_time_us = std::min(min_time_us, end_time - start_time);
    }

    printf("      min cycles/%d vals   : %9.2f\n",  QK, QK * min_time_cycles / (float) size);
    printf("      avg cycles/%d vals   : %9.2f\n",  QK, QK * total_time_cycles / (float) (size * ITERATIONS));
    printf("      float32 throughput   : %9.2f GB/s\n",  gigabytes_per_second(4 * size * ITERATIONS, total_time_us));
    printf("      quantized throughput : %9.2f GB/s\n",  gigabytes_per_second(q_size * ITERATIONS, total_time_us));
}

int main(int argc, char * argv[]) {
    quantize_perf_params params {};

    // read command line

    bool invalid_param = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "--size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            size_t size = std::stoi(argv[i]);
            if (size % 32 != 0) {
                fprintf(stderr, "error: size %zu not divisible by 32\n", size);
                invalid_param = true;
                break;
            }
            params.test_sizes.push_back(size);
        } else if (arg == "-3") {
            // quick select sizes that probably fit in CPU caches
            params.test_sizes.push_back(L1_SIZE);
            params.test_sizes.push_back(L2_SIZE);
            params.test_sizes.push_back(L3_SIZE);
        } else if (arg == "-4") {
            // quick select cache sizes + memory
            params.test_sizes.push_back(L1_SIZE);
            params.test_sizes.push_back(L2_SIZE);
            params.test_sizes.push_back(L3_SIZE);
            params.test_sizes.push_back(MEM_SIZE);
        } else if (arg == "--op") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::string op {argv[i]};
            if (op == "quantize_row_q_reference") {
                params.op_quantize_row_q_reference = true;
            } else if (op == "quantize_row_q") {
                params.op_quantize_row_q = true;
            } else if (op == "dequantize_row_q") {
                params.op_dequantize_row_q = true;
            } else if (op == "quantize_row_q_dot") {
                params.op_quantize_row_q_dot = true;
            } else if (op == "vec_dot_q") {
                params.op_vec_dot_q = true;
            } else {
                invalid_param = true;
                break;
            }
        } else if (arg == "--type") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.include_types.push_back(argv[i]);
        } else if (arg == "--alignment-offset") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            int alignment = std::stoi(argv[i]);
            if (alignment < 0 || alignment > MAX_ALIGNMENT) {
            fprintf(stderr, "error: aligment-offset must be less than %d\n", MAX_ALIGNMENT);
                invalid_param = true;
                break;
            }
            params.alignment_offset = alignment;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        return 1;
    }

    if (params.test_sizes.empty()) {
        params.test_sizes.push_back(L1_SIZE);
    }
    if (!(params.op_quantize_row_q_reference || params.op_quantize_row_q || params.op_dequantize_row_q || params.op_quantize_row_q_dot || params.op_vec_dot_q)) {
        params.op_quantize_row_q_reference = params.op_quantize_row_q = params.op_dequantize_row_q = params.op_quantize_row_q_dot = params.op_vec_dot_q = true;
    }

    std::sort(params.test_sizes.begin(), params.test_sizes.end());
    size_t largest = params.test_sizes.back();

    std::vector<uint8_t> test_data1_v(largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_data2_v(largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_q1_v(largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_q2_v(largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_out_v(largest*4 + MAX_ALIGNMENT*2);

    float * test_data1 = (float *) align_with_offset(test_data1_v.data(), params.alignment_offset);
    float * test_data2 = (float *) align_with_offset(test_data2_v.data(), params.alignment_offset);
    float * test_q1 = (float *) align_with_offset(test_q1_v.data(), params.alignment_offset);
    float * test_q2 = (float *) align_with_offset(test_q2_v.data(), params.alignment_offset);
    float * test_out = (float *) align_with_offset(test_out_v.data(), params.alignment_offset);

    generate_data(0, largest, test_data1);
    generate_data(1, largest, test_data2);


    // Initialize GGML, ensures float conversion tables are initialized
    struct ggml_init_params ggml_params = {
        /* .mem_size   = */ 1*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(ggml_params);

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        quantize_fns_t qfns = ggml_internal_get_quantize_fn(i);
        if (!params.include_types.empty() && std::find(params.include_types.begin(), params.include_types.end(), ggml_type_name(type)) == params.include_types.end()) {
            continue;
        }

        if (qfns.quantize_row_q && qfns.dequantize_row_q) {
            printf("%s\n", ggml_type_name(type));

            if (params.op_quantize_row_q_reference) {
                printf("  quantize_row_q_reference\n");
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void ) {
                        qfns.quantize_row_q_reference(test_data1, test_q1, size);
                        return test_q1[0];
                    };
                    size_t quantized_size = size / ggml_blck_size(type) * ggml_type_size(type);
                    benchmark_function(size, quantized_size, quantize_fn);
                }
                printf("\n");
            }

            if (params.op_quantize_row_q) {
                printf("  quantize_row_q\n");
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void ) {
                        qfns.quantize_row_q(test_data1, test_q1, size);
                        return test_q1[0];
                    };
                    size_t quantized_size = size / ggml_blck_size(type) * ggml_type_size(type);
                    benchmark_function(size, quantized_size, quantize_fn);
                }
                printf("\n");
            }

            if (params.op_dequantize_row_q) {
                printf("  dequantize_row_q\n");
                qfns.quantize_row_q(test_data1, test_q1, largest);
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void ) {
                        qfns.dequantize_row_q(test_q1, test_out, size);
                        return test_out[0];
                    };
                    size_t quantized_size = size / ggml_blck_size(type) * ggml_type_size(type);
                    benchmark_function(size, quantized_size, quantize_fn);
                }
                printf("\n");
            }

            if (params.op_quantize_row_q_dot) {
                printf("  quantize_row_q_dot\n");
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void ) {
                        qfns.quantize_row_q_dot(test_data1, test_q1, size);
                        return test_q1[0];
                    };
                    size_t quantized_size = size / ggml_blck_size(type) * ggml_type_size(type);
                    benchmark_function(size, quantized_size, quantize_fn);
                }
                printf("\n");
            }

            if (params.op_vec_dot_q) {
                printf("  vec_dot_q\n");
                qfns.quantize_row_q(test_data1, test_q1, largest);
                qfns.quantize_row_q(test_data2, test_q2, largest);
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void ) {
                        float result;
                        qfns.vec_dot_q(size, &result, test_q1, test_q2);
                        return result;
                    };
                    size_t quantized_size = size / ggml_blck_size(type) * ggml_type_size(type);
                    benchmark_function(size, quantized_size, quantize_fn);
                }
                printf("\n");
            }
        }
    }

    ggml_free(ctx);

    return 0;
}
