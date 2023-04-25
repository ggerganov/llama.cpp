// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>


const float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001;
const float MAX_QUANTIZATION_TOTAL_ERROR = 0.002;
const float MAX_DOT_PRODUCT_ERROR = 0.02;

const char* RESULT_STR[] = {"ok", "FAILED"};


// Generate synthetic data
void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

// Calculate RMSE between two float arrays
float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

// Total quantization error on test data
float total_quantization_error(quantize_fns_t & qfns, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);

    qfns.quantize_row_q(test_data, tmp_q.data(), test_size);
    qfns.dequantize_row_q(tmp_q.data(), tmp_out.data(), test_size);
    return array_rmse(test_data, tmp_out.data(), test_size);
}

// Total quantization error on test data
float reference_quantization_error(quantize_fns_t & qfns, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    qfns.quantize_row_q(test_data, tmp_q.data(), test_size);
    qfns.dequantize_row_q(tmp_q.data(), tmp_out.data(), test_size);

    qfns.quantize_row_q_reference(test_data, tmp_q.data(), test_size);
    qfns.dequantize_row_q(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

// Total dot product error
float dot_product_error(quantize_fns_t & qfns, size_t test_size, const float * test_data1, const float *test_data2) {
    std::vector<uint8_t> tmp_q1(2*test_size);
    std::vector<uint8_t> tmp_q2(2*test_size);

    qfns.quantize_row_q    (test_data1, tmp_q1.data(), test_size);
    qfns.quantize_row_q_dot(test_data2, tmp_q2.data(), test_size);

    float result = INFINITY;
    qfns.vec_dot_q(test_size, &result, tmp_q1.data(), tmp_q2.data());

    const float dot_ref = dot_product(test_data1, test_data2, test_size);

    return fabsf(result - dot_ref) / test_size;
}

int main(int argc, char * argv[]) {
    bool verbose = false;
    const size_t test_size = 32 * 128;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());

    // Initialize GGML, ensures float conversion tables are initialized
    struct ggml_init_params ggml_params = {
        /* .mem_size   = */ 1*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(ggml_params);

    int num_failed = 0;
    bool failed = false;

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        quantize_fns_t qfns = ggml_internal_get_quantize_fn(i);

        if (qfns.quantize_row_q && qfns.dequantize_row_q) {
            const float total_error = total_quantization_error(qfns, test_size, test_data.data());
            failed = !(total_error < MAX_QUANTIZATION_TOTAL_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }

            const float reference_error = reference_quantization_error(qfns, test_size, test_data.data());
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
            }

            const float vec_dot_error = dot_product_error(qfns, test_size, test_data.data(), test_data2.data());
            failed = !(vec_dot_error < MAX_DOT_PRODUCT_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }
        }
    }

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    ggml_free(ctx);

    return num_failed > 0;
}
