
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "../pca.hpp"

#include "ggml-cpp.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>

// Function to run PCA and print results
static void run_pca_test(float *matrix, int rows, int cols) {
    // Initialize ggml context
    size_t ctx_size = 0;
    ctx_size += rows * cols * ggml_type_size(GGML_TYPE_F32);
    ctx_size += 1 * ggml_tensor_overhead();

    struct ggml_init_params ctx_params {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(ctx_params);
    struct ggml_tensor *input_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, cols);
    memcpy(input_tensor->data, matrix, rows * cols * sizeof(float));

    PCA::pca_params pca_params;
    pca_params.n_threads = 8;
    pca_params.n_batch = 20;
    pca_params.n_iterations = 1000;
    pca_params.tolerance = 1e-5;

    PCA::pca_result result = {NULL, 0};
    PCA::run_single_pca(pca_params, input_tensor, result);

    printf("Principal components:\n");
    for (int i = 0; i < cols; i++) {
        printf("%f ", result.principal_component[i]);
    }
    printf("\nEigenvalue: %f\n", result.explained_variance);

    free(result.principal_component);
    ggml_free(ctx);
}

int main() {
    // Define matrices
    float input_matrix1[16] = {
        -0.124132, 0.740341, -0.452462, 0.777050,
        1.045571, -0.342142, -0.926047, -0.512965,
        0.710109, 0.092479, 0.630075, 1.762937,
        0.230954, -0.808937, 1.057424, 0.051361
    };

    float input_matrix2[100] = {
        440152.493740, 122038.234845, 495176.910111, 34388.521115, 909320.402079, 258779.981600, 662522.284354, 311711.076089, 520068.021178, 546710.279343,
        184854.455526, 969584.627765, 775132.823361, 939498.941564, 894827.350428, 597899.978811, 921874.235023, 88492.502052, 195982.862419, 45227.288911,
        325330.330763, 388677.289689, 271349.031774, 828737.509152, 356753.326694, 280934.509687, 542696.083158, 140924.224975, 802196.980754, 74550.643680,
        986886.936601, 772244.769297, 198715.681534, 5522.117124, 815461.428455, 706857.343848, 729007.168041, 771270.346686, 74044.651734, 358465.728544,
        115869.059525, 863103.425876, 623298.126828, 330898.024853, 63558.350286, 310982.321716, 325183.322027, 729606.178338, 637557.471355, 887212.742576,
        472214.925162, 119594.245938, 713244.787223, 760785.048617, 561277.197569, 770967.179955, 493795.596364, 522732.829382, 427541.018359, 25419.126744,
        107891.426993, 31429.185687, 636410.411264, 314355.981076, 508570.691165, 907566.473926, 249292.229149, 410382.923036, 755551.138543, 228798.165492,
        76979.909829, 289751.452914, 161221.287254, 929697.652343, 808120.379564, 633403.756510, 871460.590188, 803672.076899, 186570.058886, 892558.998490,
        539342.241916, 807440.155164, 896091.299923, 318003.474972, 110051.924528, 227935.162542, 427107.788626, 818014.765922, 860730.583256, 6952.130531,
        510747.302578, 417411.003149, 222107.810471, 119865.367334, 337615.171404, 942909.703913, 323202.932021, 518790.621743, 703018.958895, 363629.602379
    };

    float input_matrix3[9] = {
        0.374540, 0.950714, 0.731994,
        0.598658, 0.156019, 0.155995,
        0.058084, 0.866176, 0.601115
    };

    float input_matrix4[9] = {
        10.000000, 0.000000, 0.000000,
        0.000000, 5.000000, 0.000000,
        0.000000, 0.000000, 1.000000
    };

    // Run PCA for each matrix
    printf("Testing Matrix 1:\n");
    run_pca_test(input_matrix1, 4, 4);

    printf("\nTesting Matrix 2:\n");
    run_pca_test(input_matrix2, 10, 10);

    printf("\nTesting Matrix 3:\n");
    run_pca_test(input_matrix3, 3, 3);

    printf("\nTesting Matrix 4:\n");
    run_pca_test(input_matrix4, 3, 3);

    // Cleanup
    return 0;
}

