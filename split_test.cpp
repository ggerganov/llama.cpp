#include <iostream>
#include "ggml.h"

int main() {
    printf("split_test\n");
    // Initialization
    struct ggml_init_params params = ggml_init_params{1024};  // Assuming this initializes memory
    ggml_context *ctx = ggml_init(params);

    // Tensor Creation (Analogous to the PyTorch code)
    int64_t size = 18 * 7 * 64;
    int64_t dims[4] = {1, 18, 7, 64};
    ggml_tensor *tensor = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, dims);

    // Initialize tensor data (Note: Simplified for this example)
    float* tensor_data = (float*) tensor->data;
    for (int i = 0; i < size; i++) {
        tensor_data[i] = (float) i;
        printf("%f", tensor_data[i]);
    }
    printf("\n");

    // Reshaping and Transpose
    // ... (You'll need ggml equivalents of reshape and transpose)

    // Splitting (We'll focus on this part)
    int64_t num_q_heads = 12;
    int64_t num_k_heads = 3;
    int64_t num_v_heads = 3;

    ggml_tensor *a = ggml_view_3d(ctx, tensor, /*ne0*/1, /*ne1*/2, /*ne2*/3, /*nb1*/4, /*nb2*/5, /*offset*/6);
    ggml_tensor *b = ggml_view_3d(ctx, tensor, /*ne0*/1, /*ne1*/2, /*ne2*/3, /*nb1*/4, /*nb2*/5, /*offset*/6);
    ggml_tensor *c = ggml_view_3d(ctx, tensor, /*ne0*/1, /*ne1*/2, /*ne2*/3, /*nb1*/4, /*nb2*/5, /*offset*/6);

    // Accessing elements (assuming ggml provides similar access)
    float *a_data = (float*) a->data;
    std::cout << a_data[0] << std::endl;

    return 0;
}
