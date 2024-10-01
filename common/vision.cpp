#include "vision.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <vector>
#include <fstream>

llama_img * load_image_from_file(const char * fname) {
    std::ifstream file(fname, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open file");
    }
    std::vector<char> image_bytes = std::vector<char>(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>());
    // decode image to byte array
    int nx, ny, nc;
    auto * bytes = (unsigned char *) image_bytes.data();
    auto * img = stbi_load_from_memory(bytes, image_bytes.size(), &nx, &ny, &nc, 3);
    if (!img) {
        throw std::runtime_error("failed to decode image bytes");
    }
    // printf("nx=%d ny=%d nc=%d\n", nx, ny, nc);
    // GGML_ASSERT(nc == 3);
    // for (int y = 0; y < ny; y++) {
    //     for (int x = 0; x < nx; x++) {
    //         unsigned char * pix = img + x*nc + y*nc*nx;
    //         printf("%02x%02x%02x ", pix[0], pix[1], pix[2]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    llama_img * result = llama_img_alloc(nx, ny);
    memcpy(result->data, img, nx*ny*3);
    stbi_image_free(img);
    return result;
}
