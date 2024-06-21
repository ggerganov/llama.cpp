#include "ggml.h"

#include "stdlib.h"   /* abort() */
#include <cstddef>
#include <cstdio>
#include <string>
#include <stdexcept>
#include <algorithm>

#include <string.h>

#include "xxhash/xxhash.h"
#include "sha1/sha1.h"

#ifdef SHA256 // TODO: https://github.com/jb55/sha256.c
#include "sha256/sha256.h"
#endif

// uuid.uuid5(uuid.NAMESPACE_URL, 'en.wikipedia.org/wiki/Llama.cpp')
#define UUID_NAMESPACE_LLAMA_CPP "ef001206-dadc-5f6d-a15f-3359e577d4e5"
#define UUID_NAMESPACE_LLAMA_CPP_HEX 0xef, 0x00, 0x12, 0x06, 0xda, 0xdc, 0x5f, 0x6d, 0xa1, 0x5f, 0x33, 0x59, 0xe5, 0x77, 0xd4, 0xe5

struct hash_params {
    std::string input;
    bool xxhash = false;
    bool sha1 = false;
    bool uuid = false;
#ifdef SHA256
    bool sha256 = false;
#endif
};

static void hash_print_usage(const char * executable) {
    const hash_params default_params;
    printf("\n");
    printf("usage: %s [options] GGUF_IN\n", executable);
    printf("\n");
    printf("Hash a GGUF file");
    printf("\n");
    printf("options:\n");
    printf("  -h, --help              show this help message and exit\n");
    printf("      --xxhash            use xxhash\n");
    printf("      --sha1              use sha1\n");
    printf("      --uuid              use uuid\n");
#ifdef SHA256
    printf("      --sha256            use sha256\n");
#endif
    printf("\n");
}

static void hash_params_parse_ex(int argc, const char ** argv, hash_params & params) {
    std::string arg;
    const std::string arg_prefix = "--";

    int arg_idx = 1;
    for (; arg_idx < argc && strncmp(argv[arg_idx], "--", 2) == 0; arg_idx++) {
        arg = argv[arg_idx];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        bool arg_found = false;
        if (arg == "-h" || arg == "--help") {
            hash_print_usage(argv[0]);
            exit(0);
        }

        if (arg == "--xxhash") {
            arg_found = true;
            params.xxhash = true;
        }

        if (arg == "--sha1") {
            arg_found = true;
            params.sha1 = true;
        }

        if (arg == "--uuid") {
            arg_found = true;
            params.uuid = true;
        }

#ifdef SHA256
        if (arg == "--sha256") {
            arg_found = true;
            params.sha256 = true;
        }
#endif

        if (!arg_found) {
            throw std::invalid_argument("error: unknown argument: " + arg);
        }
    }

    if (!params.xxhash
            && !params.sha1
            && !params.uuid
#ifdef SHA256
            && !params.sha256
#endif
        ) {
        // By default if no swich argument provided, assume xxhash
        params.xxhash = true;
    }

    if (argc - arg_idx < 1) {
        throw std::invalid_argument("error: bad arguments");
    }

    params.input = argv[arg_idx++];
}

static bool hash_params_parse(int argc, const char ** argv, hash_params & params) {
    bool result = true;
    try {
        hash_params_parse_ex(argc, argv, params);
    }
    catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        hash_print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return result;
}

static bool gguf_hash(const hash_params & hash_params) {
    const std::string & fname = hash_params.input;
    struct ggml_context * ctx_data = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_data,
    };

    // xxhash init
    XXH64_state_t* xxhash_model_hash_state = NULL;
    if (hash_params.xxhash) {
        xxhash_model_hash_state = XXH64_createState();
        if (xxhash_model_hash_state==NULL) {
            abort();
        }

        XXH64_hash_t const seed = 0;
        if (XXH64_reset(xxhash_model_hash_state, seed) == XXH_ERROR) {
            abort();
        }
    }

    // sha1 init
    SHA1_CTX sha1_model_hash_ctx;
    if (hash_params.sha1) {
        SHA1Init(&sha1_model_hash_ctx);
    }

#ifdef SHA256
    // sha256 init
    sha256_t sha256_model_hash_ctx;
    if (hash_params.sha256) {
        sha256_init(&sha256_model_hash_ctx);
    }
#endif

    struct gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);
    const int n_tensors = gguf_get_n_tensors(ctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name);
        auto n_bytes = ggml_nbytes(cur);
        auto *raw_data = cur->data;

        if (hash_params.xxhash) {

            // Per Layer Hash
            XXH64_hash_t hash = XXH64(raw_data, n_bytes, 0);

            char hex_result[17];
            for (int  offset = 0; offset < 8; offset++) {
                unsigned int shift_bits_by = (8 * (8 - offset - 1));
                sprintf( ( hex_result + (2*offset)), "%02x", (unsigned char) (hash >> shift_bits_by)&0xff);
            }

            printf("xxhash  %s  %s:%s\n", hex_result, fname.c_str(), name);

            // Overall Model Hash
            if (XXH64_update(xxhash_model_hash_state, raw_data, n_bytes) == XXH_ERROR) abort();
        }

        if (hash_params.sha1) {

            // Per Layer Hash
            char result[21]; // sha1 outputs 20 bytes
            SHA1( result, (const char *)raw_data, n_bytes);

            char hex_result[41] = {0};
            for (int  offset = 0; offset < 20; offset++) {
                sprintf( ( hex_result + (2*offset)), "%02x", result[offset]&0xff);
            }

            printf("sha1    %s  %s:%s\n", hex_result, fname.c_str(), name);

            // Overall Model Hash
            SHA1Update( &sha1_model_hash_ctx, (unsigned char const *)raw_data, n_bytes);
        }

#ifdef SHA256
        if (hash_params.sha256) {

            // Per Layer Hash
            unsigned char result[SHA256_DIGEST_SIZE]; // sha256 outputs 32 bytes
            sha256_hash( result, (const unsigned char *)raw_data, n_bytes);

            char hex_result[SHA256_DIGEST_SIZE * 2 + 1] = {0};
            for (int  offset = 0; offset < SHA256_DIGEST_SIZE; offset++) {
                sprintf( ( hex_result + (2*offset)), "%02x", result[offset]&0xff);
            }

            printf("sha256  %s  %s:%s\n", hex_result, fname.c_str(), name);

            // Overall Model Hash
            sha256_update( &sha256_model_hash_ctx, (unsigned char const *)raw_data, n_bytes);
        }
#endif
    }

    if (hash_params.xxhash) {
        XXH64_hash_t const hash = XXH64_digest(xxhash_model_hash_state);

        char hex_result[17];
        for (int  offset = 0; offset < 8; offset++) {
            unsigned int shift_bits_by = (8 * (8 - offset - 1));
            sprintf( ( hex_result + (2*offset)), "%02x", (unsigned char) (hash >> shift_bits_by)&0xff);
        }

        printf("xxhash  %s  %s\n", hex_result, fname.c_str());
    }

    if (hash_params.sha1) {
        unsigned char result[21];
        SHA1Final(result, &sha1_model_hash_ctx);

        char hex_result[41];
        for (int  offset = 0; offset < 20; offset++) {
            sprintf( ( hex_result + (2*offset)), "%02x", result[offset]&0xff);
        }

        printf("sha1    %s  %s\n", hex_result, fname.c_str());
    }

#ifdef SHA256
    if (hash_params.sha256) {
        unsigned char result[SHA256_DIGEST_SIZE]; // sha256 outputs 32 bytes
        sha256_final( &sha256_model_hash_ctx,  result);

        char hex_result[SHA256_DIGEST_SIZE * 2 + 1] = {0};
        for (int  offset = 0; offset < SHA256_DIGEST_SIZE; offset++) {
            sprintf( ( hex_result + (2*offset)), "%02x", result[offset]&0xff);
        }

        printf("sha256  %s  %s\n", hex_result, fname.c_str());
    }
#endif

    ggml_free(ctx_data);
    gguf_free(ctx);

    return true;
}

static void generate_uuidv5(const unsigned char sha1_digest[20], unsigned char uuid[16]) {
    // Ref: https://www.rfc-editor.org/rfc/rfc9562.html#section-5.5
    // Assumes that digest was processed correctly with the expected namespace
    for (int i = 0; i < 16; i++) {
        uuid[i] = sha1_digest[i];
    }

    // Set bits corresponding to UUID ver 5
    uuid[ 6] &= ~(0xF << 4);
    uuid[ 6] |= (5 << 4);

    // Set bits corresponding to UUID variant 0b10XX
    uuid[ 8] &= ~(0xc << 4);
    uuid[ 8] |= (0x8 << 4);
}

static bool gguf_uuid(const hash_params & hash_params) {
    if (!hash_params.uuid) {
        return true;
    }

    const std::string & fname = hash_params.input;
    struct ggml_context * ctx_data = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_data,
    };

    // sha1 init
    SHA1_CTX sha1_model_hash_ctx;
    SHA1Init(&sha1_model_hash_ctx);

    unsigned char const uuidv5_namespace[] = {UUID_NAMESPACE_LLAMA_CPP_HEX};
    SHA1Update( &sha1_model_hash_ctx, (unsigned char const *)uuidv5_namespace, sizeof(uuidv5_namespace));

    struct gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);
    const int n_tensors = gguf_get_n_tensors(ctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name);
        auto n_bytes = ggml_nbytes(cur);
        auto *raw_data = cur->data;
        SHA1Update( &sha1_model_hash_ctx, (unsigned char const *)raw_data, n_bytes);
    }

    unsigned char result[21];
    SHA1Final(result, &sha1_model_hash_ctx);

    unsigned char uuid[16];
    generate_uuidv5(result, uuid);

    char string_buffer[37] = {0};
    sprintf(string_buffer, "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        uuid[0], uuid[1], uuid[2], uuid[3],
        uuid[4], uuid[5], uuid[6], uuid[7],
        uuid[8], uuid[9], uuid[10], uuid[11],
        uuid[12], uuid[13], uuid[14], uuid[15]);
    printf("UUIDv5  %s  %s\n", string_buffer, fname.c_str());

    ggml_free(ctx_data);
    gguf_free(ctx);

    return true;
}

int main(int argc, const char ** argv) {
    hash_params params;
    hash_params_parse(argc, argv, params);

    gguf_hash(params);
    gguf_uuid(params);

    return 0;
}
