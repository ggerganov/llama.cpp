// TODO: convert to proper gguf.h gguf.c structure, now I'm trying to be fast as much as possible,
// and everything is in this file for quick debugging.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>


enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 (5) support has been removed
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    // k-quantizations
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_COUNT,
};

enum gguf_metadata_value_type {
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
};

struct gguf_string_t {
    uint32_t len;
    char * string;
};

union gguf_metadata_value_t;

// Union definition for gguf_metadata_value_t
union gguf_metadata_value_t {
    uint8_t uint8;
    int8_t int8;
    uint16_t uint16;
    int16_t int16;
    uint32_t uint32;
    int32_t int32;
    float float32;
    bool bool_;
    struct gguf_string_t string;
    struct {
        uint32_t len;
        enum gguf_metadata_value_type type;
        union gguf_metadata_value_t * array;
    } array;
};


struct gguf_metadata_kv_t {
    struct gguf_string_t key;
    uint32_t value_len;
    enum gguf_metadata_value_type value_type;
    union gguf_metadata_value_t* value;
};

struct gguf_header_t {
    uint32_t magic;
    uint32_t version;
    uint32_t tensor_count;
    uint32_t metadata_kv_count;
    struct gguf_metadata_kv_t * metadata_kv;
};

struct gguf_tensor_info_t {
    struct gguf_string_t name;
    uint32_t n_dimensions;
    uint32_t dimensions[];
};

struct gguf_file_t {
    struct gguf_header_t header;
    uint8_t tensor_data[];
};

void read_gguf_file(const char * file_path, struct gguf_file_t * gguf_file) {
    FILE* file = fopen(file_path, "rb");
    if (file == NULL) {
        printf("Error opening the file.\n");
        return;
    }

    fread(&gguf_file->header.magic, sizeof(uint32_t), 1, file);

    // Verify magic and version
    if (gguf_file->header.magic != 0x47475546) {
        printf("Invalid magic number. Not a valid GGUF file.\n");
        fclose(file);
        return;
    }

    fread(&gguf_file->header.version, sizeof(uint32_t), 1, file);

    if (gguf_file->header.version != 1) {
        printf("Unsupported version. Expected version 1.\n");
        fclose(file);
        return;
    }

    fread(&gguf_file->header.tensor_count, sizeof(uint32_t), 1, file);
    fread(&gguf_file->header.metadata_kv_count, sizeof(uint32_t), 1, file);

    printf("Magic: %x\n", gguf_file->header.magic);
    printf("Version: %d\n", gguf_file->header.version);
    printf("Tensor Count: %d\n", gguf_file->header.tensor_count);
    printf("Metadata Key-Value Count: %d\n", gguf_file->header.metadata_kv_count);

    gguf_file->header.metadata_kv = (struct gguf_metadata_kv_t*)malloc(gguf_file->header.metadata_kv_count * sizeof(struct gguf_metadata_kv_t));

    for (int i = 0; i < gguf_file->header.metadata_kv_count; i++) {
        struct gguf_metadata_kv_t* kv = &gguf_file->header.metadata_kv[i];
        fread(&kv->key.len, sizeof(uint32_t), 1, file);
        kv->key.string = (char*)malloc(kv->key.len ); // Allocate memory for the key string
        fread(kv->key.string, sizeof(char), kv->key.len, file);
        //kv->key.string[kv->key.len] = '\0'; // Null-terminate the key string

        fread(&kv->value_type, sizeof(uint32_t), 1, file);

        printf("Metadata Value Type: %d\n", kv->value_type);
        printf("Metadata Key: %s\n", kv->key.string);

        // Read metadata value according to its type using reinterpret_cast
        switch (kv->value_type) {
            case GGUF_METADATA_VALUE_TYPE_UINT32:
            kv->value = (uint32_t *) malloc(sizeof(uint32_t));
            fread(kv->value, sizeof(uint32_t), 1, file);
            printf("value: %d\n", kv->value->uint32);
            break;
            case GGUF_METADATA_VALUE_TYPE_FLOAT32:
            kv->value = (float *)malloc(sizeof(float));
            fread(kv->value, sizeof(float), 1, file);
            printf("value: %f\n", (float)kv->value->float32);
            break;
            case GGUF_METADATA_VALUE_TYPE_STRING:
            fread(&kv->value_len, sizeof(uint32_t), 1, file);
            printf("value len: %d\n", kv->value_len);
kv->value = (char *)malloc(sizeof(char) * kv->value_len); // Allocate memory for the value string
fread(kv->value, sizeof(char), kv->value_len, file);
        printf("value: %s\n", (char *)kv->value);
        break;
            // ... (handle other types in a similar manner)
            default:
                printf("Unsupported metadata value type.\n");
                fclose(file);
                return;
        }
    }

    // TODO: handle reading tensor data

    fclose(file);
}

void gguf_free(struct gguf_file_t * gguf_file) {
    // Free allocated memory for key strings avd values
    for (int i = 0; i < gguf_file->header.metadata_kv_count; i++) {
        free(gguf_file->header.metadata_kv[i].key.string);
        free(gguf_file->header.metadata_kv[i].value);
    }
    free(gguf_file->header.metadata_kv);
}

int main() {
    const char* file_path = "example.gguf";
    struct gguf_file_t gguf_file;
    read_gguf_file(file_path, &gguf_file);
    gguf_free(&gguf_file);
    return 0;
}
