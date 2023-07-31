#include "ggml.h"
#include "llama-util.h"

#include <cstdio>
#include <cinttypes>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>

template<typename T>
static std::string to_string(const T & val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
}

void gguf_ex_write_str(std::ofstream & fout, const std::string & val) {
    const int32_t n = val.size();
    fout.write((const char *) &n, sizeof(n));
    fout.write(val.c_str(), n);
}

void gguf_ex_write_i32(std::ofstream & fout, int32_t val) {
    fout.write((const char *) &val, sizeof(val));
}

void gguf_ex_write_u64(std::ofstream & fout, size_t val) {
    fout.write((const char *) &val, sizeof(val));
}

template<typename T>
void gguf_ex_write_val(std::ofstream & fout, const std::string & key, enum gguf_type type, const T & val) {
    gguf_ex_write_str(fout, key);
    fout.write((const char *) &type, sizeof(type));
    fout.write((const char *) &val,  sizeof(val));

    fprintf(stdout, "%s: write param: %s = %s\n", __func__, key.c_str(), to_string(val).c_str());
}

template<>
void gguf_ex_write_val<std::string>(std::ofstream & fout, const std::string & key, enum gguf_type type, const std::string & val) {
    gguf_ex_write_str(fout, key);
    fout.write((const char *) &type, sizeof(type));

    const int32_t n = val.size();
    fout.write((const char *) &n, sizeof(n));
    fout.write(val.c_str(), n);

    fprintf(stdout, "%s: write param: %s = %s\n", __func__, key.c_str(), val.c_str());
}

template<typename T>
void gguf_ex_write_arr(std::ofstream & fout, const std::string & key, enum gguf_type type, const std::vector<T> & val) {
    gguf_ex_write_str(fout, key);
    {
        const enum gguf_type tarr = GGUF_TYPE_ARRAY;
        fout.write((const char *) &tarr, sizeof(tarr));
    }

    const int32_t n = val.size();
    fout.write((const char *) &type, sizeof(type));
    fout.write((const char *) &n,    sizeof(n));
    fout.write((const char *) val.data(), n * sizeof(T));

    fprintf(stdout, "%s: write param: %s = [", __func__, key.c_str());
    for (int i = 0; i < n; ++i) {
        fprintf(stdout, "%s", to_string(val[i]).c_str());
        if (i < n - 1) {
            fprintf(stdout, ", ");
        }
    }
    fprintf(stdout, "]\n");
}

template<>
void gguf_ex_write_arr<std::string>(std::ofstream & fout, const std::string & key, enum gguf_type type, const std::vector<std::string> & val) {
    gguf_ex_write_str(fout, key);
    {
        const enum gguf_type tarr = GGUF_TYPE_ARRAY;
        fout.write((const char *) &tarr, sizeof(tarr));
    }

    const int32_t n = val.size();
    fout.write((const char *) &type, sizeof(type));
    fout.write((const char *) &n,    sizeof(n));
    for (int i = 0; i < n; ++i) {
        const int32_t nstr = val[i].size();
        fout.write((const char *) &nstr, sizeof(nstr));
        fout.write(val[i].c_str(), nstr);
    }

    fprintf(stdout, "%s: write param: %s = [", __func__, key.c_str());
    for (int i = 0; i < n; ++i) {
        fprintf(stdout, "%s", val[i].c_str());
        if (i < n - 1) {
            fprintf(stdout, ", ");
        }
    }
    fprintf(stdout, "]\n");
}

bool gguf_ex_write(const std::string & fname) {
    std::ofstream fout(fname.c_str(), std::ios::binary);

    {
        const int32_t magic = GGUF_MAGIC;
        fout.write((const char *) &magic, sizeof(magic));
    }

    {
        const int32_t version = GGUF_VERSION;
        fout.write((const char *) &version, sizeof(version));
    }

    // NOTE: these have to match the output below!
    const int n_tensors = 10;
    const int n_kv      = 12;

    fout.write((const char*) &n_tensors, sizeof(n_tensors));
    fout.write((const char*) &n_kv, sizeof(n_kv));

    fprintf(stdout, "%s: write header\n", __func__);

    // kv data
    {
        gguf_ex_write_val< uint8_t>(fout, "some.parameter.uint8",   GGUF_TYPE_UINT8,   0x12);
        gguf_ex_write_val<  int8_t>(fout, "some.parameter.int8",    GGUF_TYPE_INT8,   -0x13);
        gguf_ex_write_val<uint16_t>(fout, "some.parameter.uint16",  GGUF_TYPE_UINT16,  0x1234);
        gguf_ex_write_val< int16_t>(fout, "some.parameter.int16",   GGUF_TYPE_INT16,  -0x1235);
        gguf_ex_write_val<uint32_t>(fout, "some.parameter.uint32",  GGUF_TYPE_UINT32,  0x12345678);
        gguf_ex_write_val< int32_t>(fout, "some.parameter.int32",   GGUF_TYPE_INT32,  -0x12345679);

        gguf_ex_write_val<float>   (fout, "some.parameter.float32", GGUF_TYPE_FLOAT32, 0.123456789f);
        gguf_ex_write_val<bool>    (fout, "some.parameter.bool",    GGUF_TYPE_BOOL,    true);

        gguf_ex_write_val<std::string>(fout, "some.parameter.string",  GGUF_TYPE_STRING,  "hello world");

        gguf_ex_write_arr<int16_t>    (fout, "some.parameter.arr.i16", GGUF_TYPE_INT16,   { 1, 2, 3, 4, });
        gguf_ex_write_arr<float>      (fout, "some.parameter.arr.f32", GGUF_TYPE_FLOAT32, { 3.145f, 2.718f, 1.414f, });
        gguf_ex_write_arr<std::string>(fout, "some.parameter.arr.str", GGUF_TYPE_STRING,  { "hello", "world", "!" });
    }

    uint64_t offset_tensor = 0;

    struct ggml_init_params params = {
        /*.mem_size   =*/ 128ull*1024ull*1024ull,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx_data = ggml_init(params);

    // tensor infos
    for (int i = 0; i < n_tensors; ++i) {
        const std::string name = "tensor_" + to_string(i);

        int64_t ne[GGML_MAX_DIMS] = { 1 };
        int32_t n_dims = rand() % GGML_MAX_DIMS + 1;

        for (int j = 0; j < n_dims; ++j) {
            ne[j] = rand() % 10 + 1;
        }

        struct ggml_tensor * cur = ggml_new_tensor(ctx_data, GGML_TYPE_F32, n_dims, ne);
        ggml_set_name(cur, name.c_str());

        {
            float * data = (float *) cur->data;
            for (int j = 0; j < ggml_nelements(cur); ++j) {
                data[j] = 100 + i;
            }
        }

        fprintf(stdout, "%s: tensor: %s, %d dims, ne = [", __func__, name.c_str(), n_dims);
        for (int j = 0; j < 4; ++j) {
            fprintf(stdout, "%s%3d", j == 0 ? "" : ", ", (int) cur->ne[j]);
        }
        fprintf(stdout, "], offset_tensor = %6" PRIu64 "\n", offset_tensor);

        gguf_ex_write_str(fout, name);
        gguf_ex_write_i32(fout, n_dims);
        for (int j = 0; j < n_dims; ++j) {
            gguf_ex_write_i32(fout, cur->ne[j]);
        }
        gguf_ex_write_i32(fout, cur->type);
        gguf_ex_write_u64(fout, offset_tensor);

        offset_tensor += GGML_PAD(ggml_nbytes(cur), GGUF_DEFAULT_ALIGNMENT);
    }

    const uint64_t offset_data = GGML_PAD((uint64_t) fout.tellp(), GGUF_DEFAULT_ALIGNMENT);

    fprintf(stdout, "%s: data offset = %" PRIu64 "\n", __func__, offset_data);

    {
        const size_t pad = offset_data - fout.tellp();

        for (size_t j = 0; j < pad; ++j) {
            fout.put(0);
        }
    }

    for (int i = 0; i < n_tensors; ++i) {
        fprintf(stdout, "%s: writing tensor %d data\n", __func__, i);

        const std::string name = "tensor_" + to_string(i);

        struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name.c_str());

        fout.write((const char *) cur->data, ggml_nbytes(cur));

        {
            const size_t pad = GGML_PAD(ggml_nbytes(cur), GGUF_DEFAULT_ALIGNMENT) - ggml_nbytes(cur);

            for (size_t j = 0; j < pad; ++j) {
                fout.put(0);
            }
        }
    }

    fout.close();

    fprintf(stdout, "%s: wrote file '%s;\n", __func__, fname.c_str());

    ggml_free(ctx_data);

    return true;
}

// just read tensor info
bool gguf_ex_read_0(const std::string & fname) {
    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ NULL,
    };

    struct gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);

    fprintf(stdout, "%s: version:      %d\n", __func__, gguf_get_version(ctx));
    fprintf(stdout, "%s: alignment:   %zu\n", __func__, gguf_get_alignment(ctx));
    fprintf(stdout, "%s: data offset: %zu\n", __func__, gguf_get_data_offset(ctx));

    // kv
    {
        const int n_kv = gguf_get_n_kv(ctx);

        fprintf(stdout, "%s: n_kv: %d\n", __func__, n_kv);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = gguf_get_key(ctx, i);

            fprintf(stdout, "%s: kv[%d]: key = %s\n", __func__, i, key);
        }
    }

    // find kv string
    {
        char findkey[32];
        sprintf(findkey, "some.parameter.string");

        int keyidx = gguf_find_key(ctx, findkey);
        if (keyidx == -1) {
            fprintf(stdout, "%s: find key: %s not found.\n", __func__, findkey);
        } else {
            const char * key_value = gguf_get_val_str(ctx, keyidx);
            fprintf(stdout, "%s: find key: %s found, kv[%d] value = %s\n", __func__, findkey, keyidx, key_value);
        }
    }

    // tensor info
    {
        const int n_tensors = gguf_get_n_tensors(ctx);

        fprintf(stdout, "%s: n_tensors: %d\n", __func__, n_tensors);

        for (int i = 0; i < n_tensors; ++i) {
            const char * name   = gguf_get_tensor_name  (ctx, i);
            const size_t offset = gguf_get_tensor_offset(ctx, i);

            fprintf(stdout, "%s: tensor[%d]: name = %s, offset = %zu\n", __func__, i, name, offset);
        }
    }

    gguf_free(ctx);

    return true;
}

// read and create ggml_context containing the tensors and their data
bool gguf_ex_read_1(const std::string & fname) {
    struct ggml_context * ctx_data = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_data,
    };

    struct gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);
    
    fprintf(stdout, "%s: version:      %d\n", __func__, gguf_get_version(ctx));
    fprintf(stdout, "%s: alignment:   %zu\n", __func__, gguf_get_alignment(ctx));
    fprintf(stdout, "%s: data offset: %zu\n", __func__, gguf_get_data_offset(ctx));

    // kv
    {
        const int n_kv = gguf_get_n_kv(ctx);

        fprintf(stdout, "%s: n_kv: %d\n", __func__, n_kv);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = gguf_get_key(ctx, i);

            fprintf(stdout, "%s: kv[%d]: key = %s\n", __func__, i, key);
        }
    }

    // tensor info
    {
        const int n_tensors = gguf_get_n_tensors(ctx);

        fprintf(stdout, "%s: n_tensors: %d\n", __func__, n_tensors);

        for (int i = 0; i < n_tensors; ++i) {
            const char * name   = gguf_get_tensor_name  (ctx, i);
            const size_t offset = gguf_get_tensor_offset(ctx, i);

            fprintf(stdout, "%s: tensor[%d]: name = %s, offset = %zu\n", __func__, i, name, offset);
        }
    }

    // data
    {
        const int n_tensors = gguf_get_n_tensors(ctx);

        for (int i = 0; i < n_tensors; ++i) {
            fprintf(stdout, "%s: reading tensor %d data\n", __func__, i);

            const char * name = gguf_get_tensor_name(ctx, i);

            struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name);

            fprintf(stdout, "%s: tensor[%d]: n_dims = %d, name = %s, data = %p\n",
                    __func__, i, cur->n_dims, cur->name, cur->data);

            // check data
            {
                const float * data = (const float *) cur->data;
                for (int j = 0; j < ggml_nelements(cur); ++j) {
                    if (data[j] != 100 + i) {
                        fprintf(stderr, "%s: tensor[%d]: data[%d] = %f\n", __func__, i, j, data[j]);
                        return false;
                    }
                }
            }
        }
    }

    fprintf(stdout, "%s: ctx_data size: %zu\n", __func__, ggml_get_mem_size(ctx_data));

    ggml_free(ctx_data);
    gguf_free(ctx);

    return true;
}

// read just the tensor info and mmap the data in user code
bool gguf_ex_read_2(const std::string & fname) {
    struct ggml_context * ctx_data = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_data,
    };

    struct gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);

    // TODO: mmap based on tensor infos

    
    struct llama_file file(fname.c_str(), "rb");
    llama_mmap data_mmap(&file, 0, false);
    const int n_tensors = gguf_get_n_tensors(ctx);

    for (int i = 0; i < n_tensors; ++i) {
        const char * name             = gguf_get_tensor_name(ctx, i);
        const size_t offset      = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i);
        struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name);

        cur->data = static_cast<char *>(data_mmap.addr) + offset;

        // print first 10 elements
    const float * data = (const float *) cur->data;
                
        printf("%s data[:10] : ", name);

        for (int j = 0; j < 10; ++j) {
            printf("%f ", data[j]);
        }

        printf("\n\n");
    }

fprintf(stdout, "%s: ctx_data size: %zu\n", __func__, ggml_get_mem_size(ctx_data));

    ggml_free(ctx_data);
    gguf_free(ctx);

    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stdout, "usage: %s data.gguf r|w\n", argv[0]);
        return -1;
    }

    const std::string fname(argv[1]);
    const std::string mode (argv[2]);

    GGML_ASSERT((mode == "r" || mode == "w") && "mode must be r or w");

    if (mode == "w") {
        GGML_ASSERT(gguf_ex_write(fname) && "failed to write gguf file");
    } else if (mode == "r") {
        GGML_ASSERT(gguf_ex_read_0(fname) && "failed to read gguf file");
        GGML_ASSERT(gguf_ex_read_1(fname) && "failed to read gguf file");
        GGML_ASSERT(gguf_ex_read_2(fname) && "failed to read gguf file");
    }

    return 0;
}
