#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include "ggml.h"
#include "llama.h"
#include "get-model.h"

// run `python3 gguf-py/tests/test_gguf.py` to generate test_writer.gguf file.
int main(int argc, char ** argv)
{
  char* fname = get_model_or_exit(argc, argv);

  struct gguf_context * ctx_gguf = NULL;
  struct ggml_context * ctx_meta = NULL;
  struct gguf_init_params params = {
      /*.no_alloc = */ true,
      /*.ctx      = */ &ctx_meta,
  };
  ctx_gguf = gguf_init_from_file(fname, params);
  if (!ctx_gguf) {
    fprintf(stderr, "%s: failed to load model from %s\n", __func__, fname);
    return 1;
  }
  int n_kv      = gguf_get_n_kv(ctx_gguf);

  for (int i = 0; i < n_kv; i++) {
    const char * name           = gguf_get_key(ctx_gguf, i);
    // skip the subkeys.
    if (name[0] == '.') { continue; }
    const enum gguf_type type   = gguf_get_kv_type(ctx_gguf, i);
    char * value = gguf_kv_to_c_str(ctx_gguf, i, name);
    printf("key: %s, type: %s, value: %s\n", name, gguf_type_name(type), value);
    free(value);
  }

  int k_id = gguf_find_key(ctx_gguf, "no_such_key");
  assert(k_id == -1);
  k_id = gguf_find_key(ctx_gguf, "tokenizer_config");
  assert(k_id != -1);

  const char * name           = gguf_get_key(ctx_gguf, k_id);
  assert(strcmp(name, "tokenizer_config") == 0);

  enum gguf_type type   = gguf_get_kv_type(ctx_gguf, k_id);
  assert(type == GGUF_TYPE_OBJ);
  char * value = gguf_kv_to_c_str(ctx_gguf, k_id, NULL);
  assert(strcmp(value, "{\"bos_token\":\"bos\", \"add_bos_token\":true}") == 0);
  free(value);

  k_id = gguf_find_key(ctx_gguf, "dict1");
  assert(k_id != -1);
  value = gguf_kv_to_c_str(ctx_gguf, k_id, NULL);
  assert(strcmp(value, "{\"key1\":2, \"key2\":\"hi\", \"obj\":{\"k\":1}}") == 0);
  free(value);

  k_id = gguf_find_key(ctx_gguf, "oArray");
  assert(k_id != -1);
  value = gguf_kv_to_c_str(ctx_gguf, k_id, NULL);
  assert(strcmp(value, "[{\"k\":4, \"o\":{\"o1\":6}}, {\"k\":9}]") == 0);
  free(value);

  k_id = gguf_find_key(ctx_gguf, "cArray");
  assert(k_id != -1);
  value = gguf_kv_to_c_str(ctx_gguf, k_id, NULL);
  assert(strcmp(value, "[3, \"hi\", [1, 2]]") == 0);
  free(value);

  k_id = gguf_find_key(ctx_gguf, "arrayInArray");
  assert(k_id != -1);
  type   = gguf_get_kv_type(ctx_gguf, k_id);
  assert(type == GGUF_TYPE_ARRAY);
  value = gguf_kv_to_c_str(ctx_gguf, k_id, NULL);
  assert(strcmp(value, "[[2, 3, 4], [5, 7, 8]]") == 0);
  free(value);

  printf("Done!\n");
}
