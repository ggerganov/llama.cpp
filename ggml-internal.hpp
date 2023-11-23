struct ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;
    bool   no_alloc_save; // this is used to save the no_alloc state when using scratch buffers

    int    n_objects;

    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;

    struct ggml_scratch scratch;
    struct ggml_scratch scratch_save;

  ggml_context():
    mem_size(0),
    mem_buffer(0),
    mem_buffer_owned(0),
    no_alloc(0),
    no_alloc_save(0),
    n_objects(0),
    objects_begin(0),
    objects_end(0),
    scratch(),
    scratch_save()
  {
    
  }
};

struct ggml_context_container {
    bool used;

    struct ggml_context context;

  ggml_context_container(): used(0),context(){
    
  }
};

typedef double ggml_float;
typedef void * thread_ret_t;

#define MAX_FREE_BLOCKS 256

struct free_block {
    void * addr;
    size_t size;
};

struct ggml_tallocr {
    struct ggml_backend_buffer * buffer;
    bool buffer_owned;
    void * base;
    size_t alignment;

    int n_free_blocks;
    struct free_block free_blocks[MAX_FREE_BLOCKS];

    size_t max_size;

    bool measure;

#ifdef GGML_ALLOCATOR_DEBUG
    struct ggml_tensor * allocated_tensors[1024];
#endif
};


struct hash_node {
    int n_children;
    int n_views;
};

typedef struct ggml_tallocr * ggml_tallocr_t;
typedef struct ggml_gallocr * ggml_gallocr_t;

struct ggml_gallocr {
    ggml_tallocr_t talloc;
    struct ggml_hash_set hash_set;
    struct hash_node * hash_values;
    size_t hash_values_size;
    ggml_tallocr_t * hash_allocs;
    int * parse_seq;
    int parse_seq_len;
};

struct ggml_allocr {
    ggml_tallocr_t talloc;
    ggml_gallocr_t galloc;
};
