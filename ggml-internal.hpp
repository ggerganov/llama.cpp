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

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

struct ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct ggml_numa_nodes {
    struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
};

struct ggml_state {
    struct ggml_context_container contexts[GGML_MAX_CONTEXTS];
    struct ggml_numa_nodes numa;

  ggml_state():contexts(), numa()
  {
    
  }
};

struct gguf_str {
    uint64_t n;  // GGUFv2
    char * data;
};

struct ggml_map_custom1_op_params {
    ggml_custom1_op_t fun;
    int n_tasks;
    void * userdata;
};

struct ggml_map_custom2_op_params {
    ggml_custom2_op_t fun;
    int n_tasks;
    void * userdata;
};

struct ggml_map_custom3_op_params {
    ggml_custom3_op_t fun;
    int n_tasks;
    void * userdata;
};
struct hash_map {
    struct ggml_hash_set set;
    struct ggml_tensor ** vals;
};

#if defined(_WIN32)
typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
#else
#include<atomic>
using namespace std;
#endif

struct ggml_compute_state_shared {
    const struct ggml_cgraph * cgraph;
    const struct ggml_cplan  * cplan;

    int64_t perf_node_start_cycles;
    int64_t perf_node_start_time_us;

    const int n_threads;

    // synchronization primitives
    atomic_int n_active; // num active threads
    atomic_int node_n;   // active graph node

    bool (*abort_callback)(void * data); // abort ggml_graph_compute when true
    void * abort_callback_data;
};
typedef pthread_t ggml_thread_t;
struct ggml_compute_state {
    ggml_thread_t thrd;
    int ith;
    struct ggml_compute_state_shared * shared;
};

union gguf_value {
    uint8_t  uint8;
    int8_t   int8;
    uint16_t uint16;
    int16_t  int16;
    uint32_t uint32;
    int32_t  int32;
    float    float32;
    uint64_t uint64;
    int64_t  int64;
    double   float64;
    bool     bool_;

    struct gguf_str str;

    struct gguf_array_T {
        enum gguf_type type;

        uint64_t n;  // GGUFv2
        void * data;
    } arr;
};

struct ggml_lbfgs_iteration_data {
    float alpha;
    float ys;
    float * s;
    float * y;
};

struct gguf_kv {
    struct gguf_str key;

    enum  gguf_type  type;
    union gguf_value value;
};



struct gguf_header {
    char magic[4];
    uint32_t version;
    uint64_t n_tensors; // GGUFv2
    uint64_t n_kv;      // GGUFv2
};

struct gguf_tensor_info {
    struct gguf_str name;

    uint32_t n_dims;
    uint64_t ne[GGML_MAX_DIMS];

    enum ggml_type type;

    uint64_t offset; // offset from start of `data`, must be a multiple of `ALIGNMENT`

    // for writing API
    const void * data;
    size_t size;
};

struct gguf_context {
    struct gguf_header header;

    struct gguf_kv          * kv;
    struct gguf_tensor_info * infos;

    size_t alignment;
    size_t offset;    // offset of `data` from beginning of file
    size_t size;      // size of `data` in bytes

    //uint8_t * padding;
    void * data;
};

struct gguf_buf {
    void * data;
    size_t size;
    size_t offset;
};


#include "ggml-backend-impl.h"
