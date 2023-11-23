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
typedef int ggml_lock_t;
typedef pthread_t ggml_thread_t;
typedef int ggml_lock_t;
typedef pthread_t ggml_thread_t;
typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
typedef void * thread_ret_t;
typedef double ggml_float;

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#define ggml_lock_lock(x)    UNUSED(x)
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0



#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join



//typedef pthread_spinlock_t ggml_lock_t;

//#define ggml_lock_init(x) pthread_spin_init(x, PTHREAD_PROCESS_PRIVATE)
//#define ggml_lock_destroy pthread_spin_destroy
//#define ggml_lock_lock    pthread_spin_lock
//#define ggml_lock_unlock  pthread_spin_unlock



#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
#define ggml_lock_lock(x)    _mm_pause()
#else
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0



#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join



