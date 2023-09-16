#include "ggml-alloc.h"
#include "ggml.h"
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/types.h>
            #include <sys/mman.h>
        #endif
    #endif
#endif

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #include <memoryapi.h>
#endif


#define UNUSED(x) (void)(x)
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define GGML_MAX_CONCUR (2*GGML_MAX_NODES)

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF printf
#define AT_PRINTF(...) ((void)0)

struct hash_node {
    struct ggml_tensor * t;
    int n_children;
    int n_views;
};

static size_t hash(void * p) {
    return (size_t)p % GGML_GRAPH_HASHTABLE_SIZE;
}

static struct hash_node * hash_get(struct hash_node hash_table[], struct ggml_tensor * t) {
    size_t h = hash(t);

    // linear probing
    size_t i = h;
    while (hash_table[i].t != NULL) {
        if (hash_table[i].t == t) {
            return &hash_table[i];
        }
        i = (i + 1) % GGML_GRAPH_HASHTABLE_SIZE;
        if (i == h) {
            // hash table is full
            GGML_ASSERT(false);
        }
    }

    hash_table[i].t = t;
    return &hash_table[i];
}

// TODO: GGML_PAD ?
static size_t aligned_offset(const void * buffer, size_t offset, size_t alignment) {
    assert(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    return offset + align;
}

struct free_block {
    void * addr;
    size_t size;
};

#define MAX_FREE_BLOCKS 256

struct ggml_allocr {
    void * data;
    size_t size;
    size_t alignment;
    int n_free_blocks;
    struct free_block free_blocks[MAX_FREE_BLOCKS];
    struct hash_node hash_table[GGML_GRAPH_HASHTABLE_SIZE];
    size_t max_size;
    bool measure;
    int parse_seq[GGML_MAX_CONCUR];
    int parse_seq_len;

#ifdef GGML_ALLOCATOR_DEBUG
    struct ggml_tensor * allocated_tensors[1024];
#endif
};

#ifdef GGML_ALLOCATOR_DEBUG
static void add_allocated_tensor(struct ggml_allocr * alloc, struct ggml_tensor * tensor) {
    for (int i = 0; i < 1024; i++) {
        if (alloc->allocated_tensors[i] == NULL) {
            alloc->allocated_tensors[i] = tensor;
            return;
        }
    }
    GGML_ASSERT(!"out of allocated_tensors");
}
static void remove_allocated_tensor(struct ggml_allocr * alloc, struct ggml_tensor * tensor) {
    for (int i = 0; i < 1024; i++) {
        if (alloc->allocated_tensors[i] == tensor ||
            (alloc->allocated_tensors[i] != NULL && alloc->allocated_tensors[i]->data == tensor->data)) {
            alloc->allocated_tensors[i] = NULL;
            return;
        }
    }
    printf("tried to free tensor %s not found\n", tensor->name);
    GGML_ASSERT(!"tensor not found");
}
#endif

static size_t ggml_allocr_get_alloc_size(struct ggml_allocr * alloc, struct ggml_tensor * tensor) {
    return ggml_nbytes(tensor);

    UNUSED(alloc);
}

// check if a tensor is allocated by this buffer
static bool ggml_allocr_is_own(struct ggml_allocr * alloc, const struct ggml_tensor * tensor) {
    void * ptr = tensor->data;
    return ptr >= alloc->data && (char *)ptr < (char *)alloc->data + alloc->max_size;
}

static bool ggml_is_view(struct ggml_tensor * t) {
    return t->view_src != NULL;
}

void ggml_allocr_alloc(struct ggml_allocr * alloc, struct ggml_tensor * tensor) {
#ifdef GGML_ALLOCATOR_DEBUG
    GGML_ASSERT(!ggml_is_view(tensor)); // views generally get data pointer from one of their sources
    GGML_ASSERT(tensor->data == NULL); // avoid allocating tensor which already has memory allocated
#endif
    size_t size = ggml_allocr_get_alloc_size(alloc, tensor);
    size = aligned_offset(NULL, size, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    size_t max_avail = 0;

    // find the best fitting free block besides the last block
    int best_fit_block = -1;
    size_t best_fit_size = SIZE_MAX;
    for (int i = 0; i < alloc->n_free_blocks - 1; i++) {
        struct free_block * block = &alloc->free_blocks[i];
        max_avail = MAX(max_avail, block->size);
        if (block->size >= size && block->size <= best_fit_size) {
            best_fit_block = i;
            best_fit_size = block->size;
        }
    }

    AT_PRINTF("block %d\n", best_fit_block);

    if (best_fit_block == -1) {
        // the last block is our last resort
        struct free_block * block = &alloc->free_blocks[alloc->n_free_blocks - 1];
        max_avail = MAX(max_avail, block->size);
        if (block->size >= size) {
            best_fit_block = alloc->n_free_blocks - 1;
        } else {
            fprintf(stderr, "%s: not enough space in the buffer (needed %zu, largest block available %zu)\n",
                    __func__, size, max_avail);
            GGML_ASSERT(!"not enough space in the buffer");
            return;
        }
    }
    struct free_block * block = &alloc->free_blocks[best_fit_block];
    void * addr = block->addr;
    block->addr = (char*)block->addr + size;
    block->size -= size;
    if (block->size == 0) {
        // remove block if empty
        alloc->n_free_blocks--;
        for (int j = best_fit_block; j < alloc->n_free_blocks; j++) {
            alloc->free_blocks[j] = alloc->free_blocks[j+1];
        }
    }

    tensor->data = addr;
    AT_PRINTF("%s: allocated data at %p\n", __func__, tensor->data);

#ifdef GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, tensor);
    size_t cur_max = (char*)addr - (char*)alloc->data + size;
    if (cur_max > alloc->max_size) {
        printf("max_size = %.2f MB: tensors: ", cur_max / 1024.0 / 1024.0);
        for (int i = 0; i < 1024; i++) {
            if (alloc->allocated_tensors[i]) {
                printf("%s (%.2f MB) ", alloc->allocated_tensors[i]->name, ggml_nbytes(alloc->allocated_tensors[i]) / 1024.0 / 1024.0);
            }
        }
        printf("\n");
    }
#endif

    alloc->max_size = MAX(alloc->max_size, (char*)addr - (char*)alloc->data + size);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
static void ggml_allocr_free_tensor(struct ggml_allocr * alloc, struct ggml_tensor * tensor) {
    void * ptr = tensor->data;

    if (ggml_allocr_is_own(alloc, tensor) == false) {
        // the tensor was not allocated in this buffer
        // this can happen because the graph allocator will try to free weights and other tensors from different buffers
        // the easiest way to deal with this is just to ignore it
        return;
    }

    size_t size = ggml_allocr_get_alloc_size(alloc, tensor);
    size = aligned_offset(NULL, size, alloc->alignment);
    AT_PRINTF("%s: freeing %s at %p (%zu bytes) - n_free_blocks = %d\n", __func__, tensor->name, ptr, size, alloc->n_free_blocks);
    AT_PRINTF("%s: alloc->data = %p alloc->data+alloc->size = %p alloc->data+alloc->max_size = %p\n", __func__, alloc->data, (char*)alloc->data + alloc->size, (char*)alloc->data + alloc->max_size);

#ifdef GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, tensor);
#endif

    // see if we can merge with an existing block
    for (int i = 0; i < alloc->n_free_blocks; i++) {
        struct free_block * block = &alloc->free_blocks[i];
        // check if ptr is at the end of the block
        if ((char*)block->addr + block->size == ptr) {
            block->size += size;
            // check if we can merge with the next block
            if (i < alloc->n_free_blocks - 1 && (char*)block->addr + block->size == alloc->free_blocks[i+1].addr) {
                block->size += alloc->free_blocks[i+1].size;
                alloc->n_free_blocks--;
                for (int j = i+1; j < alloc->n_free_blocks; j++) {
                    alloc->free_blocks[j] = alloc->free_blocks[j+1];
                }
            }
            return;
        }
        // check if ptr is at the beginning of the block
        if ((char*)ptr + size == block->addr) {
            block->addr = ptr;
            block->size += size;
            // check if we can merge with the previous block
            if (i > 0 && (char*)alloc->free_blocks[i-1].addr + alloc->free_blocks[i-1].size == block->addr) {
                alloc->free_blocks[i-1].size += block->size;
                alloc->n_free_blocks--;
                for (int j = i; j < alloc->n_free_blocks; j++) {
                    alloc->free_blocks[j] = alloc->free_blocks[j+1];
                }
            }
            return;
        }
    }
    // otherwise, add a new block
    GGML_ASSERT(alloc->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    int insert_pos = 0;
    while (insert_pos < alloc->n_free_blocks && alloc->free_blocks[insert_pos].addr < ptr) {
        insert_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    for (int i = alloc->n_free_blocks; i > insert_pos; i--) {
        alloc->free_blocks[i] = alloc->free_blocks[i-1];
    }
    // insert the new block
    alloc->free_blocks[insert_pos].addr = ptr;
    alloc->free_blocks[insert_pos].size = size;
    alloc->n_free_blocks++;
}

void ggml_allocr_set_parse_seq(struct ggml_allocr * alloc, const int * list, int n) {
    for (int i = 0; i < n; i++) {
        alloc->parse_seq[i] = list[i];
    }
    alloc->parse_seq_len = n;
}

void ggml_allocr_reset(struct ggml_allocr * alloc) {
    alloc->n_free_blocks = 1;
    size_t align_offset = aligned_offset(alloc->data, 0, alloc->alignment);
    alloc->free_blocks[0].addr = (char *)alloc->data + align_offset;
    alloc->free_blocks[0].size = alloc->size - align_offset;
}

struct ggml_allocr * ggml_allocr_new(void * data, size_t size, size_t alignment) {
    struct ggml_allocr * alloc = (struct ggml_allocr *)malloc(sizeof(struct ggml_allocr) /* + n_free_blocks * sizeof(struct free_block) */);

    *alloc = (struct ggml_allocr){
        /*.data          = */ data,
        /*.size          = */ size,
        /*.alignment     = */ alignment,
        /*.n_free_blocks = */ 0,
        /*.free_blocks   = */ {{0}},
        /*.hash_table    = */ {{0}},
        /*.max_size      = */ 0,
        /*.measure       = */ false,
        /*.parse_seq     = */ {0},
        /*.parse_seq_len = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {0},
#endif
    };

    ggml_allocr_reset(alloc);

    return alloc;
}

// OS specific functions to allocate and free uncommitted virtual memory
static void * alloc_vmem(size_t size) {
#if defined(_WIN32)
    return VirtualAlloc(NULL, size, MEM_RESERVE, PAGE_NOACCESS);
#elif defined(_POSIX_MAPPED_FILES)
    void * ptr = mmap(NULL, size, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (ptr == MAP_FAILED) {
        return NULL;
    }
    return ptr;
#else
    // use a fixed address for other platforms
    uintptr_t base_addr = (uintptr_t)-size - 0x100;
    return (void *)base_addr;
#endif
}

static void free_vmem(void * base_addr, size_t size) {
#if defined(_WIN32)
    VirtualFree(base_addr, 0, MEM_RELEASE);
    UNUSED(size);
#elif defined(_POSIX_MAPPED_FILES)
    munmap(base_addr, size);
#else
    // nothing to do
    UNUSED(base_addr);
    UNUSED(size);
#endif
}

// allocate uncommitted virtual memory to measure the size of the graph
static void alloc_measure_vmem(void ** base_addr, size_t * size) {
    // 128GB for 64-bit, 1GB for 32-bit
    *size = sizeof(void *) == 4 ? 1ULL<<30 : 1ULL<<37;
    do {
        *base_addr = alloc_vmem(*size);
        if (*base_addr != NULL) {
            AT_PRINTF("allocated %.2f GB of virtual memory for measure buffer at %p\n", *size / 1024.0 / 1024.0 / 1024.0, *base_addr);
            return;
        }
        // try again with half the size
        *size /= 2;
    } while (*size > 0);

    GGML_ASSERT(!"failed to allocate virtual memory for measure buffer");
}

static void free_measure_vmem(void * base_addr, size_t size) {
    free_vmem(base_addr, size);
}

struct ggml_allocr * ggml_allocr_new_measure(size_t alignment) {
    struct ggml_allocr * alloc = (struct ggml_allocr *)malloc(sizeof(struct ggml_allocr) /* + n_free_blocks * sizeof(struct free_block) */);

    void * base_addr;
    size_t size;

    alloc_measure_vmem(&base_addr, &size);

    *alloc = (struct ggml_allocr){
        /*.data          = */ base_addr,
        /*.size          = */ size,
        /*.alignment     = */ alignment,
        /*.n_free_blocks = */ 0,
        /*.free_blocks   = */ {{0}},
        /*.hash_table    = */ {{0}},
        /*.max_size      = */ 0,
        /*.measure       = */ true,
        /*.parse_seq     = */ {0},
        /*.parse_seq_len = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {0},
#endif
    };

    ggml_allocr_reset(alloc);

    return alloc;
}

void ggml_allocr_free(struct ggml_allocr * alloc) {
    if (alloc->measure) {
        free_measure_vmem(alloc->data, alloc->size);
    }
    free(alloc);
}

bool ggml_allocr_is_measure(struct ggml_allocr * alloc) {
    return alloc->measure;
}

//////////// compute graph allocator

static bool ggml_are_same_layout(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    if (a->type != b->type) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
        if (a->nb[i] != b->nb[i]) {
            return false;
        }
    }
    return true;
}

static bool ggml_op_can_inplace(enum ggml_op op) {
    switch (op) {
        case GGML_OP_SCALE:
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_UNARY:
        case GGML_OP_ROPE:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_CONT:
            return true;

        default:
            return false;
    }
}

static void allocate_node(struct ggml_allocr * alloc, struct ggml_tensor * node) {
    struct hash_node * ht = alloc->hash_table;
    if (node->data == NULL) {
        if (ggml_is_view(node)) {
            assert(node->view_src->data != NULL);
            node->data = (char *)node->view_src->data + node->view_offs;
        } else {
            // see if we can reuse a parent's buffer (inplace)
            if (ggml_op_can_inplace(node->op)) {
                for (int i = 0; i < GGML_MAX_SRC; i++) {
                    struct ggml_tensor * parent = node->src[i];
                    if (parent == NULL) {
                        break;
                    }

                    // if the node's data is external, then we cannot re-use it
                    if (ggml_allocr_is_own(alloc, parent) == false) {
                        AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, node->name, parent->data);
                        continue;
                    }

                    struct hash_node * p_hn = hash_get(ht, parent);
                    if (parent->data != NULL && p_hn->n_children == 1 && p_hn->n_views == 0 && ggml_are_same_layout(node, parent)) {
                        if (ggml_is_view(parent)) {
                            struct ggml_tensor * view_src = parent->view_src;
                            struct hash_node * view_src_hn = hash_get(ht, view_src);
                            if (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                                // TODO: the offset of the view parent must be kept to ensure that the op doesn't overwrite
                                // the parent's data that it will need later (same layout requirement). the problem is that then
                                // we cannot free the tensor because the original address of the allocation is lost.
                                // adding a view_src pointer to the tensor would solve this and simplify the code dealing with views
                                // for now, we only reuse the parent's data if the offset is zero (view_src->data == parent->data)
                                AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, node->name);
                                node->data = parent->data;
                                return;
                            }
                        }
                        else {
                            AT_PRINTF("reusing parent %s for %s\n", parent->name, node->name);
                            node->data = parent->data;
                            return;
                        }
                    }
                }
            }
            ggml_allocr_alloc(alloc, node);
        }
    }
}

static size_t ggml_allocr_alloc_graph_tensors_n(
    struct ggml_allocr * alloc,
    struct ggml_cgraph ** graphs, int n_graphs,
    struct ggml_tensor *** inputs, struct ggml_tensor *** outputs) {

    // reset hash table
    struct hash_node * ht = alloc->hash_table;
    memset(ht, 0, sizeof(struct hash_node) * GGML_GRAPH_HASHTABLE_SIZE);

    // count number of children and views
    for (int g = 0; g < n_graphs; g++) {
        struct ggml_cgraph * gf = graphs[g];
        for (int i = 0; i < gf->n_nodes; i++) {
            struct ggml_tensor * node = gf->nodes[i];

            if (ggml_is_view(node)) {
                struct ggml_tensor * view_src = node->view_src;
                hash_get(ht, view_src)->n_views += 1;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * parent = node->src[j];
                if (parent == NULL) {
                    break;
                }
                hash_get(ht, parent)->n_children += 1;
            }
        }
    }

    // allocate tensors
    for (int g = 0; g < n_graphs; g++) {
        struct ggml_cgraph * gf = graphs[g];
        AT_PRINTF("####### graph %d/%d\n", g, n_graphs);
        // graph inputs are allocated first to ensure that they are not overwritten by each other
        if (inputs != NULL && inputs[g] != NULL) {
            for (int i = 0; inputs[g][i] != NULL; i++) {
                struct ggml_tensor * input = inputs[g][i];
                AT_PRINTF("input: %s\n", input->name);
                allocate_node(alloc, input);
            }
        }
        // if we have parse_seq then we allocate nodes following the list, and we only free nodes at barriers
        int last_barrier_pos = 0;
        int n_nodes = alloc->parse_seq_len ? alloc->parse_seq_len : gf->n_nodes;

        for (int ind = 0; ind < n_nodes; ind++) {
            // allocate a node if there is no parse_seq or this is not a barrier
            if ((alloc->parse_seq_len==0) || alloc->parse_seq[ind] != -1) {
                int i = alloc->parse_seq_len ? alloc->parse_seq[ind] : ind;
                struct ggml_tensor * node = gf->nodes[i];

                // allocate parents (leafs)
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * parent = node->src[j];
                    if (parent == NULL) {
                        break;
                    }
                    allocate_node(alloc, parent);
                }

                // allocate node
                allocate_node(alloc, node);

                AT_PRINTF("exec: %s (%s) <= ", ggml_op_name(node->op), node->name);
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * parent = node->src[j];
                    if (parent == NULL) {
                        break;
                    }
                    AT_PRINTF("%s", parent->name);
                    if (j < GGML_MAX_SRC - 1 && node->src[j + 1] != NULL) {
                        AT_PRINTF(", ");
                    }
                }
                AT_PRINTF("\n");
            }

            // update parents
            // update immediately if there is no parse_seq
            // update only at barriers if there is parse_seq
            if ((alloc->parse_seq_len == 0) || alloc->parse_seq[ind] == -1) {
                int update_start = alloc->parse_seq_len ? last_barrier_pos : ind;
                int update_end   = alloc->parse_seq_len ? ind              : ind + 1;
                for (int i = update_start; i < update_end; i++) {
                    int node_i = alloc->parse_seq_len ? alloc->parse_seq[i] : i;
                    struct ggml_tensor * node = gf->nodes[node_i];

                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * parent = node->src[j];
                        if (parent == NULL) {
                            break;
                        }
                        struct hash_node * p_hn = hash_get(ht, parent);
                        p_hn->n_children -= 1;

                        //AT_PRINTF("parent %s: %d children, %d views\n", parent->name, parent->n_children, parent->n_views);

                        if (p_hn->n_children == 0 && p_hn->n_views == 0) {
                            if (ggml_is_view(parent)) {
                                struct ggml_tensor * view_src = parent->view_src;
                                struct hash_node * view_src_hn = hash_get(ht, view_src);
                                view_src_hn->n_views -= 1;
                                AT_PRINTF("view_src %s: %d children, %d views\n", view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                                if (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src->data != node->data) {
                                    ggml_allocr_free_tensor(alloc, view_src);
                                }
                            }
                            else {
                                if (parent->data != node->data) {
                                    ggml_allocr_free_tensor(alloc, parent);
                                }
                            }
                        }
                    }
                }
                AT_PRINTF("\n");
                if (alloc->parse_seq_len) {
                    last_barrier_pos = ind + 1;
                }
            }
        }
        // free graph outputs here that wouldn't be freed otherwise because they have no children
        if (outputs != NULL && outputs[g] != NULL) {
            for (int i = 0; outputs[g][i] != NULL; i++) {
                struct ggml_tensor * output = outputs[g][i];
                AT_PRINTF("output: %s\n", output->name);
                ggml_allocr_free_tensor(alloc, output);
            }
        }
    }

    return alloc->max_size;
}

size_t ggml_allocr_alloc_graph(struct ggml_allocr * alloc, struct ggml_cgraph * graph) {
    return ggml_allocr_alloc_graph_tensors_n(alloc, &graph, 1, NULL, NULL);
}

size_t ggml_allocr_max_size(struct ggml_allocr * alloc) {
    return alloc->max_size;
}
