#include "ggml-profile.h"

#include <stdint.h>
#include <stdlib.h>

#include <chrono>

#ifdef GGML_GRAPH_PROFILER

extern "C" void ggml_profile_graph_init(struct ggml_cgraph *cg, int n_threads)
{
    if (!getenv("GGML_GRAPH_PROFILE")) { return; }

    // The number of threads may change between passes (pp vs tg).
    // Allocate for max_n_threads for simplicity for now.
    // TODO: use aligned allocator

    size_t node_size = sizeof(struct ggml_profile_data) * GGML_MAX_N_THREADS;
    size_t pvec_size = sizeof(std::intptr_t) * cg->n_nodes;
    size_t data_size = node_size * cg->n_nodes;
    size_t t_size    = pvec_size + data_size;

    cg->prof = (struct ggml_profile_data **) malloc(t_size);
    if (!cg->prof) {
        fprintf(stderr, "ggml-profile: failed to allocate profiling data : n_threads %d n_nodes %d\n", n_threads, cg->n_nodes);
        return;
    }

    memset(cg->prof, 0, t_size);

    // init pre-thread pointers
    uint8_t * data = (uint8_t *) cg->prof + pvec_size;
    for (int i=0; i < cg->n_nodes; i++) {
        cg->prof[i] = (struct ggml_profile_data *) data; data += node_size;
    }
}

extern "C" void ggml_profile_graph_start(struct ggml_cgraph *cg, int n_threads)
{
    if (!cg->prof) { ggml_profile_graph_init(cg, n_threads); }
    if (!cg->prof) { return; }
}

static inline int ggml_profile_format_tensor_dims(char *str, struct ggml_tensor *t)
{
    return sprintf(str, "%d:%d:%d:%d",
        (int) t->ne[0], (int) t->ne[1], (int) t->ne[3], (int) t->ne[3]);
}

static inline void ggml_profile_format_op_dims(char *str, struct ggml_tensor *t)
{
    char *p = str;

    // append src0 and src1 (if any)
    if (t->src[0]) {
       p += ggml_profile_format_tensor_dims(p, t->src[0]);

       for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
           p += sprintf(p, " x ");
           p += ggml_profile_format_tensor_dims(p, t->src[i]);
       }

       p += sprintf(p, " -> ");
    }

    // format self dims separately for better visual alignment
    char self[64];
    ggml_profile_format_tensor_dims(self, t);

    p += sprintf(p, "%12s", self);
}

static inline void ggml_profile_format_op_types(char *str, struct ggml_tensor *t)
{
    char *p = str;

    // append src0 and src1 (if any)
    if (t->src[0]) {
       p += sprintf(p, "%s", ggml_type_name(t->src[0]->type));

       for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
           p += sprintf(p, " x ");
           p += sprintf(p, "%s", ggml_type_name(t->src[i]->type));
       }

       p += sprintf(p, " -> ");
    }

    p += sprintf(p, "%3s", ggml_type_name(t->type));
}


extern "C" void ggml_profile_graph_finish(struct ggml_cgraph *cg, int n_threads)
{
    if (!cg->prof) { return; }

    fprintf(stderr, "ggml-profile: | node idx | op name | proc (nsec) | sync (nsec) | total (nsec) | op dims | op types | tensor name |\n");
    fprintf(stderr, "ggml-profile: | -------: | :------ | ----------: | ----------: | -----------: | ------: | -------: | ----------: |\n");

    char dims[64 * GGML_MAX_SRC];
    char types[16 * GGML_MAX_SRC];

    for (int i = 0; i < cg->n_nodes; i++) {
        uint64_t p_nsec = 0;
        uint64_t s_nsec = 0;
        uint64_t t_nsec = 0;

        // add up per thread counters and reset them
        for (int t=0; t < n_threads; t++) {
            p_nsec += cg->prof[i][t].nsec[GGML_PROF_OP_SYNC] - cg->prof[i][t].nsec[GGML_PROF_OP_START];
            s_nsec += cg->prof[i][t].nsec[GGML_PROF_OP_END]  - cg->prof[i][t].nsec[GGML_PROF_OP_SYNC];
            t_nsec += cg->prof[i][t].nsec[GGML_PROF_OP_END]  - cg->prof[i][t].nsec[GGML_PROF_OP_START];

            cg->prof[i][t].nsec[GGML_PROF_OP_START] = 0;
            cg->prof[i][t].nsec[GGML_PROF_OP_SYNC]  = 0;
            cg->prof[i][t].nsec[GGML_PROF_OP_END]   = 0;
        }

        ggml_profile_format_op_dims(dims, cg->nodes[i]);
        ggml_profile_format_op_types(types, cg->nodes[i]);

        fprintf(stderr, "ggml-profile: | %04d | %10s | %10lu | %10lu | %10lu | %46s | %22s | %20s |\n",
            i, ggml_op_name(cg->nodes[i]->op),
            (unsigned long) p_nsec, (unsigned long) s_nsec, (unsigned long) t_nsec,
            dims, types, cg->nodes[i]->name);
    }
    fprintf(stderr, "ggml-profile:   \n"); // empty line to split tables
}

extern "C" void ggml_profile_graph_free(struct ggml_cgraph *cg)
{
    if (!cg->prof) { return; }

    free(cg->prof); cg->prof = nullptr;
}

extern "C" void ggml_profile_op_event(const struct ggml_cgraph *cg, enum ggml_profile_event e, int node_n, int ith)
{
    if (!cg->prof) { return; }

    using clock = std::chrono::high_resolution_clock;
    cg->prof[node_n][ith].nsec[e] = std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

#endif // GGML_GRAPH_PROFILER
