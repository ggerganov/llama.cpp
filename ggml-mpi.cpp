#include "ggml-mpi.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define UNUSED GGML_UNUSED

static bool have_init = false;

static void* send_buffer;

struct ggml_mpi_context {
    int rank;
    int size;
    MPI_Comm comm;
    int layer_start;
    int layer_end;
    MPI_Status status;

    struct ggml_tensor *inp0;
    std::string name;
    struct ggml_backend * wrapped_backend;
    std::vector<ggml_backend_t> backends;
    ggml_backend_sched_t scheduler;
    bool remote;
    void* send_buffer;
    int trans_id;
    int recv_trans_id;
};

void ggml_mpi_backend_init(void) {
    int ret;

    GGML_ASSERT(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &ret) == MPI_SUCCESS);
    have_init = true;
    const int buffer_size = 128*1024*1024*8;
    send_buffer = calloc(1, buffer_size); // 128MB buffer
    fprintf(stderr, "BUFFER ATTACH RETCODE=%d\n", MPI_Buffer_attach(send_buffer, buffer_size));
}

void ggml_mpi_sync_pipelined(
        struct ggml_mpi_context *   ctx_mpi,
        void * val,
        int count,
        MPI_Datatype datatype,
        int tag
);

void ggml_mpi_backend_free(void) {
    MPI_Finalize();
}

struct ggml_mpi_context * ggml_mpi_init(void) {

    if (!have_init) {
        ggml_mpi_backend_init();
    }

    auto * ctx = new ggml_mpi_context;

    MPI_Comm_rank(MPI_COMM_WORLD, &ctx->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx->size);
    ctx->comm = MPI_COMM_WORLD;
    ctx->remote = false;

    ctx->send_buffer = send_buffer;

    return ctx;
}

struct ggml_mpi_context * ggml_mpi_split_comm(struct ggml_mpi_context * ctx, int color, int key) {
    auto * newCtx = static_cast<ggml_mpi_context *>(calloc(1, sizeof(struct ggml_mpi_context)));
    MPI_Comm_split(ctx->comm, color, key, &newCtx->comm);
    MPI_Comm_rank(newCtx->comm, &newCtx->rank);
    MPI_Comm_size(newCtx->comm, &newCtx->size);
    return newCtx;
}

void ggml_mpi_free(struct ggml_mpi_context * ctx) {
    MPI_Comm_free(&(ctx->comm));
    free(ctx);
}

int ggml_mpi_rank(struct ggml_mpi_context * ctx) {
    return ctx->rank;
}

size_t ggml_mpi_size(struct ggml_mpi_context * ctx) {
    return ctx->size;
}

int ggml_mpi_next_node(struct ggml_mpi_context * ctx_mpi) {
    return (ctx_mpi->rank + 1) % ctx_mpi->size;
}

int ggml_mpi_prev_node(struct ggml_mpi_context * ctx_mpi) {
    int temp = (ctx_mpi->rank - 1);
    return (temp >= 0) ? temp : ctx_mpi->size - 1;
}

void ggml_mpi_sync_pipelined(
        struct ggml_mpi_context *   ctx_mpi,
        void * val,
        int count,
        MPI_Datatype datatype,
        int tag
) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }

//    printf("Rank %d sync pipelined with tag %d\n", ctx_mpi->rank, tag);


    if (ctx_mpi->rank != 0) {
        MPI_Recv(val, count, datatype, ggml_mpi_prev_node(ctx_mpi), tag, ctx_mpi->comm, MPI_STATUS_IGNORE);
    }
    if(ctx_mpi->rank < ctx_mpi->size - 1) {
        GGML_ASSERT(ctx_mpi->send_buffer != nullptr);
        GGML_ASSERT(val != nullptr || count == 0);
        GGML_ASSERT(count < 128*1024*1024);

        const int retval = MPI_Bsend(val, count, datatype, ggml_mpi_next_node(ctx_mpi), tag, ctx_mpi->comm);
        GGML_ASSERT(retval == MPI_SUCCESS);

    }
}

void ggml_mpi_barrier(struct ggml_mpi_context * ctx_mpi) {
    MPI_Barrier(ctx_mpi->comm);
}

void ggml_mpi_probe(struct ggml_mpi_context * ctx_mpi, int src, int tag) {
    MPI_Probe((src >= 0) ? src : MPI_ANY_SOURCE, (tag >= 0) ? tag : MPI_ANY_TAG, ctx_mpi->comm, &(ctx_mpi->status));
}

int ggml_mpi_iprobe(struct ggml_mpi_context * ctx_mpi, int src, int tag) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return 0;
    }

    int ret;
    MPI_Iprobe((src >= 0) ? src : MPI_ANY_SOURCE, (tag >= 0) ? tag : MPI_ANY_TAG, ctx_mpi->comm, &ret, &(ctx_mpi->status));
    return ret;
}

int ggml_mpi_status_tag(struct ggml_mpi_context * ctx_mpi) {
    return ctx_mpi->status.MPI_TAG;
}

int ggml_mpi_status_count_int32(struct ggml_mpi_context * ctx_mpi) {
    int32_t count;
    MPI_Get_count(&ctx_mpi->status, MPI_INT32_T, &count);
    return count;
}

void ggml_mpi_eval_init(
        struct ggml_mpi_context *   ctx_mpi,
                int32_t         *   n_tokens,
                int32_t         **  pos,
                int32_t         **  n_seq_ids,
                int32_t         *** seq_id,
                int8_t          **  logits,
                uint32_t            n_seq_max) {


    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }

    int32_t old_n_tokens = *n_tokens;
    ggml_mpi_sync_pipelined(ctx_mpi, n_tokens, 1, MPI_INT, GGML_MPI_N_TOKENS);

    if (old_n_tokens != *n_tokens) {
        *pos = static_cast<int32_t *>(realloc(*pos, *n_tokens * sizeof(int32_t)));
        *n_seq_ids = static_cast<int32_t *>(realloc(*n_seq_ids, *n_tokens * sizeof(int32_t)));
        *logits = static_cast<int8_t *>(realloc(*logits, *n_tokens * sizeof(int32_t)));
    }

    int8_t* temp_logits = (int8_t*) calloc(*n_tokens, sizeof(int8_t));

    if (ctx_mpi->rank == 0 && *logits != nullptr) {
        ggml_mpi_sync_pipelined(ctx_mpi, *logits, *n_tokens, MPI_INT8_T, GGML_MPI_BATCH_LOGITS);
    } else {
        ggml_mpi_sync_pipelined(ctx_mpi, temp_logits, *n_tokens, MPI_INT8_T, GGML_MPI_BATCH_LOGITS);
    }



    if (ctx_mpi->rank != 0) {
        bool should_set_batch_logits = false;
        for (int i = 0; i < *n_tokens; i++) {
            if (temp_logits[i]) {
                should_set_batch_logits = true;
                break;
            }
        }
        if (should_set_batch_logits) {
            if (*logits != NULL) {
                free(*logits);
                *logits = NULL;
            }
            *logits = temp_logits;
        } else {
            if (*logits != NULL) {
                free(*logits);
                *logits = NULL;
            }
            free(temp_logits);
        }
    } else {
        free(temp_logits);
    }

    // For now, we assume that the pos, seq_ids, tokens, etc have been
    // pre-allocated for the largest possible sizes, even on worker nodes.

    GGML_ASSERT(n_seq_ids != nullptr);
    GGML_ASSERT(*n_seq_ids != nullptr);

    GGML_ASSERT(n_tokens != nullptr);


    ggml_mpi_sync_pipelined(ctx_mpi, *n_seq_ids, *n_tokens, MPI_INT32_T, GGML_MPI_N_SEQ_IDS);

    // We need to know the total number of sequence
    // ids, so we count them all up
    int32_t total_n_seq_ids = 0;
    for (int32_t i = 0; i < *n_tokens; i++) {
        total_n_seq_ids += (*n_seq_ids)[i];
    }

    // MPI can't chase the pointers for multidimensional arrays, so we flatten them first
    // for transit
    int32_t * flattened_seq_ids = static_cast<int32_t *>(calloc(total_n_seq_ids, sizeof(int32_t)));

    int32_t current_index = 0;

    // Only rank 0 needs to flatten since the others don't have the real seq_id
    if (ctx_mpi->rank == 0) {
        for (int32_t i = 0; i < *n_tokens; i++) {
            for (int32_t j = 0; j < (*n_seq_ids)[i]; j++) {
                flattened_seq_ids[current_index] = (*seq_id)[i][j];
                current_index++;
            }
        }
    }



    ggml_mpi_sync_pipelined(ctx_mpi, *pos, *n_tokens, MPI_INT32_T, GGML_MPI_POS);
    ggml_mpi_sync_pipelined(ctx_mpi, flattened_seq_ids, total_n_seq_ids, MPI_INT32_T, GGML_MPI_SEQ_IDS);

    current_index = 0;
    for (int32_t i = 0; i < *n_tokens; i++) {
        for (int32_t j = 0; j < (*n_seq_ids)[i]; j++) {
            (*seq_id)[i][j] = flattened_seq_ids[current_index];
            current_index++;
        }

    }
    free(flattened_seq_ids);
}


void ggml_mpi_sync_int(
        struct ggml_mpi_context * ctx_mpi,
                        int32_t * val
) {
    MPI_Bcast(val, 1, MPI_INT32_T, 0, ctx_mpi->comm);
}

void ggml_mpi_sync_ints_pipelined(
        struct ggml_mpi_context * ctx_mpi,
        int32_t * vals,
        int count,
        int tag
) {
    ggml_mpi_sync_pipelined(ctx_mpi, vals, count, MPI_INT32_T, tag);
    int old_trans = ctx_mpi->trans_id;
    ggml_mpi_sync_pipelined(ctx_mpi, &ctx_mpi->trans_id, 1, MPI_INT32_T, GGML_MPI_TRANS_ID);
    ctx_mpi->recv_trans_id = ctx_mpi->trans_id;
    ctx_mpi->trans_id = old_trans;
}

static void ggml_mpi_tensor_send(const struct ggml_tensor * t, const void* data, int mpi_rank_dst, MPI_Comm comm) {
    MPI_Datatype mpi_type;

//    fprintf(stderr, "Type: %d\n", t->type);

    switch (t->type) {
        case GGML_TYPE_I32: mpi_type = MPI_INT32_T; break;
        case GGML_TYPE_F32: mpi_type = MPI_FLOAT;   break;
        case GGML_TYPE_F16: mpi_type = MPI_INT16_T;   break;
        default: GGML_ASSERT(false && "not implemented");
    }
    int rank;
    MPI_Comm_rank(comm, &rank);
//    fprintf(stderr, "Sending tensor %s (buffer %s) from %d to %d\n", t->name, ggml_backend_buffer_name(t->buffer), rank, mpi_rank_dst);

    GGML_ASSERT(rank != mpi_rank_dst);

    const int retval = MPI_Bsend(data, ggml_nelements(t), mpi_type, mpi_rank_dst, 0, comm);
    GGML_ASSERT(retval == MPI_SUCCESS);

}
static void ggml_mpi_tensor_send(const struct ggml_tensor * t, int mpi_rank_dst, MPI_Comm comm) {
    ggml_mpi_tensor_send(t, t->data, mpi_rank_dst, comm);
}

static void ggml_mpi_tensor_recv(const struct ggml_tensor * t, void * data, int mpi_rank_src, MPI_Comm comm) {
    MPI_Datatype mpi_type;

    switch (t->type) {
        case GGML_TYPE_I32: mpi_type = MPI_INT32_T; break;
        case GGML_TYPE_F32: mpi_type = MPI_FLOAT;   break;
        default: GGML_ASSERT(false && "not implemented");
    }

    MPI_Status status; UNUSED(status);
//    fprintf(stderr, "%s: tensor receive == null: %d\n", __func__, t->data == NULL);
    int rank;
    MPI_Comm_rank(comm, &rank);
//    fprintf(stderr, "Receiving tensor %s (buffer %s) from %d at %d\n", t->name, ggml_backend_buffer_name(t->buffer), mpi_rank_src, rank);

    GGML_ASSERT(rank != mpi_rank_src);

    const int retval = MPI_Recv(data, ggml_nelements(t), mpi_type, mpi_rank_src, MPI_ANY_TAG, comm, &status);
    GGML_ASSERT(retval == MPI_SUCCESS);
}

static void ggml_mpi_tensor_recv(struct ggml_tensor * t, int mpi_rank_src, MPI_Comm comm) {
    ggml_mpi_tensor_recv(t, t->data, mpi_rank_src, comm);
}

uint16_t** ggml_mpi_split_range(
    struct ggml_mpi_context * ctx_mpi,
    uint16_t start,
    uint16_t end,
    const float node_weights[]
) {
    // Splits the range given by start and end
    // over the available nodes. This implementation
    // assumes that node 0 handles the final part of the range
    // while node 1 handles the beginning, to form a ring pipeline

    uint16_t range_length = end - start + 1;
    uint16_t ** ranges = (uint16_t**) malloc(sizeof(uint16_t*) * ctx_mpi->size);
    for (int i = 0; i < ctx_mpi->size; i++) {
        ranges[i] = (uint16_t*) malloc(sizeof(uint16_t) * 2);
    }
    uint16_t next_layer = 0;
    for (int i=0; i < ctx_mpi->size; i++) {
        ranges[i][0] = next_layer;
        ranges[i][1] = MIN(end, ranges[i][0] + (node_weights[i] * range_length) + start);
        next_layer = ranges[i][1]+1;
    }

//    ranges[0][0] = next_layer;
//    ranges[0][1] = MIN(end, next_layer + (node_weights[0] * range_length) + start);
    return ranges;

}

// BACKEND V2

struct ggml_backend_mpi_buffer_context {
    ggml_backend_buffer_t wrapped_buffer;
    ggml_mpi_context * ctx_mpi;
};

struct ggml_backend_mpi_buffer_type_context {
    std::string name;
    ggml_backend_buffer_type_t wrapped_buffer_type;
    ggml_mpi_context * ctx_mpi;
};

int ggml_backend_mpi_buffer_type_rank(ggml_backend_buffer_type_t buft);

int ggml_backend_mpi_buffer_type_local_rank(ggml_backend_buffer_type_t buft);

GGML_CALL static const char * ggml_backend_mpi_buffer_type_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml_backend_mpi_buffer_type_context *) buft->context;


    return strdup(
            (
                    ctx->name +
                    " Buffer Type(Rank " +
                    std::to_string(
                            ggml_backend_mpi_buffer_type_rank(buft)
                    ) +
                    ", local rank " +
                    std::to_string(ggml_backend_mpi_buffer_type_local_rank(buft)) +
                    "):" +
                    std::string(
                            ctx->wrapped_buffer_type->iface.get_name(ctx->wrapped_buffer_type)
                    )
            ).c_str()
    );
}

MPI_Comm ggml_backend_mpi_buffer_type_get_comm(ggml_backend_buffer_type_t buft) {
    auto * buft_ctx = (ggml_backend_mpi_buffer_type_context *) buft->context;
    return buft_ctx->ctx_mpi->comm;

}

MPI_Comm ggml_backend_mpi_buffer_get_comm(ggml_backend_buffer_t buffer) {
    return ggml_backend_mpi_buffer_type_get_comm(buffer->buft);
}

MPI_Comm ggml_backend_mpi_get_comm(ggml_backend_t backend) {
    auto * ctx = (ggml_mpi_context *) backend->context;

    return ctx->comm;
}

int ggml_backend_mpi_buffer_local_rank(ggml_backend_buffer_t buffer) {
    int rank;
    int ret = MPI_Comm_rank(ggml_backend_mpi_buffer_get_comm(buffer), &rank);
    GGML_ASSERT(ret == MPI_SUCCESS);
    return rank;
}

int ggml_backend_mpi_buffer_type_local_rank(ggml_backend_buffer_type_t buft) {
    int rank;
    int ret = MPI_Comm_rank(ggml_backend_mpi_buffer_type_get_comm(buft), &rank);
    GGML_ASSERT(ret == MPI_SUCCESS);
    return rank;
}

int ggml_backend_mpi_local_rank(ggml_backend_t backend) {
    int rank;
    int ret = MPI_Comm_rank(ggml_backend_mpi_get_comm(backend), &rank);
    GGML_ASSERT(ret == MPI_SUCCESS);
    return rank;
}

int ggml_backend_mpi_buffer_rank(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_mpi_buffer_context *) buffer->context;
    return ctx->ctx_mpi->rank;
}

int ggml_backend_mpi_buffer_type_rank(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft->iface.get_name == ggml_backend_mpi_buffer_type_name);
    auto * ctx = (ggml_backend_mpi_buffer_type_context *) buft->context;
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(ctx->ctx_mpi != nullptr);
    return ctx->ctx_mpi->rank;
}

int ggml_backend_mpi_rank(ggml_backend_t backend) {
    auto * ctx = (ggml_mpi_context *) backend->context;
    return ctx->rank;
}

GGML_CALL static const char * ggml_backend_mpi_buffer_name(ggml_backend_buffer_t buffer);

ggml_backend_buffer_type_t ggml_backend_mpi_buffer_type_unwrap(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft != nullptr);
    auto * ctx = (ggml_backend_mpi_buffer_type_context *) buft->context;

    GGML_ASSERT(ctx != nullptr);

    ggml_backend_buffer_type_t wrapped_buffer_type = ctx->wrapped_buffer_type;
    return wrapped_buffer_type;

}

ggml_backend_buffer_t ggml_backend_mpi_buffer_unwrap(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer != nullptr);
//    fprintf(stderr, "Attempting unwrap of %s\n", ggml_backend_buffer_name(buffer));
//    if(buffer->iface.get_name != ggml_backend_mpi_buffer_name) {
//        return buffer;
//    }
    auto * ctx = (ggml_backend_mpi_buffer_context *) buffer->context;
    GGML_ASSERT(ctx != nullptr);
    ggml_backend_buffer_t wrapped_buffer = ctx->wrapped_buffer;
    GGML_ASSERT(wrapped_buffer != nullptr);
    wrapped_buffer->usage = buffer->usage;
    wrapped_buffer->size = buffer->size;
    if (wrapped_buffer->buft->iface.get_name == ggml_backend_mpi_buffer_type_name) {
        wrapped_buffer->buft = ggml_backend_mpi_buffer_type_unwrap(wrapped_buffer->buft);
    }
    return wrapped_buffer;

}




GGML_CALL static const char * ggml_backend_mpi_buffer_name(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_mpi_buffer_unwrap(buffer) != nullptr && ggml_backend_mpi_buffer_unwrap(buffer)->iface.get_name != ggml_backend_mpi_buffer_name);

    return strdup(
            (

                    "MPI Buffer(Rank " +
                    std::to_string(ggml_backend_mpi_buffer_rank(buffer)) +
                    ", local rank " +
                    std::to_string(ggml_backend_mpi_buffer_local_rank(buffer)) +
                    "):" +
                    std::string(
                            ggml_backend_buffer_name(
                                    ggml_backend_mpi_buffer_unwrap(buffer)
                            )
                    )
            ).c_str()
    );
}


GGML_CALL static const char * ggml_backend_mpi_buffer_type_name(ggml_backend_buffer_type_t buft);

GGML_CALL void ggml_backend_mpi_buffer_type_copy_ctx(ggml_backend_buffer_type_t src, ggml_backend_buffer_type_t dst) {
    if (src->iface.get_name == ggml_backend_mpi_buffer_type_name) {
        *((ggml_backend_mpi_buffer_type_context *) dst->context)->ctx_mpi = *((ggml_backend_mpi_buffer_type_context *) src->context)->ctx_mpi;
    } else {
        GGML_ASSERT(!"Buffer type must be wrapped in ggml_backend_mpi_buffer_type_t");
    }
}

GGML_CALL void ggml_backend_mpi_buffer_copy_ctx(ggml_backend_buffer_t src, ggml_backend_buffer_t dst) {
    if (src->iface.get_name == ggml_backend_mpi_buffer_name) {
        *((ggml_backend_mpi_buffer_context *) dst->context)->ctx_mpi = *((ggml_backend_mpi_buffer_context *) src->context)->ctx_mpi;
        ggml_backend_mpi_buffer_type_copy_ctx(src->buft, dst->buft);
    } else {
        GGML_ASSERT(!"Buffer must be wrapped in ggml_backend_mpi_buffer_t");
    }
}

GGML_CALL void ggml_backend_mpi_buffer_copy_ctx_from_type(ggml_backend_buffer_type_t src, ggml_backend_buffer_t dst) {
    if (src->iface.get_name == ggml_backend_mpi_buffer_type_name) {
        *((ggml_backend_mpi_buffer_context *) dst->context)->ctx_mpi = *((ggml_backend_mpi_buffer_type_context *) src->context)->ctx_mpi;
        ggml_backend_mpi_buffer_type_copy_ctx(src, dst->buft);
    } else {
        GGML_ASSERT(!"Buffer must be wrapped in ggml_backend_mpi_buffer_t");
    }
}

ggml_backend_buffer_type_t ggml_backend_mpi_buffer_type_set_wrapped_buffer_type(ggml_backend_buffer_type_t orig, ggml_backend_buffer_type_t buft) {
    if (orig->iface.get_name == ggml_backend_mpi_buffer_type_name) {
        ((ggml_backend_mpi_buffer_type_context*)(orig->context))->wrapped_buffer_type = buft;
    } else {
        GGML_ASSERT(!"Original buffer type must be an MPI buffer type.");
    }

    return orig;

}

ggml_backend_buffer_t ggml_backend_mpi_set_wrapped_buffer(ggml_backend_buffer_t orig, ggml_backend_buffer_t buf) {
    GGML_ASSERT(buf != nullptr);
    GGML_ASSERT(buf->iface.get_name != ggml_backend_mpi_buffer_name);
    if (orig->iface.get_name == ggml_backend_mpi_buffer_name) {
        ((ggml_backend_mpi_buffer_context*)(orig->context))->wrapped_buffer = buf;
        if (orig->buft != nullptr) {
            ggml_backend_mpi_buffer_type_set_wrapped_buffer_type(orig->buft, buf->buft);
        }
    } else {
        fprintf(stderr, "Original buffer name: %s\n", ggml_backend_buffer_name(orig));
        GGML_ASSERT(!"Original buffer must be an MPI buffer.");

    }
    return orig;
}

GGML_CALL static enum ggml_status ggml_backend_mpi_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {

    struct ggml_mpi_context * ctx = (ggml_mpi_context *) backend->context;


    std::vector<std::pair<ggml_backend_buffer_t, std::vector<ggml_backend_buffer_t>>> old_buffs(
            cgraph->n_nodes);
    std::vector<ggml_backend_buffer_t> old_view_buffs(cgraph->n_nodes);


    for (int i = 0; i < cgraph->n_nodes; i++) {
        old_buffs[i].first = cgraph->nodes[i]->buffer;


        for (auto &src: cgraph->nodes[i]->src) {
            if (src == nullptr) {
                break;
            }
//            fprintf(stderr, "Previous source: %s\n", src->name);
            old_buffs[i].second.push_back(src->buffer);

        }

        auto *src = cgraph->nodes[i]->view_src;
        if (src != nullptr) {
            if (src->buffer->buft != nullptr) {
                old_view_buffs[i] = src->buffer;

            }
        }
    }

    size_t n_srcs = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]->buffer->buft->iface.get_name == ggml_backend_mpi_buffer_type_name) {
            cgraph->nodes[i]->buffer = ggml_backend_mpi_buffer_unwrap(cgraph->nodes[i]->buffer);
        }

        for (auto &src: cgraph->nodes[i]->src) {
            if (src == nullptr) {
                break;
            }
            if (src->buffer->buft->iface.get_name == ggml_backend_mpi_buffer_type_name) {
                n_srcs++;
                src->buffer = ggml_backend_mpi_buffer_unwrap(src->buffer);
//                fprintf(stderr, "After unwrapping source: %s\n", src->name);

            }
        }

        auto *src = cgraph->nodes[i]->view_src;
        if (src != nullptr) {
            if (src->buffer->buft != nullptr) {

                if (src->buffer->buft->iface.get_name == ggml_backend_mpi_buffer_type_name) {
                    n_srcs++;
                    src->buffer = ggml_backend_mpi_buffer_unwrap(src->buffer);
                }
            }
        }
    }
    std::vector<ggml_backend_buffer_type_t> old_buffs_leaves;
    for (int i = 0; i < cgraph->n_leafs; i++) {
        old_buffs_leaves.push_back(cgraph->leafs[i]->buffer->buft);
        if (cgraph->leafs[i]->buffer->buft->iface.get_name == ggml_backend_mpi_buffer_type_name) {
            cgraph->leafs[i]->buffer = ggml_backend_mpi_buffer_unwrap(cgraph->leafs[i]->buffer);
        }
    }

    // TODO exploding memory usage cause we replace the buffer with the wrapped buffer,
    //  but don't free the contexts, and then create new ones when we re-wrap


    if (!ctx->remote) {
        ggml_backend_sched_t sched = ggml_backend_sched_new(ctx->backends.data(), nullptr,
                                                            (int) ctx->backends.size(), cgraph->n_nodes + cgraph->n_leafs + n_srcs, false);

//        ggml_backend_sched_reserve(sched, cgraph);
        ggml_backend_sched_graph_compute(sched, cgraph);
        ggml_backend_sched_free(sched);

    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]->buffer->iface.get_name != ggml_backend_mpi_buffer_name) {
            cgraph->nodes[i]->buffer = ggml_backend_mpi_set_wrapped_buffer(old_buffs[i].first, cgraph->nodes[i]->buffer);
        }


        for (int iter = 0; iter < GGML_MAX_SRC; iter++) {
            auto* src_node = cgraph->nodes[i]->src[iter];
            if (src_node == nullptr) {
                break;
            }

//            fprintf(stderr, "After compute src: %s\n", src_node->name);

            if (src_node->buffer->iface.get_name == ggml_backend_mpi_buffer_name) {
                continue;
            }

            src_node->buffer = ggml_backend_mpi_set_wrapped_buffer(old_buffs[i].second[iter], src_node->buffer);

//            fprintf(stderr, "After setting wrapped buffer src: %s\n", src_node->name);

        }
        if(cgraph->nodes[i]->view_src != nullptr && cgraph->nodes[i]->view_src->buffer->buft != nullptr) {

            if (old_view_buffs[i] != nullptr) {
                if (old_view_buffs[i]->iface.get_name == ggml_backend_mpi_buffer_name && cgraph->nodes[i]->view_src->buffer->iface.get_name != ggml_backend_mpi_buffer_name) {
                    cgraph->nodes[i]->view_src->buffer = ggml_backend_mpi_set_wrapped_buffer(old_view_buffs[i], cgraph->nodes[i]->view_src->buffer);
                }
            }
        }

    }


    // FIXME check if this is correct or not (it's probably not)
    for (int i = 0; i < cgraph->n_leafs; i++) {
        GGML_ASSERT(false);
        cgraph->leafs[i]->buffer = ggml_backend_mpi_wrap_buffer(cgraph->leafs[i]->buffer);
        ggml_backend_mpi_buffer_type_set_rank(cgraph->leafs[i]->buffer->buft, ctx->rank);
    }

    return GGML_STATUS_SUCCESS;
}


static const char * ggml_backend_mpi_name(ggml_backend_t backend) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    return strdup(("MPI(Rank " + std::to_string(ggml_backend_mpi_rank(backend)) + ", local rank " + std::to_string(ggml_backend_mpi_local_rank(backend)) + ")").c_str());
}

static void ggml_backend_mpi_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_mpi_context *>(backend->context);

    delete ctx;


    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_mpi_get_default_buffer_type(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_mpi_context *>(backend->context);


    auto * buff = ggml_backend_mpi_wrap_buffer_type(ctx->backends.front()->iface.get_default_buffer_type(ctx->backends.front()));
    ggml_backend_mpi_buffer_type_set_rank(buff, ctx->rank);
    return buff;
}

GGML_CALL static bool ggml_backend_mpi_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    return ggml_backend_supports_op(((ggml_mpi_context *) backend->context)->backends.front(),op);
}



std::vector<ggml_mpi_device> ggml_mpi_available_devices_internal() {
    static bool has_init = false;
    if (!has_init) {
        ggml_mpi_backend_init();
        has_init = true;
    }
    std::vector<ggml_mpi_device> devices;
    int s;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    devices.resize(s);
    for (int i = 0; i < s; i++) {
        devices[i] = ggml_mpi_device{
                i,
                ggml_mpi_init(),
                ("MPI_COMM_WORLD:" + std::to_string(i)).c_str(),
                1
        };
    }
    return devices;
}



GGML_CALL bool ggml_backend_is_mpi(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_mpi_name;
}






GGML_CALL static ggml_backend_buffer_t ggml_backend_mpi_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {

    auto* buffer = ggml_backend_mpi_wrap_buffer(
            ggml_backend_buft_alloc_buffer(ggml_backend_mpi_buffer_type_unwrap(buft), size)
            );

    ggml_backend_mpi_buffer_copy_ctx_from_type(buft, buffer);

    return buffer;
}

GGML_CALL static size_t ggml_backend_mpi_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_get_alignment(ggml_backend_mpi_buffer_type_unwrap(buft));
}

GGML_CALL static size_t ggml_backend_mpi_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_get_max_size(ggml_backend_mpi_buffer_type_unwrap(buft));
}

GGML_CALL static size_t ggml_backend_mpi_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    // Have to do this instead of calling ggml_backend_type_get_alloc_size because that signature doesn't have const on tensor
    size_t ret = ggml_backend_mpi_buffer_type_unwrap(buft)->iface.get_alloc_size(ggml_backend_mpi_buffer_type_unwrap(buft), tensor);
    return ret;
}

GGML_CALL static bool ggml_backend_mpi_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    return backend != nullptr && ggml_backend_is_mpi(backend) && ggml_backend_mpi_buffer_type_rank(buft) == ggml_backend_mpi_rank(backend)
        && ggml_backend_buft_supports_backend(ggml_backend_mpi_buffer_type_unwrap(buft), ((ggml_mpi_context*)backend->context)->backends.front());
}

GGML_CALL static bool ggml_backend_mpi_buffer_type_is_host(ggml_backend_buffer_type_t buft) {

    return ggml_backend_mpi_buffer_type_rank(buft) == ggml_backend_mpi_buffer_type_local_rank(buft) && ggml_backend_buft_is_host(ggml_backend_mpi_buffer_type_unwrap(buft));
}


static std::map<ggml_backend_buffer_type_t, ggml_backend_buffer_type_t> cached_wrappers;

static std::map<ggml_backend_buffer_t, ggml_backend_buffer_t> cached_buffer_wrappers;

static std::map<ggml_backend_t *, ggml_backend_t> cached_backends;



GGML_CALL ggml_backend_buffer_type_t ggml_backend_mpi_wrap_buffer_type(ggml_backend_buffer_type_t buft) {

    GGML_ASSERT(buft->iface.get_name != ggml_backend_mpi_buffer_type_name);


    ggml_backend_buffer_type_i ggml_backend_mpi_buffer_type_interface = {
            /* .get_name         = */ ggml_backend_mpi_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_mpi_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_mpi_buffer_type_get_alignment,
            /* .get_max_size     = */ (buft->iface.get_max_size != nullptr ) ? ggml_backend_mpi_buffer_type_get_max_size : nullptr,
            /* .get_alloc_size   = */ (buft->iface.get_alloc_size != nullptr ) ? ggml_backend_mpi_buffer_type_get_alloc_size : nullptr,
            /* .supports_backend = */ ggml_backend_mpi_buffer_type_supports_backend,
            /* .is_host          = */ (buft->iface.is_host != nullptr ) ? ggml_backend_mpi_buffer_type_is_host : nullptr,
    };



    auto* ggml_backend_wrapped_buffer_type = new ggml_backend_buffer_type {
            /* .iface    = */ ggml_backend_mpi_buffer_type_interface,
            /* .context  = */ new ggml_backend_mpi_buffer_type_context{
                                /* .name                = */ "MPI",
                                /* .wrapped_buffer_type = */ buft,
                                /* .ctx_mpi             = */ ggml_mpi_init()
                            }
    };

    // Set rank to 0 as default
    ggml_backend_mpi_buffer_type_set_rank(ggml_backend_wrapped_buffer_type, 0);


    return ggml_backend_wrapped_buffer_type;
}

GGML_CALL ggml_backend_buffer_type_t ggml_backend_mpi_wrap_buffer_type_cached(ggml_backend_buffer_type_t buft) {
    if (cached_wrappers.find(buft) != cached_wrappers.end()) {
//        fprintf(stderr, "Returning cached buffer type with name %s\n",
//                cached_wrappers[buft]->iface.get_name(cached_wrappers[buft]));


        return cached_wrappers[buft];
    }

    auto * ggml_backend_wrapped_buffer_type = ggml_backend_mpi_wrap_buffer_type(buft);
    cached_wrappers[buft] = ggml_backend_wrapped_buffer_type;
    return ggml_backend_wrapped_buffer_type;

}



GGML_CALL static void * ggml_backend_mpi_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_mpi_buffer_context *) buffer->context;
    return ctx->wrapped_buffer->iface.get_base(ctx->wrapped_buffer);
}

GGML_CALL static void ggml_backend_mpi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_mpi_buffer_context *) buffer->context;
    return ctx->wrapped_buffer->iface.free_buffer(ctx->wrapped_buffer);
}

GGML_CALL static void ggml_backend_mpi_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_mpi_buffer_context *) buffer->context;

    if (ggml_backend_mpi_buffer_rank(buffer) != ggml_backend_mpi_buffer_local_rank(buffer)) {
        return;
    }

//    fprintf(stderr, "SETTING TENSOR WITHOUT MPI CALLS FOR %s (%s) AND TGT BUFFER %s\n", tensor->name, ggml_backend_buffer_name(tensor->buffer), ggml_backend_buffer_name(buffer));
    ctx->wrapped_buffer->iface.set_tensor(ctx->wrapped_buffer, tensor, data, offset, size);
}

GGML_CALL static void ggml_backend_mpi_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    int rank = ggml_backend_mpi_buffer_local_rank(tensor->buffer);

    int src_rank = ggml_backend_mpi_buffer_rank(tensor->buffer);

//    if (ggml_backend_mpi_buffer_rank(buffer) != ggml_backend_mpi_buffer_local_rank(buffer)) {
//        return;
//    }

    if (rank != src_rank) {
//        fprintf(stderr, "Getting tensor: %s, buffer %s\n", tensor->name, ggml_backend_buffer_name(buffer));
        ggml_mpi_tensor_recv(tensor, data, ggml_backend_mpi_buffer_rank(tensor->buffer), ggml_backend_mpi_buffer_get_comm(tensor->buffer));
        return;
    }

    ggml_backend_mpi_buffer_unwrap(buffer)->iface.get_tensor(ggml_backend_mpi_buffer_unwrap(buffer), tensor, data, offset, size);
}

GGML_CALL static bool ggml_backend_mpi_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_mpi_buffer_rank(src->buffer) == ggml_backend_mpi_buffer_rank(dst->buffer) && ggml_backend_mpi_buffer_local_rank(buffer) == ggml_backend_mpi_buffer_rank(src->buffer)) {
        return ggml_backend_mpi_buffer_unwrap(buffer)->iface.cpy_tensor(ggml_backend_mpi_buffer_unwrap(buffer), src,
                                                                        dst);
    }

    return true;
}

GGML_CALL static void ggml_backend_mpi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    return ggml_backend_mpi_buffer_unwrap(buffer)->iface.clear(ggml_backend_mpi_buffer_unwrap(buffer), value);
}

GGML_CALL static void ggml_backend_mpi_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
//    fprintf(stderr, "Init tensor with buffer %s, tensor %s, tensor buffer %s, tensor view src %s, tensor vs buff %s\n",
//            ggml_backend_buffer_name(buffer), tensor->name, ggml_backend_buffer_name(tensor->buffer), tensor->view_src !=
//                    nullptr ? tensor->view_src->name : "", tensor->view_src != nullptr ? ggml_backend_buffer_name(tensor->view_src->buffer) : "");
    auto *orig_buffer = tensor->buffer;
    tensor->buffer = ggml_backend_mpi_buffer_unwrap(tensor->buffer);

    bool view_src_null = tensor->view_src == nullptr;
    ggml_backend_buffer_t orig_view_src_buffer = nullptr;
    if (!view_src_null) {
         orig_view_src_buffer = tensor->view_src->buffer;
        tensor->view_src->buffer = ggml_backend_mpi_buffer_unwrap(tensor->view_src->buffer);
    }

    std::vector<ggml_backend_buffer_t> orig_src_buffers(0);
    for (auto & src : tensor->src) {
        if (src == nullptr) {
            break;
        }


        orig_src_buffers.push_back(src->buffer);

        if (src->buffer != nullptr && src->buffer->iface.get_name == ggml_backend_mpi_buffer_name) {
            src->buffer = ggml_backend_mpi_buffer_unwrap(src->buffer);
        }
    }


    ggml_backend_buffer_init_tensor(ggml_backend_mpi_buffer_unwrap(buffer), tensor);
    tensor->buffer = ggml_backend_mpi_set_wrapped_buffer(orig_buffer, tensor->buffer);
    if (!view_src_null) {
        tensor->view_src->buffer = ggml_backend_mpi_set_wrapped_buffer(orig_view_src_buffer, tensor->view_src->buffer);
    }

    for (size_t i = 0; i < orig_src_buffers.size(); i++) {
        if (orig_src_buffers[i]->iface.get_name == ggml_backend_mpi_buffer_name) {
            tensor->src[i]->buffer = ggml_backend_mpi_set_wrapped_buffer(orig_src_buffers[i], tensor->src[i]->buffer);
        }
    }
}





GGML_CALL ggml_backend_buffer_t ggml_backend_mpi_wrap_buffer(ggml_backend_buffer_t buf) {

    struct ggml_backend_buffer_i mpi_backend_buffer_i = {
            /* .get_name        = */ ggml_backend_mpi_buffer_name,
            /* .free_buffer     = */ ggml_backend_mpi_buffer_free_buffer,
            /* .get_base        = */ ggml_backend_mpi_buffer_get_base,
            /* .init_tensor     = */ (buf->iface.init_tensor != nullptr) ? ggml_backend_mpi_buffer_init_tensor : nullptr,
            /* .set_tensor      = */ ggml_backend_mpi_buffer_set_tensor,
            /* .get_tensor      = */ ggml_backend_mpi_buffer_get_tensor,
            /* .cpy_tensor      = */ ggml_backend_mpi_buffer_cpy_tensor,
            /* .clear           = */ ggml_backend_mpi_buffer_clear,
            /* .reset           = */ nullptr,
    };

//    if (cached_buffer_wrappers.find(buf) != cached_buffer_wrappers.end()) {
//        fprintf(stderr, "Returning cached buffer with name %s\n", cached_buffer_wrappers[buf]->iface.get_name(cached_buffer_wrappers[buf]));
//        auto * ret = new ggml_backend_buffer;
//        *ret = *cached_buffer_wrappers[buf];
//        auto * ret_type = new ggml_backend_buffer_type;
//        *ret_type = *ret->buft;
//        ret->buft = ret_type;
//        return ret;
//    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//    fprintf(stderr, "Wrapping buffer %s at rank %d\n", ggml_backend_buffer_name(buf), rank);

    if (buf->iface.get_name == ggml_backend_mpi_buffer_name) {
        fprintf(stderr, "WRAPPING AN ALREADY WRAPPED BUFFER: %s\n", ggml_backend_buffer_name(buf));
        GGML_ASSERT(false);
    }

    ggml_backend_buffer_type_t t = ggml_backend_mpi_wrap_buffer_type(buf->buft);

    auto *buffer = new ggml_backend_buffer {
            /* .interface = */ mpi_backend_buffer_i,
            /* .buft      = */ t,
            /* .context   = */ new ggml_backend_mpi_buffer_context{
                                buf, ggml_mpi_init()},
            /* .size      = */ buf->size,
            /* .usage     = */ buf->usage
    };

    // Default to node 0 when wrapping buffers
    ggml_backend_mpi_buffer_set_rank(buffer, 0);

    cached_buffer_wrappers[buf] = buffer;



    return buffer;
}

bool ggml_backend_mpi_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, struct ggml_tensor * src, struct ggml_tensor * dst) {
    int src_rank = ggml_backend_mpi_buffer_rank(src->buffer);
    int dst_rank = ggml_backend_mpi_buffer_rank(dst->buffer);

    auto * src_ctx = static_cast<ggml_mpi_context *>(backend_src->context);
    auto * dst_ctx = static_cast<ggml_mpi_context *>(backend_dst->context);


    if (src_ctx->remote && dst_ctx->remote) {
        return true;
    }

    if (src_rank == dst_rank) {
        src->buffer = ggml_backend_mpi_buffer_unwrap(src->buffer);
        if (src->view_src) {
            src->view_src->buffer = ggml_backend_mpi_buffer_unwrap(src->view_src->buffer);
        }
        dst->buffer = ggml_backend_mpi_buffer_unwrap(dst->buffer);
        if (dst->view_src) {
            dst->view_src->buffer = ggml_backend_mpi_buffer_unwrap(dst->view_src->buffer);
        }
        ggml_backend_tensor_copy_async(((ggml_mpi_context *) backend_src->context)->backends.front(),((ggml_mpi_context *) backend_dst->context)->backends.front(), src, dst);

        src->buffer = ggml_backend_mpi_wrap_buffer(src->buffer);
        if (src->view_src) {
            src->view_src->buffer = ggml_backend_mpi_wrap_buffer(src->view_src->buffer);
        }
        dst->buffer = ggml_backend_mpi_wrap_buffer(dst->buffer);
        if (dst->view_src) {
            dst->view_src->buffer = ggml_backend_mpi_wrap_buffer(dst->view_src->buffer);
        }
//        src->buffer->iface.cpy_tensor(src->buffer, src, dst);
        return true;
    }

    if (src_rank == ggml_backend_mpi_local_rank(backend_src)) {
        ggml_mpi_tensor_send(src, dst_rank, dst_ctx->comm);
    } else if (dst_rank == ggml_backend_mpi_local_rank(backend_dst)){
        ggml_mpi_tensor_recv(dst, src_rank, src_ctx->comm);
    }
//    fprintf(stderr, "ATTEMPTING ASYNC COPY FOR SRC TENSOR %s TO DST TENSOR %s WITH SRC BACKEND %s AND DST BACKEND %s\n", src->name, dst->name, ggml_backend_name(backend_src), ggml_backend_name(backend_dst));
    return true;

}

void ggml_backend_mpi_set_tensor_async(ggml_backend_t backend, struct ggml_tensor * dst, const void* data, size_t offset, size_t size) {
    int dst_rank = ggml_backend_mpi_buffer_rank(dst->buffer);


    auto * ctx = static_cast<ggml_mpi_context *>(backend->context);

    GGML_ASSERT(ctx->rank == dst_rank);

    if (dst_rank == ggml_backend_mpi_buffer_local_rank(dst->buffer)) {
        auto * old_buffer = dst->buffer;
        dst->buffer = ggml_backend_mpi_buffer_unwrap(dst->buffer);
        if (dst->view_src) {
            dst->view_src->buffer = ggml_backend_mpi_buffer_unwrap(dst->view_src->buffer);
        }
        ggml_backend_tensor_set_async(((ggml_mpi_context *) backend->context)->backends.front(), dst, data, offset, size);
        dst->buffer = ggml_backend_mpi_wrap_buffer(dst->buffer);
        if (dst->view_src) {
            dst->view_src->buffer = ggml_backend_mpi_wrap_buffer(dst->view_src->buffer);
        }
//        dst->buffer = old_buffer;
    } else {

        ggml_mpi_tensor_send(dst, data, ctx->rank, ctx->comm);
    }


}

GGML_CALL static void ggml_backend_mpi_synchronize(ggml_backend_t backend) {
    if (!((ggml_mpi_context*)backend->context)->remote) {
        ggml_backend_synchronize(((ggml_mpi_context*)backend->context)->backends.front());
    }
}

ggml_backend_t ggml_backend_mpi_init(ggml_backend_t * wrapped_backends, size_t num_backends, int rank) {

    static ggml_guid backend_mpi_guid = {0xec, 0x39, 0xce, 0x40, 0xc3, 0x43, 0x49, 0x36, 0x96, 0x03, 0x55, 0x77, 0x5c, 0x1f, 0x44, 0xd3};


    ggml_mpi_context * ctx = ggml_mpi_init();
    std::vector<ggml_backend_t> wrapped_backends_v;
    for (size_t i = 0; i < num_backends; i++) {
        wrapped_backends_v.push_back(wrapped_backends[i]);
    }
    if (ctx->rank == rank) {

    } else {
        ctx->remote = true;
    }
    ctx->backends = wrapped_backends_v;
    ctx->rank = rank;
    struct ggml_backend_i mpi_backend_i = {
            /* .get_name                = */ ggml_backend_mpi_name,
            /* .free                    = */ ggml_backend_mpi_free,
            /* .get_default_buffer_type = */ ggml_backend_mpi_get_default_buffer_type,
            /* .set_tensor_async        = */ ggml_backend_mpi_set_tensor_async,
            /* .get_tensor_async        = */ nullptr,
            /* .cpy_tensor_async        = */ ggml_backend_mpi_cpy_tensor_async,
            /* .synchronize             = */ ggml_backend_mpi_synchronize,
            /* .graph_plan_create       = */ nullptr,
            /* .graph_plan_free         = */ nullptr,
            /* .graph_plan_compute      = */ nullptr,
            /* .graph_compute           = */ ggml_backend_mpi_graph_compute,
            /* .supports_op             = */ ggml_backend_mpi_supports_op,
            /* .offload_op              = */ nullptr,
            /* .event_new               = */ nullptr,
            /* .event_free              = */ nullptr,
            /* .event_record            = */ nullptr,
            /* .event_wait              = */ nullptr,
            /* .event_synchronize       = */ nullptr,
    };

    auto *mpi_backend = new ggml_backend {
            /* .guid      = */ &backend_mpi_guid,
            /* .interface = */ mpi_backend_i,
            /* .context   = */ ctx,
    };

    cached_backends[wrapped_backends] = mpi_backend;

    return mpi_backend;
}

static ggml_backend_t ggml_backend_reg_mpi_init(const char * params, void * user_data) {
    // TODO check what the parameters are for. Could use it to setup the MPI comms and routes?
    GGML_UNUSED(params);
    ggml_mpi_backend_init();
    auto * v = new std::vector<ggml_backend_t>();
    v->push_back(ggml_backend_cpu_init());
    return ggml_backend_mpi_init(v->data(), 1, 0);
}




extern "C" GGML_CALL int ggml_backend_mpi_reg_devices();

int ggml_backend_mpi_reg_devices() {
    auto devices = ggml_mpi_available_devices_internal();
    for (const auto & device : devices) {
        ggml_backend_register(
                device.name,
                ggml_backend_reg_mpi_init,
                ggml_backend_mpi_wrap_buffer_type(ggml_backend_cpu_buffer_type()),
                reinterpret_cast<void *>(intptr_t(device.index))
        );
    }
    return devices.size();
}



GGML_CALL void ggml_backend_mpi_buffer_type_set_rank(ggml_backend_buffer_type_t buft, int rank) {
    if (buft->iface.get_name == ggml_backend_mpi_buffer_type_name) {
        ((ggml_backend_mpi_buffer_type_context *) buft->context)->ctx_mpi->rank = rank;
    } else {
        GGML_ASSERT(!"Buffer type must be wrapped in ggml_backend_mpi_buffer_type");
    }
}

GGML_CALL void ggml_backend_mpi_buffer_set_rank(ggml_backend_buffer_t buf, int rank) {
    if (buf->iface.get_name == ggml_backend_mpi_buffer_name) {
        ((ggml_backend_mpi_buffer_context *) buf->context)->ctx_mpi->rank = rank;
        ggml_backend_mpi_buffer_type_set_rank(buf->buft, rank);
    } else {
        GGML_ASSERT(!"Buffer type must be wrapped in ggml_backend_mpi_buffer_type");
    }
}


