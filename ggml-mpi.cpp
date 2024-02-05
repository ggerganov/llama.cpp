#include "ggml-mpi.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define UNUSED GGML_UNUSED

struct ggml_mpi_context {
    int rank;
    int size;
    MPI_Comm comm;
    int layer_start;
    int layer_end;
    struct ggml_tensor *inp0;
    std::string name;
};

void ggml_mpi_backend_init(void) {
    int ret;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &ret);
}

void ggml_mpi_backend_free(void) {
    MPI_Finalize();
}

struct ggml_mpi_context * ggml_mpi_init(void) {
    auto * ctx = static_cast<ggml_mpi_context *>(calloc(1, sizeof(struct ggml_mpi_context)));

    MPI_Comm_rank(MPI_COMM_WORLD, &ctx->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx->size);
    ctx->comm = MPI_COMM_WORLD;

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

void ggml_mpi_eval_init(
        struct ggml_mpi_context *   ctx_mpi,
                int32_t         *   n_tokens,
                int32_t         **  pos,
                int32_t         **  n_seq_ids,
                int32_t         *** seq_id,
                int8_t          **  logits) {


    MPI_Barrier(ctx_mpi->comm);
    int32_t old_n_tokens = *n_tokens;
    MPI_Bcast(n_tokens, 1, MPI_INT32_T, 0, ctx_mpi->comm);

    // If what was passed in differs from what was broadcast,
    // we can't guarantee the allocated sizes are correct
    // TODO check how often this is done and if it's a problem,
    //      try to allocate ahead of time
    if (old_n_tokens != *n_tokens) {
        *pos = static_cast<int32_t *>(realloc(*pos, *n_tokens * sizeof(int32_t)));
        *n_seq_ids = static_cast<int32_t *>(realloc(*n_seq_ids, *n_tokens * sizeof(int32_t)));
        *logits = static_cast<int8_t *>(realloc(*logits, *n_tokens * sizeof(int32_t)));
    }



//    MPI_Bcast(&total_n_seq_ids,     1, MPI_INT32_T, 0, ctx_mpi->comm);
    MPI_Bcast(*n_seq_ids,   *n_tokens, MPI_INT32_T, 0, ctx_mpi->comm);

    // We need to know the total number of sequence
    // ids, so we count them all up
    int32_t total_n_seq_ids = 0;
    for (int32_t i = 0; i < *n_tokens; i++) {
        total_n_seq_ids += (*n_seq_ids)[i];
    }

    // MPI can't chase the pointers for multidimensional arrays, so we flatten them first
    // for transit
    auto * flattened_seq_ids = static_cast<int32_t *>(calloc(total_n_seq_ids, sizeof(int32_t)));

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


    MPI_Bcast(             *pos, *n_tokens,        MPI_INT32_T, 0, ctx_mpi->comm);
    MPI_Bcast(flattened_seq_ids,  total_n_seq_ids, MPI_INT32_T, 0, ctx_mpi->comm);
    //MPI_Bcast(*logits,               *n_tokens,        MPI_INT8_T, 0, ctx_mpi->comm);
    auto ** new_seq_id = static_cast<int32_t **>(calloc(*n_tokens, sizeof(int32_t *)));
    current_index = 0;
    for (int32_t i = 0; i < *n_tokens; i++) {
        new_seq_id[i] = static_cast<int32_t *>(calloc((*n_seq_ids)[i], sizeof(int32_t)));
        for (int32_t j = 0; j < (*n_seq_ids)[i]; j++) {
            new_seq_id[i][j] = flattened_seq_ids[current_index];
            current_index++;
        }
    }
    free(flattened_seq_ids);
    //free(*seq_id); // <- something is still holding onto this, need to investigate
    *seq_id = new_seq_id;
}

void ggml_mpi_synch_int(
        struct ggml_mpi_context * ctx_mpi,
                        int32_t * val
) {
    MPI_Bcast(val, 1, MPI_INT32_T, 0, ctx_mpi->comm);
}

static int ggml_graph_get_node_idx(struct ggml_cgraph * gf, const char * name) {
    struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
    if (t == NULL) {
        fprintf(stderr, "%s: tensor %s not found\n", __func__, name);
        return -1;
    }

    for (int i = 0; i < gf->n_nodes; i++) {
        if (gf->nodes[i] == t) {
            return i;
        }
    }

    fprintf(stderr, "%s: tensor %s not found in graph (should not happen)\n", __func__, name);
    return -1;
}


static void ggml_mpi_tensor_send(struct ggml_tensor * t, int mpi_rank_dst, MPI_Comm comm) {
    MPI_Datatype mpi_type;

    switch (t->type) {
        case GGML_TYPE_I32: mpi_type = MPI_INT32_T; break;
        case GGML_TYPE_F32: mpi_type = MPI_FLOAT;   break;
        default: GGML_ASSERT(false && "not implemented");
    }

    const int retval = MPI_Send(t->data, ggml_nelements(t), mpi_type, mpi_rank_dst, 0, comm);
    GGML_ASSERT(retval == MPI_SUCCESS);
}

static void ggml_mpi_tensor_recv(struct ggml_tensor * t, int mpi_rank_src, MPI_Comm comm) {
    MPI_Datatype mpi_type;

    switch (t->type) {
        case GGML_TYPE_I32: mpi_type = MPI_INT32_T; break;
        case GGML_TYPE_F32: mpi_type = MPI_FLOAT;   break;
        default: GGML_ASSERT(false && "not implemented");
    }

    MPI_Status status; UNUSED(status);
    fprintf(stderr, "%s: tensor receive == null: %d\n", __func__, t->data == NULL);
    const int retval = MPI_Recv(t->data, ggml_nelements(t), mpi_type, mpi_rank_src, MPI_ANY_TAG, comm, &status);
    GGML_ASSERT(retval == MPI_SUCCESS);
}

uint16_t** ggml_mpi_split_range(
    struct ggml_mpi_context * ctx_mpi,
    uint16_t start,
    uint16_t end,
    float node_weights[]
) {
    // Splits the range given by start and end
    // over the available nodes. This implementation
    // assumes that node 0 handles the final part of the range
    // while node 1 handles the beginning, to form a ring pipeline

    // Only node 0 deals with the device splits, other nodes
    // get the splits from the scatter layers operation

    if (ctx_mpi->rank != 0) {
        return NULL;
    }

    uint16_t range_length = end - start + 1;
    uint16_t ** ranges = (uint16_t**) malloc(sizeof(uint16_t*) * ctx_mpi->size);
    for (int i = 0; i < ctx_mpi->size; i++) {
        ranges[i] = (uint16_t*) malloc(sizeof(uint16_t) * 2);
    }
    uint16_t next_layer = 0;
    for (int i=1; i < ctx_mpi->size; i++) {
        ranges[i][0] = next_layer;
        ranges[i][1] = MIN(end, ranges[i][0] + (node_weights[i] * range_length) + start);
        next_layer = ranges[i][1];
    }

    ranges[0][0] = next_layer;
    ranges[0][1] = MIN(end, next_layer + (node_weights[0] * range_length) + start);
    return ranges;

}

void ggml_mpi_scatter_layers(
    struct ggml_mpi_context * ctx_mpi,
    uint16_t ** layer_ranges
) {
    // Layer ranges is a 2d array with the first dimension
    // having a length of the number of nodes and the second
    // dimension having a length of 2. The inner arrays contain
    // the start and end layer ID for a node.
    uint16_t flattened_ranges[ctx_mpi->size * 2];

    if (layer_ranges != NULL) {
        for (int i = 0; i < ctx_mpi->size * 2; i += 2) {
            flattened_ranges[i] = layer_ranges[i/2][0];
            flattened_ranges[i + 1] = layer_ranges[i/2][1];
        }
    }

    uint16_t received_range[2];
    MPI_Scatter(flattened_ranges, 2, MPI_UINT16_T, received_range, 2, MPI_UINT16_T, 0, ctx_mpi->comm);
    ctx_mpi->layer_start = received_range[0];
    ctx_mpi->layer_end = received_range[1];
    fprintf(stderr, "Ranges for rank %d: [%d, %d]\n", ctx_mpi->rank, ctx_mpi->layer_start, ctx_mpi->layer_end);
}

void ggml_mpi_graph_creation_post(struct ggml_mpi_context * ctx_mpi, struct ggml_cgraph * gf, int   n_layers) {
    const int mpi_rank = ctx_mpi->rank;
    const int mpi_size = ctx_mpi->size;

    struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (inp_tokens == NULL) {
        fprintf(stderr, "%s: tensor 'inp_tokens' not found\n", __func__);
        return;
    }

    struct ggml_tensor * inp0 = ggml_graph_get_tensor(gf, "layer_inp_0");
    if (inp0 == NULL) {
        fprintf(stderr, "%s: tensor 'inp0' not found\n", __func__);
        return;
    }

    ctx_mpi->inp0 = inp0;

//    fprintf(stderr, "gf->nodes[0] == %s\n", ggml_get_name(gf->nodes[0]));
//
//    GGML_ASSERT(inp0 == gf->nodes[0]);

    // distribute the compute graph into slices across the MPI nodes
    //
    // the main node (0) processes the last layers + the remainder of the compute graph
    // and is responsible to pass the input tokens to the first node (1)
    //
    // node 1:   [(  0) * n_per_node, (  1) * n_per_node)
    // node 2:   [(  1) * n_per_node, (  2) * n_per_node)
    // ...
    // node n-1: [(n-2) * n_per_node, (n-1) * n_per_node)
    // node 0:   [(n-1) * n_per_node,            n_nodes)
    //


    for (int i = 0; i < gf->n_nodes; i++) {
        gf->nodes[i]->backend = GGML_BACKEND_MPI_SPLIT;
    }


    {


        //const int n_per_node = (n_layers + (mpi_size - 1)) / mpi_size;

        const int mpi_idx = mpi_rank > 0 ? mpi_rank - 1 : mpi_size - 1;

        //const int il0 =               (mpi_idx + 0) * n_per_node;
        //const int il1 = MIN(n_layers, (mpi_idx + 1) * n_per_node);
        int il0 = ctx_mpi->layer_start;
        int il1 = MIN(n_layers, ctx_mpi->layer_end);

        char name_l0[GGML_MAX_NAME];
        char name_l1[GGML_MAX_NAME];

        snprintf(name_l0, sizeof(name_l0), "layer_inp_%d", il0);
        snprintf(name_l1, sizeof(name_l1), "layer_inp_%d", il1);

        const int idx_l0 =                ggml_graph_get_node_idx(gf, name_l0);
        const int idx_l1 = mpi_rank > 0 ? ggml_graph_get_node_idx(gf, name_l1) + 1 : gf->n_nodes;

        if (idx_l0 < 0 || idx_l1 < 0) {
            fprintf(stderr, "%s: layer input nodes not found\n", __func__);
            return;
        }

        // attach the input data to all nodes that need it
        // TODO: not great - should be able to do this without modifying the compute graph (see next TODO below)
        for (int i = idx_l0; i < idx_l1; i++) {
            if (gf->nodes[i]->src[0] == gf->nodes[idx_l0]) {
                gf->nodes[i]->src[0] =  inp0;
            }
            if (gf->nodes[i]->src[1] == gf->nodes[idx_l0]) {
                gf->nodes[i]->src[1] =  inp0;
            }
        }

        // TODO: instead of rearranging the nodes, we should be able to execute a subset of the compute graph
        for (int i = 1; i < idx_l1 - idx_l0; i++) {
            gf->nodes[i] = gf->nodes[idx_l0 + i];
        }

        // the first node performs the "get_rows" operation, the rest of the nodes get the data from the previous node
        if (mpi_idx != 0) {
            gf->nodes[0]->op = GGML_OP_NONE;
        }

        gf->n_nodes = idx_l1 - idx_l0;

    }
}

// TODO: there are many improvements that can be done to this implementation
void ggml_mpi_graph_compute_pre(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf,
                            int   n_layers) {
    const int mpi_rank = ctx_mpi->rank;
    const int mpi_size = ctx_mpi->size;

    struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (inp_tokens == NULL) {
        fprintf(stderr, "%s: tensor 'inp_tokens' not found\n", __func__);
        return;
    }

    struct ggml_tensor * inp0 = ctx_mpi->inp0;
    if (inp0 == NULL) {
        fprintf(stderr, "%s: tensor 'inp0' not found\n", __func__);
        return;
    }

    if (mpi_rank > 0) {
        if (mpi_rank == 1) {
            // the first node (1) receives the input tokens from the main node (0)
            if (inp_tokens->data == NULL) {

            }
            ggml_mpi_tensor_recv(inp_tokens, 0, ctx_mpi->comm);
        } else {
            // recv input data for each node into the "inp0" tensor (i.e. the first node in the compute graph)
            fprintf(stderr, "%s:%d: receiving layer inp0\n", __func__, ctx_mpi->rank);
            ggml_mpi_tensor_recv(inp0, mpi_rank - 1, ctx_mpi->comm);
        }
    } else if (mpi_size > 1) {
        // node 0 sends the input tokens to node 1
        ggml_mpi_tensor_send(inp_tokens, 1, ctx_mpi->comm);

        // recv the output data from the last node
        ggml_mpi_tensor_recv(inp0, mpi_size - 1, ctx_mpi->comm);
    }
}

void ggml_mpi_graph_compute_post(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf,
                            int   n_layers) {
    UNUSED(n_layers);

    const int mpi_rank = ctx_mpi->rank;
    const int mpi_size = ctx_mpi->size;

    // send the output data to the next node
    if (mpi_rank > 0) {
        ggml_mpi_tensor_send(gf->nodes[gf->n_nodes - 1], (mpi_rank + 1) % mpi_size, ctx_mpi->comm);
    }
}

// BACKEND V2

static const char * ggml_backend_mpi_name(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_mpi_context *>(backend->context);
    return ctx->name.c_str();
}

static void ggml_backend_mpi_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_mpi_context *>(backend->context);

    delete ctx;


    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_mpi_get_default_buffer_type(ggml_backend_t backend) {
    return ggml_backend_cpu_buffer_type();
}

GGML_CALL static bool ggml_backend_mpi_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_CPY:
            return op->type != GGML_TYPE_IQ2_XXS && op->type != GGML_TYPE_IQ2_XS; // missing type_traits.from_float
        case GGML_OP_MUL_MAT:
            return op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == ggml_internal_get_type_traits(op->src[0]->type).vec_dot_type;
        default:
            return true;
    }

    GGML_UNUSED(backend);
}

static struct ggml_backend_i mpi_backend_i = {
        /* .get_name                = */ ggml_backend_mpi_name,
        /* .free                    = */ ggml_backend_mpi_free,
        /* .get_default_buffer_type = */ ggml_backend_mpi_get_default_buffer_type,
        /* .set_tensor_async        = */ NULL,
        /* .get_tensor_async        = */ NULL,
        /* .cpy_tensor_async        = */ NULL,
        /* .synchronize             = */ NULL,
        /* .graph_plan_create       = */ NULL,
        /* .graph_plan_free         = */ NULL,
        /* .graph_plan_compute      = */ NULL,
        /* .graph_compute           = */ ggml_backend_graph_compute,
        /* .supports_op             = */ ggml_backend_mpi_supports_op,
};


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

ggml_backend_buffer_type_t ggml_backend_mpi_wrap_buffer(ggml_backend_buffer_type_t buft) {
    auto* ggml_backend_wrapped_buffer_type = new ggml_backend_buffer_type{
            /* .iface    = */ buft->iface,
            /* .context  = */ buft->context,
    };

    return ggml_backend_wrapped_buffer_type;
}

ggml_backend_t ggml_backend_mpi_init(int index) {
    auto *mpi_backend = new ggml_backend {
            /* .interface = */ mpi_backend_i,
            /* .context   = */ ggml_mpi_init(),
    };

    return mpi_backend;
}

static ggml_backend_t ggml_backend_reg_mpi_init(const char * params, void * user_data) {
    GGML_UNUSED(params);
    return ggml_backend_mpi_init(intptr_t(user_data));
}



ggml_backend_buffer_type_t ggml_backend_mpi_buffer_type(int index) {
    return ggml_backend_cpu_buffer_type();
}

extern "C" GGML_CALL int ggml_backend_mpi_reg_devices();

int ggml_backend_mpi_reg_devices() {
    auto devices = ggml_mpi_available_devices_internal();
    for (const auto & device : devices) {
        ggml_backend_register(
                device.name,
                ggml_backend_reg_mpi_init,
                ggml_backend_mpi_buffer_type(device.index),
                reinterpret_cast<void *>(intptr_t(device.index))
        );
    }
    return devices.size();
}




