#pragma once
#include <stdint.h>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The context used for MPI operations,
 * a program may make use of more than one
 * context but must always have at least one.
 *
 * The context stores required information like the
 * node rank and a communicator to use for MPI operations.
 * A context is guaranteed to be internally consistent,
 * meaning that a context's stored rank is valid within
 * the context's communicator.
 */
struct ggml_mpi_context;


/**
 * Initialize the MPI library and the GGML MPI backend.
 * Calling more than once during the lifetime of the program
 * leads to undefined behavior. This function must be called before
 * any MPI operations.
 */
void ggml_mpi_backend_init(void);

/**
 * Frees the MPI backend, must be called only once at termination
 * of the program. No MPI operations may be completed after calling this function,
 * and attempting to do so will lead to undefined behavior.
 */
void ggml_mpi_backend_free(void);

/**
 * Construct a new MPI context using the MPI_WORLD
 * communicator. This is useful only to create the
 * initial context, as calling multiple times
 * will only create effective copies of the same data.
 *
 * @return A context for us in the global communicator.
 */
struct ggml_mpi_context * ggml_mpi_init(void);

/**
 * Create a new context by splitting the given context's
 * communicator, creating a "sub-communicator." This is a collective
 * operation and must be performed by all nodes within the same communicator.
 * The color and key have the same meaning as in MPI_Comm_split(), i.e.
 * the color is used to determine the sub-communicator this node will belong to,
 * and the key is the relative rank of this node in the new communicator.
 *
 * An example: if a node passes a color of 1, and a different node passes a color of 2,
 * the nodes will belong to two different sub-communicators. If two nodes pass the same
 * color, then their ranks will be ordered by the order of their keys. If they pass the same
 * key, then the tie will be broken by the nodes' ranks in the old communicator.
 *
 * The communicator used by the given context remains entirely valid, so it is advisable
 * to store both the old and new contexts. This allows an application to
 * select at runtime which communicator to perform MPI operations with. An example
 * would be to segregate the nodes into multiple domains categorized by the functions
 * they perform, and use the original context to broadcast to all nodes in the cluster.
 *
 * @param ctx The context containing the communicator to split.
 * @param color The sub-communicator that this node will belong to.
 * @param key The relative rank of this node in the new communicator.
 * @return A new context with all values referencing the newly-created communicator.
 */
struct ggml_mpi_context * ggml_mpi_split_comm(struct ggml_mpi_context * ctx, int color, int key);

/**
 * Frees the given context, including the communicator. No MPI
 * operations besides ggml_mpi_backend_freee(void) should be executed after
 * running this function.
 *
 * @param ctx The context to free.
 */
void ggml_mpi_free(struct ggml_mpi_context * ctx);

/**
 * Get the rank of this node in the given context's communicator.
 *
 * @param ctx The context to use to determine the rank with regards to.
 * @return The rank of this node.
 */
int ggml_mpi_rank(struct ggml_mpi_context * ctx);

/**
 * Get the number of nodes that are a part of
 * the communicator referenced by the given context.
 *
 * @param ctx The context containing the communicator used for this size check.
 * @return The number of nodes that are a part of the given context's communicator.
 */
int ggml_mpi_size(struct ggml_mpi_context * ctx);

/**
 * Synchronize needed information among the nodes
 * to prepare for running an evaluation iteration.
 * This is a collective operation and all nodes must
 * call this function. It will block until all
 * nodes have entered it, to prevent any desync
 * between nodes.
 *
 * @param ctx_mpi The context in which to prepare for evaluation.
 * @param n_tokens A pointer to the n_tokens, which will be synchronized after this function.
 * @param n_past A pointer to the n_past, which will be synchronized after this function.
 * @param n_threads A pointer to the n_threads, which is unused currently.
 */
void ggml_mpi_eval_init(
        struct ggml_mpi_context * ctx_mpi,
                            int * n_tokens,
                            int * n_past,
                            int * n_threads);

/**
 * Split a range across all nodes within the given
 * context, weighting the allocations by the given weights.
 * The dimensions of the returned 2d array are (number of nodes in the context, 2).
 * The first element in the inner array is the starting point of the range allocated
 * to the node indicated by the index into the outer array,
 * and the second element is the end point of the allocated range, inclusive.
 *
 * @param ctx_mpi The context used to determine the number of nodes
 *                to split the range across.
 * @param start The starting point of the range.
 * @param end The end point of the range, inclusive.
 * @param node_weights How to weight the allocations across the nodes,
 *                     must sum to 1.0.
 * @return A 2d array, the first dimension is the number of nodes in the context
 *         and the second dimension is 2.
 */
uint16_t** ggml_mpi_split_range(
    struct ggml_mpi_context * ctx_mpi,
    uint16_t start,
    uint16_t end,
    float node_weights[]
);

/**
 * Scatter the layer ranges across all nodes
 * in the given context. This is a collective operation
 * and must be called by all nodes that are within the same
 * communicator. The given layer ranges must be in the same
 * format as created by the ggml_mpi_split_range().
 *
 * @param ctx_mpi The context to scatter the layers across.
 * @param layer_ranges The pre-split ranges to scatter to the nodes.
 */
void ggml_mpi_scatter_layers(
    struct ggml_mpi_context * ctx_mpi,
    uint16_t ** layer_ranges
);

/**
 * Modify compute graph to only process allocated
 * layers.
 *
 * @param ctx_mpi The context containing the allocated layer range.
 * @param gf The compute graph to modify
 * @param n_layers The number of layers in the model, used as an upper bound in the layer ranges.
 */
void ggml_mpi_graph_compute_pre(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf,
                            int   n_layers);

/**
 * Sends the output tensor to the next node for processing
 * of later layers.
 *
 * @param ctx_mpi The context to use for MPI operations.
 * @param gf The graph used in the computations
 * @param n_layers The number of layers in the model.
 */
void ggml_mpi_graph_compute_post(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf,
                            int   n_layers);

#ifdef __cplusplus
}
#endif
