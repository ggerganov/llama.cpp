#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_RPC_MAX_SERVERS       16

struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];
};

enum rpc_cmd {
    ALLOC_BUFFER = 0,
    BUFFER_GET_BASE,
    FREE_BUFFER,
    BUFFER_CLEAR,
    SET_TENSOR,
    GET_TENSOR,
    COPY_TENSOR,
    GRAPH_COMPUTE,
};

GGML_API GGML_CALL void ggml_rpc_init(const char * rpc_servers);

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_rpc_init(int server_id);
GGML_API GGML_CALL bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(int server_id);

GGML_API GGML_CALL int  ggml_backend_rpc_get_server_count(void);

GGML_API GGML_CALL void rpc_serve_client(ggml_backend_t backend, int sockfd);

#ifdef  __cplusplus
}
#endif
