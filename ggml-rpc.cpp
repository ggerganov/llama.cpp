#include "ggml-rpc.h"
#include "ggml.h"
#include "ggml-backend-impl.h"

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netdb.h>
#include <string.h>
#include <unistd.h>

#define UNUSED GGML_UNUSED

#define GGML_DEBUG 1
#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = { 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
    return &guid;
}

struct ggml_backend_rpc_buffer_type_context {
    int sockfd;
    std::string name;
};

struct ggml_backend_rpc_context {
    std::string endpoint;
    std::string name;
    int sockfd;
    ggml_backend_buffer_type_t buft;
};

struct ggml_backend_rpc_buffer_context {
    int sockfd;
    uint64_t remote_ptr;
    std::string name;
};


// RPC helper functions

static int socket_connect(const char * host, int port) {
    struct sockaddr_in addr;
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        return -1;
    }
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    struct hostent * server = gethostbyname(host);
    if (server == NULL) {
        fprintf(stderr, "Cannot resolve host '%s'\n", host);
        return -1;
    }
    bcopy((char *)server->h_addr, (char *)&addr.sin_addr.s_addr, server->h_length);
    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        return -1;
    }
    return sock;
}

static bool send_data(int sockfd, const void * data, size_t size) {
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        ssize_t n = send(sockfd, (const uint8_t *)data + bytes_sent, size - bytes_sent, 0);
        if (n < 0) {
            return false;
        }
        bytes_sent += n;
    }
    return true;
}

static bool recv_data(int sockfd, void * data, size_t size) {
    size_t bytes_recv = 0;
    while (bytes_recv < size) {
        ssize_t n = recv(sockfd, (uint8_t *)data + bytes_recv, size - bytes_recv, 0);
        if (n <= 0) {
            return false;
        }
        bytes_recv += n;
    }
    return true;
}

static bool send_rpc_cmd(int sockfd, enum rpc_cmd cmd, const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    uint8_t cmd_byte = cmd;
    if (!send_data(sockfd, &cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    uint64_t input_size = input.size();
    if (!send_data(sockfd, &input_size, sizeof(input_size))) {
        return false;
    }
    if (!send_data(sockfd, input.data(), input.size())) {
        return false;
    }
    uint64_t output_size;
    if (!recv_data(sockfd, &output_size, sizeof(output_size))) {
        return false;
    }
    if (output_size == 0) {
        output.clear();
        return true;
    }
    output.resize(output_size);
    if (!recv_data(sockfd, output.data(), output_size)) {
        return false;
    }
    return true;
}

// RPC client-side implementation

GGML_CALL static const char * ggml_backend_rpc_buffer_get_name(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    return ctx->name.c_str();
}

GGML_CALL static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // input serialization format: | remote_ptr (8 bytes) |
    std::vector<uint8_t> input(sizeof(uint64_t), 0);
    uint64_t remote_ptr = ctx->remote_ptr;
    memcpy(input.data(), &remote_ptr, sizeof(remote_ptr));
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(ctx->sockfd, FREE_BUFFER, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.empty());
    delete ctx;
}

GGML_CALL static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    static std::unordered_map<ggml_backend_buffer_t, void *> cache;
    if (cache.find(buffer) != cache.end()) {
        return cache[buffer];
    }
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // input serialization format: | remote_ptr (8 bytes) |
    std::vector<uint8_t> input(sizeof(uint64_t), 0);
    uint64_t remote_ptr = ctx->remote_ptr;
    memcpy(input.data(), &remote_ptr, sizeof(remote_ptr));
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(ctx->sockfd, BUFFER_GET_BASE, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.size() == sizeof(uint64_t));
    // output serialization format: | base_ptr (8 bytes) |
    uint64_t base_ptr;
    memcpy(&base_ptr, output.data(), sizeof(base_ptr));
    void * base = reinterpret_cast<void *>(base_ptr);
    cache[buffer] = base;
    return base;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor) {
    rpc_tensor result;
    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (tensor->buffer) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        result.buffer = ctx->remote_ptr;
    } else {
        result.buffer = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;
    result.data = reinterpret_cast<uint64_t>(tensor->data);
    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

static ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    snprintf(result->name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

GGML_CALL static void ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    UNUSED(buffer);
    if (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(tensor->ne[0] % 512 == 0 && "unsupported quantized tensor");
    }
}

GGML_CALL static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    int input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    std::vector<uint8_t> input(input_size, 0);
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    memcpy(input.data(), &rpc_tensor, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), data, size);
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(ctx->sockfd, SET_TENSOR, input, output);
    GGML_ASSERT(status);
}

GGML_CALL static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) |
    int input_size = sizeof(rpc_tensor) + 2*sizeof(uint64_t);
    std::vector<uint8_t> input(input_size, 0);
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    memcpy(input.data(), &rpc_tensor, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), &size, sizeof(size));
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(ctx->sockfd, GET_TENSOR, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.size() == size);
    // output serialization format: | data (size bytes) |
    memcpy(data, output.data(), size);
}

GGML_CALL static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    // check if src and dst are on the same server
    ggml_backend_buffer_t src_buffer = src->buffer;
    ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
    ggml_backend_buffer_t dst_buffer = dst->buffer;
    ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
    if (src_ctx->sockfd != dst_ctx->sockfd) {
        return false;
    }
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // input serialization format: | rpc_tensor src | rpc_tensor dst |
    int input_size = 2*sizeof(rpc_tensor);
    std::vector<uint8_t> input(input_size, 0);
    rpc_tensor rpc_src = serialize_tensor(src);
    rpc_tensor rpc_dst = serialize_tensor(dst);
    memcpy(input.data(), &rpc_src, sizeof(rpc_src));
    memcpy(input.data() + sizeof(rpc_src), &rpc_dst, sizeof(rpc_dst));
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(ctx->sockfd, COPY_TENSOR, input, output);
    GGML_ASSERT(status);
    // output serialization format: | result (1 byte) |
    GGML_ASSERT(output.size() == 1);
    return output[0];
}

GGML_CALL static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // serialization format: | bufptr (8 bytes) | value (1 byte) |
    int input_size = sizeof(uint64_t) + sizeof(uint8_t);
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &ctx->remote_ptr, sizeof(ctx->remote_ptr));
    memcpy(input.data() + sizeof(ctx->remote_ptr), &value, sizeof(value));
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(ctx->sockfd, BUFFER_CLEAR, input, output);
    GGML_ASSERT(status);
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .get_name        = */ ggml_backend_rpc_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

GGML_CALL static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    // input serialization format: | size (8 bytes) |
    int input_size = sizeof(uint64_t);
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &size, sizeof(size));
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(buft_ctx->sockfd, ALLOC_BUFFER, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.size() == 2*sizeof(uint64_t));
    // output serialization format: | remote_ptr (8 bytes) | remote_size (8 bytes) |
    uint64_t remote_ptr;
    memcpy(&remote_ptr, output.data(), sizeof(remote_ptr));
    size_t remote_size;
    memcpy(&remote_size, output.data() + sizeof(uint64_t), sizeof(remote_size));

    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
        ggml_backend_rpc_buffer_interface,
        new ggml_backend_rpc_buffer_context{buft_ctx->sockfd, remote_ptr, "RPC"},
        remote_size);

    return buffer;
}

GGML_CALL static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    // TODO: this is hardcoded for now but it should come from the remote backend
    return 32;
}

GGML_CALL static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    UNUSED(buft);
    return ggml_nbytes(tensor);
}

GGML_CALL static bool ggml_backend_rpc_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    if (!ggml_backend_is_rpc(backend)) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    return buft_ctx->sockfd == rpc_ctx->sockfd;
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .supports_backend = */ ggml_backend_rpc_buffer_type_supports_backend,
    /* .is_host          = */ NULL,
};


GGML_CALL static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

GGML_CALL static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)rpc_ctx->buft->context;
    //close(rpc_ctx->sockfd);
    delete buft_ctx;
    delete rpc_ctx->buft;
    delete rpc_ctx;
    delete backend;
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_rpc_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    return ctx->buft;
}

GGML_CALL static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    UNUSED(backend);
    // this is no-op because we don't have any async operations
}

static void add_tensor(ggml_tensor * tensor, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], tensors, visited);
    }
    add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(serialize_tensor(tensor));
}

static void serialize_graph(const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    int output_size = sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    output.resize(output_size, 0);
    memcpy(output.data(), &n_nodes, sizeof(n_nodes));
    uint64_t * out_nodes = (uint64_t *)(output.data() + sizeof(n_nodes));
    for (uint32_t i = 0; i < n_nodes; i++) {
        out_nodes[i] = reinterpret_cast<uint64_t>(cgraph->nodes[i]);
    }
    uint32_t * out_ntensors = (uint32_t *)(output.data() + sizeof(n_nodes) + n_nodes * sizeof(uint64_t));
    *out_ntensors = n_tensors;
    rpc_tensor * out_tensors = (rpc_tensor *)(output.data() + sizeof(n_nodes) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t));
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
}

GGML_CALL static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    std::vector<uint8_t> input;
    serialize_graph(cgraph, input);
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(rpc_ctx->sockfd, GRAPH_COMPUTE, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.size() == 1);
    return (enum ggml_status)output[0];
}

GGML_CALL static bool ggml_backend_rpc_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    UNUSED(backend);
    UNUSED(op);
    GGML_ASSERT(false && "not implemented");
    return false;
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .get_default_buffer_type = */ ggml_backend_rpc_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .supports_op             = */ ggml_backend_rpc_supports_op,
    /* .offload_op              = */ NULL,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

static std::vector<std::string> endpoints;

GGML_API GGML_CALL void ggml_rpc_init(const char * rpc_servers) {
    endpoints.clear();
    GGML_ASSERT(rpc_servers != NULL);
    std::string servers(rpc_servers);
    size_t pos = 0;
    while ((pos = servers.find(",")) != std::string::npos) {
        std::string server = servers.substr(0, pos);
        endpoints.push_back(server);
        servers.erase(0, pos + 1);
    }
    endpoints.push_back(servers);
}

static ggml_backend_t instances[GGML_RPC_MAX_SERVERS] = {0};

GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(int server_id) {
    ggml_backend_rpc_init(server_id);
    return ggml_backend_rpc_get_default_buffer_type(instances[server_id]);
}

GGML_CALL ggml_backend_t ggml_backend_rpc_init(int server_id) {
    if (server_id < 0 || server_id >= ggml_backend_rpc_get_server_count()) {
        return nullptr;
    }
    if (instances[server_id]) {
        return instances[server_id];
    }
    std::string endpoint = endpoints[server_id];
    GGML_PRINT_DEBUG("Connecting to %s\n", endpoint.c_str());
    // split the endpoint into host and port
    size_t pos = endpoint.find(":");
    std::string host = endpoint.substr(0, pos);
    int port = std::stoi(endpoint.substr(pos + 1));
    int sockfd = socket_connect(host.c_str(), port);
    GGML_ASSERT(sockfd >= 0 && "failed to connect to the server");

    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .sockfd = */ sockfd,
        /* .name   = */ "RPC" + std::to_string(server_id)
    };

    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .context = */ buft_ctx
    };

    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .endpoint = */ endpoint,
        /* .name     = */ "RPC",
        /* .sockfd   = */ sockfd,
        /* .buft     = */ buft
    };

    instances[server_id] = new ggml_backend {
        /* .guid      = */ ggml_backend_rpc_guid(),
        /* .interface = */ ggml_backend_rpc_interface,
        /* .context   = */ ctx
    };

    return instances[server_id];
}

GGML_API GGML_CALL bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

GGML_API GGML_CALL int ggml_backend_rpc_get_server_count(void) {
    return endpoints.size();
}

// RPC server-side implementation

static void rpc_alloc_buffer(ggml_backend_t backend, const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    // input serialization format: | size (8 bytes) |
    uint64_t size;
    memcpy(&size, input.data(), sizeof(size));
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    uint64_t remote_ptr = reinterpret_cast<uint64_t>(buffer);
    uint64_t remote_size = buffer->size;
    GGML_PRINT_DEBUG("[%s] size: %lu -> remote_ptr: %lx, remote_size: %lu\n", __func__, size, remote_ptr, remote_size);
    // output serialization format: | remote_ptr (8 bytes) | remote_size (8 bytes) |
    output.resize(2*sizeof(uint64_t), 0);
    memcpy(output.data(), &remote_ptr, sizeof(remote_ptr));
    memcpy(output.data() + sizeof(uint64_t), &remote_size, sizeof(remote_size));
}

static void rpc_buffer_get_base(const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    // input serialization format: | remote_ptr (8 bytes) |
    uint64_t remote_ptr;
    memcpy(&remote_ptr, input.data(), sizeof(remote_ptr));
    GGML_PRINT_DEBUG("[%s] remote_ptr: %lx\n", __func__, remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(remote_ptr);
    void * base = ggml_backend_buffer_get_base(buffer);
    // output serialization format: | base_ptr (8 bytes) |
    uint64_t base_ptr = reinterpret_cast<uint64_t>(base);
    output.resize(sizeof(uint64_t), 0);
    memcpy(output.data(), &base_ptr, sizeof(base_ptr));
}

static void rpc_free_buffer(const std::vector<uint8_t> & input) {
    // input serialization format: | remote_ptr (8 bytes) |
    uint64_t remote_ptr;
    memcpy(&remote_ptr, input.data(), sizeof(remote_ptr));
    GGML_PRINT_DEBUG("[%s] remote_ptr: %lx\n", __func__, remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(remote_ptr);
    ggml_backend_buffer_free(buffer);
}

static void rpc_buffer_clear(const std::vector<uint8_t> & input) {
    // input serialization format: | remote_ptr (8 bytes) | value (1 byte) |
    uint64_t remote_ptr;
    memcpy(&remote_ptr, input.data(), sizeof(remote_ptr));
    uint8_t value;
    memcpy(&value, input.data() + sizeof(uint64_t), sizeof(value));
    GGML_PRINT_DEBUG("[%s] remote_ptr: %lx, value: %u\n", __func__, remote_ptr, value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(remote_ptr);
    ggml_backend_buffer_clear(buffer, value);
}

static void rpc_set_tensor(const std::vector<uint8_t> & input) {
    // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    GGML_PRINT_DEBUG("[%s] buffer: %p, data: %p, offset: %lu, size: %lu\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);
    const void * data = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    ggml_backend_tensor_set(tensor, data, offset, size);
    ggml_free(ctx);
}

static void rpc_get_tensor(const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) |
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    uint64_t size;
    memcpy(&size, input.data() + sizeof(rpc_tensor) + sizeof(offset), sizeof(size));

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    GGML_PRINT_DEBUG("[%s] buffer: %p, data: %p, offset: %lu, size: %lu\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);
    // output serialization format: | data (size bytes) |
    output.resize(size, 0);
    ggml_backend_tensor_get(tensor, output.data(), offset, size);
    ggml_free(ctx);
}

static void rpc_copy_tensor(const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    // serialization format: | rpc_tensor src | rpc_tensor dst |
    const rpc_tensor * rpc_src = (const rpc_tensor *)input.data();
    const rpc_tensor * rpc_dst = (const rpc_tensor *)(input.data() + sizeof(rpc_src));

    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * src = deserialize_tensor(ctx, rpc_src);
    ggml_tensor * dst = deserialize_tensor(ctx, rpc_dst);
    GGML_PRINT_DEBUG("[%s] src->buffer: %p, dst->buffer: %p\n", __func__, (void*)src->buffer, (void*)dst->buffer);
    bool result = ggml_backend_buffer_copy_tensor(src, dst);
    // output serialization format: | result (1 byte) |
    output.resize(1, 0);
    output[0] = result;
    ggml_free(ctx);
}

static struct ggml_tensor * create_node(uint64_t id,
                                        struct ggml_context * ctx,
                                        const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                        std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (id == 0) {
        return nullptr;
    }
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    const rpc_tensor * tensor = tensor_ptrs.at(id);
    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    result->view_offs = tensor->view_offs;
    return result;
}

static void rpc_graph_compute(ggml_backend_t backend, const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_nodes;
    memcpy(&n_nodes, input.data(), sizeof(n_nodes));
    const uint64_t * nodes = (const uint64_t *)(input.data() + sizeof(n_nodes));
    uint32_t n_tensors;
    memcpy(&n_tensors, input.data() + sizeof(n_nodes) + n_nodes*sizeof(uint64_t), sizeof(n_tensors));
    const rpc_tensor * tensors = (const rpc_tensor *)(input.data() + sizeof(n_nodes) + n_nodes*sizeof(uint64_t) + sizeof(n_tensors));
    GGML_PRINT_DEBUG("[%s] n_nodes: %u, n_tensors: %u\n", __func__, n_nodes, n_tensors);

    static size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    for (uint32_t i = 0; i < n_nodes; i++) {
        graph->nodes[i] = create_node(nodes[i], ctx, tensor_ptrs, tensor_map);
    }
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    // output serialization format: | status (1 byte) |
    output.resize(1, 0);
    output[0] = status;
    ggml_free(ctx);
}

void rpc_serve_client(ggml_backend_t backend, int sockfd) {
    while (true) {
        uint8_t cmd;
        if (!recv_data(sockfd, &cmd, 1)) {
            break;
        }
        std::vector<uint8_t> input;
        std::vector<uint8_t> output;
        uint64_t input_size;
        if (!recv_data(sockfd, &input_size, sizeof(input_size))) {
            break;
        }
        input.resize(input_size);
        if (!recv_data(sockfd, input.data(), input_size)) {
            break;
        }
        switch (cmd) {
            case ALLOC_BUFFER: {
                rpc_alloc_buffer(backend, input, output);
                break;
            }
            case BUFFER_GET_BASE: {
                rpc_buffer_get_base(input, output);
                break;
            }
            case FREE_BUFFER: {
                rpc_free_buffer(input);
                break;
            }
            case BUFFER_CLEAR: {
                rpc_buffer_clear(input);
                break;
            }
            case SET_TENSOR: {
                rpc_set_tensor(input);
                break;
            }
            case GET_TENSOR: {
                rpc_get_tensor(input, output);
                break;
            }
            case COPY_TENSOR: {
                rpc_copy_tensor(input, output);
                break;
            }
            case GRAPH_COMPUTE: {
                rpc_graph_compute(backend, input, output);
                break;
            }
            default: {
                fprintf(stderr, "Unknown command: %d\n", cmd);
                break;
            }
        }
        uint64_t output_size = output.size();
        if (!send_data(sockfd, &output_size, sizeof(output_size))) {
            break;
        }
        if (!send_data(sockfd, output.data(), output_size)) {
            break;
        }
    }
}
