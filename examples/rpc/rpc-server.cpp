#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "ggml-rpc.h"
#include <memory>
#include <string>
#ifndef _WIN32
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif
#include <stdio.h>
#include <stdlib.h>

static ggml_backend_t create_backend() {
    ggml_backend_t backend = NULL;
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    backend = ggml_backend_cuda_init(0); // init device 0
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#elif GGML_USE_METAL
    fprintf(stderr, "%s: using Metal backend\n", __func__);
    backend = ggml_backend_metal_init();
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
    }
#endif

    // if there aren't GPU Backends fallback to CPU backend
    if (!backend) {
        fprintf(stderr, "%s: using CPU backend\n", __func__);
        backend = ggml_backend_cpu_init();
    }
    return backend;
}

static void get_backend_memory(size_t * free_mem, size_t * total_mem) {
#ifdef GGML_USE_CUDA
    ggml_backend_cuda_get_device_memory(0, free_mem, total_mem);
#else
    // TODO: implement for other backends
    *free_mem = 1;
    *total_mem = 1;
#endif
}

static std::shared_ptr<socket_t> make_socket(sockfd_t fd) {
#ifdef _WIN32
    if (fd == INVALID_SOCKET) {
        return nullptr;
    }
#else
    if (fd < 0) {
        return nullptr;
    }
#endif
    return std::make_shared<socket_t>(fd);
}

static std::shared_ptr<socket_t> create_server_socket(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    auto sock = make_socket(sockfd);
    if (sock == nullptr) {
        return nullptr;
    }

    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        return nullptr;
    }
    if (listen(sockfd, 5) < 0) {
        return nullptr;
    }
    return sock;
}

int main(int argc, char * argv[])
{
#ifdef _WIN32
    WSADATA wsaData;
    int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (res != 0) {
        fprintf(stderr, "WSAStartup failed: %d\n", res);
        return 1;
    }
#endif
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <host> <port>\n", argv[0]);
        return 1;
    }
    const char * host = argv[1];
    int port = std::stoi(argv[2]);

    ggml_backend_t backend = create_backend();
    if (!backend) {
        fprintf(stderr, "Failed to create backend\n");
        return 1;
    }

    printf("Starting RPC server on %s:%d\n", host, port);
    auto server_socket = create_server_socket(host, port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return 1;
    }
    while (true) {
        auto client_socket_fd = accept(server_socket->fd, NULL, NULL);
        auto client_socket = make_socket(client_socket_fd);
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return 1;
        }
        // set TCP_NODELAY to disable Nagle's algorithm
        int flag = 1;
        int ret = setsockopt(client_socket->fd, IPPROTO_TCP, TCP_NODELAY, (char *) &flag, sizeof(int));
        if (ret < 0) {
            fprintf(stderr, "Failed to set TCP_NODELAY\n");
            continue;
        }
        size_t free_mem, total_mem;
        get_backend_memory(&free_mem, &total_mem);
        printf("Accepted client connection, free_mem=%zu, total_mem=%zu\n", free_mem, total_mem);
        rpc_serve_client(backend, client_socket->fd, free_mem, total_mem);
        printf("Client connection closed\n");
    }
#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
