#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "ggml-rpc.h"
#include <memory>
#include <string>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static ggml_backend_t create_backend() {
    ggml_backend_t backend = NULL;
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    backend = ggml_backend_cuda_init(0); // init device 0
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif

#ifdef GGML_USE_METAL
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

static int create_server_socket(const char * host, int port) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        return -1;
    }

    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        return -1;
    }
    if (listen(sockfd, 5) < 0) {
        return -1;
    }
    return sockfd;
}

int main(int argc, char * argv[])
{
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
    int server_socket = create_server_socket(host, port);
    if (server_socket < 0) {
        fprintf(stderr, "Failed to create server socket\n");
        return 1;
    }
    while (true) {
        struct sockaddr_in cli_addr;
        socklen_t clilen = sizeof(cli_addr);
        int client_socket = accept(server_socket, (struct sockaddr *) &cli_addr, &clilen);
        if (client_socket < 0) {
            fprintf(stderr, "Failed to accept client connection\n");
            return 1;
        }
        printf("Accepted client connection\n");
        rpc_serve_client(backend, client_socket);
        printf("Client connection closed\n");
        close(client_socket);
    }
    return 0;
}
