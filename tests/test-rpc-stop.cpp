#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

// Regression test: the rpc server must stop even when a client is connected and idle
//
// the server serves one client at a time and parks in a blocking read while a client
// connection is open, a stop that only breaks the listening socket leaves the server
// thread parked, the server never returns and the process cannot exit
//
// the test starts the server in a thread, connects an idle client, requests a stop
// and asserts the server returns within a bound

static std::mutex              g_stop_mutex;
static std::condition_variable g_stop_cv;
static bool                    g_stopped = false;

int main(void) {
    ggml_backend_load_all();

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu != nullptr);

    std::thread server_thread([cpu]() mutable {
        ggml_backend_rpc_start_server("127.0.0.1:18287", nullptr, 2, 1, &cpu);
        {
            std::lock_guard<std::mutex> lock(g_stop_mutex);
            g_stopped = true;
        }
        g_stop_cv.notify_all();
    });

    // wait for the listener, a raw client that sends nothing parks the server in its
    // first blocking read, this is the state that used to hang the stop
    int client_fd = -1;
    for (int i = 0; i < 200 && client_fd < 0; ++i) {
        const int fd = socket(AF_INET, SOCK_STREAM, 0);
        GGML_ASSERT(fd >= 0);
        struct sockaddr_in addr;
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = inet_addr("127.0.0.1");
        addr.sin_port        = htons(18287);
        if (connect(fd, (struct sockaddr *) &addr, sizeof(addr)) == 0) {
            client_fd = fd;
        } else {
            close(fd);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    GGML_ASSERT(client_fd >= 0);

    ggml_backend_rpc_request_stop();

    {
        std::unique_lock<std::mutex> lock(g_stop_mutex);
        const bool stopped = g_stop_cv.wait_for(lock, std::chrono::seconds(10), []() { return g_stopped; });
        GGML_ASSERT(stopped);
    }

    server_thread.join();
    close(client_fd);

    printf("ok\n");

    return 0;
}
