#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "ggml-rpc.h"
#ifdef _WIN32
#  include <windows.h>
#else
#  include <unistd.h>
#endif
#include <ctime>
#include <string>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <map>
#include <fstream>
#include <chrono>
#include <iomanip>

class Registry {
public:
    Registry(const std::string& server_ip, int port) : server_ip(server_ip), port(port) {
        log("Registry initialized");
    }

    void add_field(const std::string& key, const std::string& value) {
        payload[key] = value;
        log("Added field: " + key + " = " + value);
    }

    bool register_with_central(const std::string& endpoint) {
        payload["endpoint"] = endpoint;
        payload["status"] = "unknown";
        payload["last_checked"] = "unknown";
        std::string json_payload = create_json_payload();
        log("Attempting to register with payload: " + json_payload);
        return send_request(json_payload);
    }

private:
    std::string server_ip;
    int port;
    std::map<std::string, std::string> payload;

    std::string create_json_payload() const {
        std::ostringstream json;
        json << "{";
        for (auto it = payload.begin(); it != payload.end(); ++it) {
            json << "\"" << it->first << "\":\"" << it->second << "\"";
            if (std::next(it) != payload.end()) {
                json << ",";
            }
        }
        json << "}";
        return json.str();
    }

    std::string build_http_request(const std::string& data) const {
        std::ostringstream request;
        request << "POST /register HTTP/1.1\r\n";
        request << "Host: " << server_ip << "\r\n";
        request << "Content-Type: application/json\r\n";
        request << "Content-Length: " << data.length() << "\r\n";
        request << "Connection: close\r\n\r\n";
        request << data;
        return request.str();
    }

    bool send_request(const std::string& data) const {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            log("Socket creation error");
            return false;
        }

        struct sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(port);

        if (inet_pton(AF_INET, server_ip.c_str(), &serv_addr.sin_addr) <= 0) {
            log("Invalid address / Address not supported");
            return false;
        }

        if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
            log("Connection failed");
            return false;
        }

        std::string request = build_http_request(data);
        send(sock, request.c_str(), request.length(), 0);
        log("Sent request: " + request);

        char buffer[1024] = {0};
        read(sock, buffer, 1024);
        close(sock);

        log("Received response: " + std::string(buffer));
        return true;
    }

    void log(const std::string& message) const {
        std::ofstream log_file("registry_log.txt", std::ios_base::app);
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        log_file << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S") << " - " << message << std::endl;
    }
};

void print_colored(const std::string& text, const std::string& color_code) {
    std::cout << "\033[" << color_code << "m" << text << "\033[0m" << std::endl;
}

struct rpc_server_params {
    std::string host        = "0.0.0.0";
    int         port        = 50052;
    size_t      backend_mem = 0;
};

static void print_usage(int /*argc*/, char ** argv, rpc_server_params params) {
    fprintf(stderr, "Usage: %s [options]\n\n", argv[0]);
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -H HOST, --host HOST  host to bind to (default: %s)\n", params.host.c_str());
    fprintf(stderr, "  -p PORT, --port PORT  port to bind to (default: %d)\n", params.port);
    fprintf(stderr, "  -m MEM, --mem MEM     backend memory size (in MB)\n");
    fprintf(stderr, "\n");
}

static bool rpc_server_params_parse(int argc, char ** argv, rpc_server_params & params) {
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg == "-H" || arg == "--host") {
            if (++i >= argc) {
                return false;
            }
            params.host = argv[i];
        } else if (arg == "-p" || arg == "--port") {
            if (++i >= argc) {
                return false;
            }
            params.port = std::stoi(argv[i]);
            if (params.port <= 0 || params.port > 65535) {
                return false;
            }
        } else if (arg == "-m" || arg == "--mem") {
            if (++i >= argc) {
                return false;
            }
            params.backend_mem = std::stoul(argv[i]) * 1024 * 1024;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params);
            exit(0);
        }
    }
    return true;
}

void printAntigmaLogo() {
    std::cout << R"(

                                                                           
   _|_|    _|      _|  _|_|_|_|_|  _|_|_|    _|_|_|  _|      _|    _|_|    
 _|    _|  _|_|    _|      _|        _|    _|        _|_|  _|_|  _|    _|  
 _|_|_|_|  _|  _|  _|      _|        _|    _|  _|_|  _|  _|  _|  _|_|_|_|  
 _|    _|  _|    _|_|      _|        _|    _|    _|  _|      _|  _|    _|  
 _|    _|  _|      _|      _|      _|_|_|    _|_|_|  _|      _|  _|    _|  
                                                                           
                                                                           
                                                   
    )" << '\n';
}

static ggml_backend_t create_backend() {
    ggml_backend_t backend = NULL;
    printAntigmaLogo();
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
    #ifdef _WIN32
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        *total_mem = status.ullTotalPhys;
        *free_mem = status.ullAvailPhys;
    #else
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        *total_mem = pages * page_size;
        *free_mem = *total_mem;
    #endif
#endif
}

int main(int argc, char * argv[]) {
    rpc_server_params params;
    Registry registry("0.0.0.0", 5000);

    if (!rpc_server_params_parse(argc, argv, params)) {
        fprintf(stderr, "Invalid parameters\n");
        return 1;
    }
    ggml_backend_t backend = create_backend();
    if (!backend) {
        fprintf(stderr, "Failed to create backend\n");
        return 1;
    }
    std::string endpoint = params.host + ":" + std::to_string(params.port);

    size_t free_mem, total_mem;
    if (params.backend_mem > 0) {
        free_mem = params.backend_mem;
        total_mem = params.backend_mem;
    } else {
        get_backend_memory(&free_mem, &total_mem);
    }
    printf("\nStarting Antigma node on %s, backend memory: %zu MB\n", endpoint.c_str(), free_mem / (1024 * 1024));
    if (registry.register_with_central(endpoint)) {
        print_colored("Registered successfully", "32");
    } else {
        print_colored("Registered successfully", "31");
    }
    start_rpc_server(backend, endpoint.c_str(), free_mem, total_mem);
    ggml_backend_free(backend);
    return 0;
}
