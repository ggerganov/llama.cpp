#include "tcp_server.h"

#include <iostream>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>

#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

class PosixStream : public std::istream {
    public:
        PosixStream(int fd) : std::istream(&buf), buf(fd) {}
        ~PosixStream() { close(buf.get_fd()); }

    private:
        class PosixStreamBuf : public std::streambuf {
            public:
                PosixStreamBuf(int fd) : fd(fd) {}
                int get_fd() const { return fd; }

            protected:
                virtual int_type underflow() {
                    if (gptr() < egptr()) {
                        return traits_type::to_int_type(*gptr());
                    }

                    ssize_t num_read = ::read(fd, buffer, BUFFER_SIZE);
                    if (num_read <= 0) {
                        return traits_type::eof();
                    }

                    setg(buffer, buffer, buffer + num_read);
                    return traits_type::to_int_type(*gptr());
                }

            private:
                static const int BUFFER_SIZE = 1024;
                int fd;
                char buffer[BUFFER_SIZE];
        };

        PosixStreamBuf buf;
};

void die(const char *msg, ...)
{
    va_list ap;

    va_start(ap, msg);
    vfprintf(stderr, msg, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

static char *read_argument(uint8_t **param_buf, size_t *param_buf_size, FILE *instream) {
    bool done = false;
    uint8_t *buf = *param_buf;
    size_t bufsize = *param_buf_size;
    size_t bufpos = 0;
    while (!done) {
        if (bufpos == bufsize) {
            bufsize += 1024;
            buf = (uint8_t *)realloc(buf, bufsize);
            if (!buf) {
                die("failed to allocate memory");
            }
        }

        int c = fgetc(instream);
        if (c == EOF) {
            die("unexpected EOF client socket");
        }
        buf[bufpos++] = (uint8_t)c;
        if (c == 0) {
            // done reading argument
            break;
        }
    }
    *param_buf = buf;
    *param_buf_size = bufsize;
    return strdup((char *)buf);
}

static int read_arguments(int argc, char **argv, FILE *instream) {
    int i = 1;
    size_t param_buf_size = 0;
    uint8_t *param_buf = nullptr;

    for (i = 1; i < argc; i++) {
        argv[i] = read_argument(&param_buf, &param_buf_size, instream);
    }

    free(param_buf);
    return i;
}

static int serve_model(
        gpt_params params,
        gpt_vocab vocab,
        llama_model model,
        int64_t t_load_us,
        int64_t t_main_start_us,
        int sock_fd)
{
    char *response_data;
    int argc;
    char **argv;
    FILE *instream = fdopen(sock_fd, "r");
    FILE *outstream = fdopen(sock_fd, "w");
    setvbuf(instream, NULL, _IONBF, 0);

    // start by reading the parameter count
    if (fscanf(instream, "%d\n", &argc) != 1) {
        fprintf(outstream, "Error: First line must be character count\n");
        fflush(outstream);
        return 1;
    }

    argc += 1;  // add one extra argument to emulate the program command line
    argv = (char **)malloc(argc * sizeof *argv);
    argv[0] = nullptr;
    if (read_arguments(argc, argv, instream) != argc) {
        fprintf(outstream, "Error: Failed to read arguments\n");
        fflush(outstream);
    }

    if (gpt_params_parse(argc, argv, params) == false) {
        fprintf(outstream, "Error: Failed to parse parameters\n");
        fflush(outstream);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        free(argv[i]);
    }
    free(argv);

    PosixStream tcp_is(sock_fd);

    return llama_main(params, vocab, model, t_load_us, t_main_start_us, tcp_is, outstream, outstream);
}

int listen_tcp(
        gpt_params params,
        gpt_vocab vocab,
        llama_model model,
        int64_t t_main_start_us,
        int64_t t_load_us) {
    int listen_fd;
    int status;
    pid_t child;
    struct addrinfo hints;
    struct addrinfo *servinfo, *p;
    int yes = 1;

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    // This should only ever listen on a loopback address. Access from outside
    // should be proxied via nginx or similar software
    status = getaddrinfo("127.0.0.1", params.listen_port.c_str(), &hints, &servinfo);
    if (status) {
        die("getaddrinfo error: %s", gai_strerror(status));
    }

    // bind to the first addrinfo we can from the getaddrinfo results
    for (p = servinfo; p != NULL; p = p->ai_next) {
        listen_fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (listen_fd == -1) {
            perror("server: socket");
            continue;
        }

        if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes)) {
            die("setsockopt error: %s", params.listen_port.c_str(), strerror(errno));
        }

        if (bind(listen_fd, p->ai_addr, p->ai_addrlen) == 0) {
            break;
        }

        close(listen_fd);
        perror("server: bind");
    }

    freeaddrinfo(servinfo);

    if (p == NULL) {
        die("failed to bind: %s", strerror(errno));
    }

    if (listen(listen_fd, 20)) {
        die("listen error: %s", strerror(errno));
    }
    // Don't track child processes, so ignore SIGCHLD to prevent zombies
    signal(SIGCHLD, SIG_IGN);

    for (;;) {
        struct sockaddr_in client_addr = {0};
        socklen_t client_addr_len = 0;

        int sock_fd = accept(listen_fd,
                (struct sockaddr *)&client_addr,
                &client_addr_len);
        if (sock_fd < 0) {
            fprintf(stderr, "accept error: %s\n", strerror(errno));
            break;
        }

        child = fork();
        if (child == 0) {
            // close the listen_fd since we won't use it in the child
            close(listen_fd);
            int ret = serve_model(params, vocab, model, t_main_start_us, t_load_us, sock_fd);
            close(sock_fd);
            return ret;
        } else {
            // close the client since we won't use it in the server
            close(sock_fd);
            sock_fd = 0;
        }
    }
    close(listen_fd);

    // ignore SIGTERM since we'll send it to the group
    signal(SIGTERM, SIG_IGN);
    // tell children to exit
    kill(0, SIGTERM);
    // wait for children to terminate
    wait(&status);
    return 0;
}
