// llama-share-registry: unix-socket daemon that pins shared CUDA VMM weight segments
// so they outlive the process that exported them (wire format in src/llama-share.h).
//
// The daemon needs no CUDA: segments arrive as file descriptors received over the
// socket, and holding a descriptor is what anchors the underlying physical allocation.
// It keeps a (digest, bus) -> fd table; the first REGISTER for a (digest, bus) pair
// wins, duplicates are acknowledged and their fd closed, queries on the same bus are
// answered with a HIT carrying a fresh descriptor of the stored fd.

#include "llama-share.h"

#include <array>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <errno.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace {

// ---------------------------------------------------------------------------
// socket helpers for the wire format, one message per connection (kept here
// rather than in the library so the registry stays standalone and exports no
// symbols; kept in sync with src/llama-share.cpp on protocol changes)
// ---------------------------------------------------------------------------

// send one frame: hdr + trailing payload, optionally carrying fd via SCM_RIGHTS
bool send_frame(int fd, const llama_share_hdr & hdr, const std::string & trailing,
                bool have_fd, int fd_out) {
    std::vector<uint8_t> frame(sizeof(hdr) + trailing.size());
    memcpy(frame.data(), &hdr, sizeof(hdr));
    memcpy(frame.data() + sizeof(hdr), trailing.data(), trailing.size());

    iovec iov = { frame.data(), frame.size() };
    union {
        cmsghdr h;
        char buf[CMSG_SPACE(sizeof(int))];
    } cmsg_buf = {};
    msghdr msg = {};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    if (have_fd) {
        msg.msg_control = cmsg_buf.buf;
        msg.msg_controllen = sizeof(cmsg_buf);
        cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
        if (cmsg == nullptr) {
            return false;
        }
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type  = SCM_RIGHTS;
        cmsg->cmsg_len   = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(cmsg), &fd_out, sizeof(int));
    }
    while (true) {
        const ssize_t n = sendmsg(fd, &msg, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        return n == (ssize_t) frame.size();
    }
}

// receive one frame: hdr + trailing (name bytes, then bus bytes), optionally an fd
bool recv_frame(int fd, llama_share_hdr & hdr, std::string & name, std::string & bus, int * fd_in) {
    *fd_in = -1;

    const size_t payload_max = 4096; // name_len + bus_len sanity cap
    std::vector<uint8_t> frame(sizeof(hdr) + payload_max);
    size_t got = 0;
    bool have_hdr = false;

    while (true) {
        size_t want = sizeof(hdr);
        if (have_hdr) {
            const size_t payload_len = hdr.name_len + hdr.bus_len;
            if (payload_len > payload_max) {
                return false;
            }
            want = sizeof(hdr) + payload_len;
        }
        if (got >= want) {
            break;
        }

        iovec iov = { frame.data() + got, want - got };
        union {
            cmsghdr h;
            char buf[CMSG_SPACE(sizeof(int))];
        } cmsg_buf = {};
        msghdr msg = {};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsg_buf.buf;
        msg.msg_controllen = sizeof(cmsg_buf);

        const ssize_t n = recvmsg(fd, &msg, 0);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        got += n;
        for (cmsghdr * cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
            if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS &&
                    cmsg->cmsg_len >= CMSG_LEN(sizeof(int))) {
                int recvd = -1;
                memcpy(&recvd, CMSG_DATA(cmsg), sizeof(int));
                if (*fd_in < 0) {
                    *fd_in = recvd;
                } else {
                    close(recvd); // never expected; don't leak
                }
            }
        }
        if (!have_hdr && got >= sizeof(hdr)) {
            memcpy(&hdr, frame.data(), sizeof(hdr));
            have_hdr = true;
        }
    }

    const size_t trailing_len = hdr.name_len + hdr.bus_len;
    if (trailing_len > payload_max) {
        return false;
    }
    std::string trailing((const char *) frame.data() + sizeof(hdr), trailing_len);
    name.assign(trailing, 0, hdr.name_len);
    bus.assign(trailing, hdr.name_len, hdr.bus_len);
    return true;
}

// ---------------------------------------------------------------------------
// registry state
// ---------------------------------------------------------------------------

struct entry {
    int         fd = -1;
    uint64_t    seg_size = 0;
    std::string name;
};

std::map<std::array<uint8_t, 32>, std::map<std::string, entry>> table;

std::array<uint8_t, 32> digest_key(const llama_share_hdr & hdr) {
    std::array<uint8_t, 32> key;
    memcpy(key.data(), hdr.digest, key.size());
    return key;
}

std::string digest_to_hex(const std::array<uint8_t, 32> & digest) {
    static const char * hex = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (size_t i = 0; i < digest.size(); i++) {
        out += hex[digest[i] >> 4];
        out += hex[digest[i] & 0xf];
    }
    return out;
}

volatile sig_atomic_t g_stop = 0;

void sig_handler(int) {
    g_stop = 1;
}

size_t pinned_segments() {
    size_t pinned = 0;
    for (const auto & kv : table) {
        pinned += kv.second.size();
    }
    return pinned;
}

// replies echo the request type so the client can validate the exchange
llama_share_hdr reply_hdr(uint32_t type, llama_share_msg_status status, uint32_t bus_len = 0) {
    llama_share_hdr hdr = {};
    hdr.magic   = LLAMA_SHARE_MAGIC;
    hdr.version = LLAMA_SHARE_VERSION;
    hdr.type    = type;
    hdr.status  = status;
    hdr.bus_len = bus_len;
    return hdr;
}

void reply_err(int fd, uint32_t type = 0) {
    const llama_share_hdr rep = reply_hdr(type, LLAMA_SHARE_STATUS_ERR);
    send_frame(fd, rep, "", false, -1);
}

// one message per connection
void handle_conn(int fd) {
    llama_share_hdr hdr = {};
    std::string name;
    std::string bus;
    int fd_in = -1;
    if (!recv_frame(fd, hdr, name, bus, &fd_in)) {
        if (fd_in >= 0) {
            close(fd_in);
        }
        return;
    }

    if (hdr.magic != LLAMA_SHARE_MAGIC || hdr.version != LLAMA_SHARE_VERSION) {
        if (fd_in >= 0) {
            close(fd_in);
        }
        reply_err(fd, hdr.type);
        return;
    }

    switch (hdr.type) {
        case LLAMA_SHARE_QUERY:
            {
                if (fd_in >= 0) {
                    close(fd_in); // queries carry no fd; don't leak if one arrives
                }
                const auto it_digest = table.find(digest_key(hdr));
                if (it_digest == table.end()) {
                    const llama_share_hdr rep = reply_hdr(hdr.type, LLAMA_SHARE_STATUS_MISS);
                    send_frame(fd, rep, "", false, -1);
                    break;
                }
                const auto it_bus = it_digest->second.find(bus);
                if (it_bus == it_digest->second.end()) {
                    const llama_share_hdr rep = reply_hdr(hdr.type, LLAMA_SHARE_STATUS_MISS);
                    send_frame(fd, rep, "", false, -1);
                    break;
                }
                const entry & e = it_bus->second;
                llama_share_hdr rep = {};
                rep.magic    = LLAMA_SHARE_MAGIC;
                rep.version  = LLAMA_SHARE_VERSION;
                rep.type     = hdr.type;
                rep.status   = LLAMA_SHARE_STATUS_HIT;
                rep.bus_len  = bus.size();
                rep.seg_size = e.seg_size;
                memcpy(rep.digest, hdr.digest, 32);
                send_frame(fd, rep, bus, true, e.fd);
                break;
            }
        case LLAMA_SHARE_REGISTER:
            {
                if (fd_in < 0) {
                    reply_err(fd, hdr.type);
                    break;
                }
                const auto it_bus = table[digest_key(hdr)].find(bus);
                if (it_bus != table[digest_key(hdr)].end()) {
                    // keep-first: the stored fd is the live anchor; close the duplicate
                    close(fd_in);
                    const llama_share_hdr rep = reply_hdr(hdr.type, LLAMA_SHARE_STATUS_OK);
                    send_frame(fd, rep, "", false, -1);
                    break;
                }
                entry e;
                e.fd       = fd_in;
                e.seg_size = hdr.seg_size;
                e.name     = name;
                table[digest_key(hdr)][bus] = std::move(e);

                printf("llama-share-registry: registered '%s' digest %s (bus %s, %llu bytes) - %zu segments pinned\n",
                        name.c_str(), digest_to_hex(digest_key(hdr)).c_str(), bus.c_str(),
                        (unsigned long long) hdr.seg_size, pinned_segments());
                fflush(stdout);

                const llama_share_hdr rep = reply_hdr(hdr.type, LLAMA_SHARE_STATUS_OK);
                send_frame(fd, rep, "", false, -1);
                break;
            }
        default:
            if (fd_in >= 0) {
                close(fd_in);
            }
            reply_err(fd, hdr.type);
            break;
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <unix-socket-path>\n", argv[0]);
        return 1;
    }
    const std::string socket_path = argv[1];

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = sig_handler;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    if (socket_path.size() >= sizeof(addr.sun_path)) {
        fprintf(stderr, "%s: socket path too long: %s\n", argv[0], socket_path.c_str());
        return 1;
    }
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    const int sfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sfd < 0) {
        fprintf(stderr, "%s: socket() failed: %s\n", argv[0], strerror(errno));
        return 1;
    }
    if (bind(sfd, (const sockaddr *) &addr, sizeof(addr)) != 0) {
        fprintf(stderr, "%s: bind(%s) failed: %s (running already? stale socket?)\n",
                argv[0], socket_path.c_str(), strerror(errno));
        return 1;
    }
    if (listen(sfd, 16) != 0) {
        fprintf(stderr, "%s: listen() failed: %s\n", argv[0], strerror(errno));
        return 1;
    }

    // permissions: the registry serves shared model weights; keep it owner-only
    chmod(socket_path.c_str(), 0600);

    printf("llama-share-registry: listening on %s\n", socket_path.c_str());
    fflush(stdout);

    while (!g_stop) {
        const int cfd = accept(sfd, nullptr, nullptr);
        if (cfd < 0) {
            if (errno == EINTR) {
                break;
            }
            continue;
        }
        handle_conn(cfd); // one message per connection
        close(cfd);
    }

    close(sfd);
    unlink(socket_path.c_str());

    printf("llama-share-registry: shut down, %zu segments pinned at exit\n", pinned_segments());
    return 0;
}