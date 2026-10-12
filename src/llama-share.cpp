#include "llama-share.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <inttypes.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llama-impl.h"

//
// llama_share - cross-process sharing of weight tensors over CUDA VMM segments
//
// see the header comment in llama-share.h for the symmetric load algorithm
//

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4)
// ---------------------------------------------------------------------------

namespace llama_share_detail {

struct sha256 {
    uint32_t h[8];
    uint64_t nbytes = 0;
    uint8_t  pending[64] = {};
    size_t   pending_len = 0;

    sha256() {
        static const uint32_t iv[8] = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
        };
        memcpy(h, iv, sizeof(h));
    }

    static uint32_t ror(uint32_t x, int n) {
        return (x >> n) | (x << (32 - n));
    }

    void block(const uint8_t * p) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        };

        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t) p[4*i] << 24) | ((uint32_t) p[4*i + 1] << 16) | ((uint32_t) p[4*i + 2] << 8) | (uint32_t) p[4*i + 3];
        }
        for (int i = 16; i < 64; i++) {
            const uint32_t s0 = ror(w[i-15], 7) ^ ror(w[i-15], 18) ^ (w[i-15] >> 3);
            const uint32_t s1 = ror(w[i-2], 17) ^ ror(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }

        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            const uint32_t s1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + s1 + ch + k[i] + w[i];
            const uint32_t s0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + mj;
            hh = g; g = f; f = e; e = d + t1;
            d = c;  c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update(const void * data, size_t len) {
        const uint8_t * p = (const uint8_t *) data;
        nbytes += len;
        while (len > 0) {
            if (pending_len == sizeof(pending)) {
                block(pending);
                pending_len = 0;
            }
            const size_t take = std::min(len, sizeof(pending) - pending_len);
            memcpy(pending + pending_len, p, take);
            pending_len += take;
            p += take;
            len -= take;
        }
    }

    std::array<uint8_t, 32> final() {
        const uint64_t bit_len = nbytes * 8;
        const uint8_t end_of_data = 0x80;
        update(&end_of_data, 1);
        const uint8_t zero = 0;
        while (pending_len != 56) {
            update(&zero, 1);
        }
        uint8_t len_bytes[8];
        for (int i = 0; i < 8; i++) {
            len_bytes[i] = (uint8_t) (bit_len >> (56 - 8*i));
        }
        memcpy(pending + 56, len_bytes, 8);
        block(pending);
        pending_len = 0;

        std::array<uint8_t, 32> out;
        for (int i = 0; i < 8; i++) {
            out[4*i     ] = (uint8_t) (h[i] >> 24);
            out[4*i + 1] = (uint8_t) (h[i] >> 16);
            out[4*i + 2] = (uint8_t) (h[i] >> 8);
            out[4*i + 3] = (uint8_t) (h[i]);
        }
        return out;
    }
};

// ---------------------------------------------------------------------------
// unix-socket helpers (one message per connection)
// ---------------------------------------------------------------------------

int unix_connect(const std::string & path) {
    sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        return -1;
    }
    strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    if (connect(fd, (const sockaddr *) &addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

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

    std::string trailing((const char *) frame.data() + sizeof(hdr), hdr.name_len + hdr.bus_len);
    name.assign(trailing, 0, hdr.name_len);
    bus.assign(trailing, hdr.name_len, hdr.bus_len);
    return true;
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

bool manifest_stat_state(const std::vector<std::string> & gguf_paths,
                         uint64_t & total_bytes, uint64_t & max_mtime) {
    total_bytes = 0;
    max_mtime = 0;
    for (const std::string & path : gguf_paths) {
        struct stat st;
        if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
            return false;
        }
        total_bytes += (uint64_t) st.st_size;
        max_mtime = std::max(max_mtime, (uint64_t) st.st_mtime);
    }
    return true;
}

} // namespace llama_share_detail

using namespace llama_share_detail;

// ---------------------------------------------------------------------------
// llama_share::impl
// ---------------------------------------------------------------------------

struct llama_share::impl {
    // proc-addressed CUDA backend functions; all of them must resolve or sharing
    // is disabled entirely (no hard link to the CUDA backend from the loader)
    using is_shareable_fn_t   = bool (*)(ggml_backend_buffer_t);
    using export_fn_t         = int (*)(ggml_backend_buffer_t, int *, size_t *);
    using is_attached_fn_t    = bool (*)(ggml_backend_buffer_t);
    using attach_spec_fn_t    = bool (*)(ggml_backend_buffer_type_t, const char *, int, size_t);
    using bus_id_fn_t         = const char * (*)(int, char *, size_t);
    using shareable_buft_fn_t = ggml_backend_buffer_type_t (*)(int);

    is_shareable_fn_t   fn_is_shareable   = nullptr;
    export_fn_t         fn_export         = nullptr;
    is_attached_fn_t    fn_is_attached    = nullptr;
    attach_spec_fn_t    fn_attach_spec    = nullptr;
    bus_id_fn_t         fn_bus_id         = nullptr;
    shareable_buft_fn_t fn_shareable_buft = nullptr;

    bool enabled = false;

    std::string ipc_path;
    std::string manifest_path;
    std::vector<std::string> gguf_paths;

    // shareable buffer types handed out, mapped to the bus id of their device
    std::map<ggml_backend_buffer_type_t, std::string> buft_bus;

    // manifest read from manifest_path, only used when fresh
    struct manifest_entry {
        std::array<uint8_t, 32> digest;
        size_t nbytes;
    };
    std::unordered_map<std::string, manifest_entry> manifest;
    bool manifest_fresh = false;

    // name -> entry observed this run; written at finish()
    std::unordered_map<std::string, manifest_entry> new_entries;

    // per-tensor digest state between begin_tensor and end_tensor
    bool hashing = false;
    sha256 hasher;

    uint64_t n_attached = 0;
    uint64_t n_registered = 0;
    uint64_t n_skipped_bytes = 0;

    void manifest_load() {
        if (gguf_paths.empty()) {
            LLAMA_LOG_WARN("%s: no weight files given; ignoring manifest %s (freshness cannot be checked)\n",
                           __func__, manifest_path.c_str());
            return;
        }

        FILE * f = fopen(manifest_path.c_str(), "r");
        if (f == nullptr) {
            return; // no manifest yet: nothing to share, one gets written at finish()
        }

        std::vector<std::string> lines;
        {
            char buf[1024];
            bool first = true;
            while (fgets(buf, sizeof(buf), f) != nullptr) {
                if (first && strncmp(buf, "#gguf-share-manifest", 20) != 0) {
                    LLAMA_LOG_WARN("%s: %s is not a share manifest; ignoring it\n", __func__, manifest_path.c_str());
                    fclose(f);
                    return;
                }
                if (buf[0] != '#') {
                    std::string line(buf);
                    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
                        line.pop_back();
                    }
                    lines.push_back(std::move(line));
                }
                first = false;
            }
            fclose(f);
        }

        std::unordered_map<std::string, manifest_entry> parsed;
        bool parse_ok = true;
        for (const std::string & line : lines) {
            const size_t t1 = line.find('\t');
            const size_t t2 = t1 == std::string::npos ? std::string::npos : line.find('\t', t1 + 1);
            if (t1 == std::string::npos || t2 == std::string::npos) {
                parse_ok = false;
                break;
            }
            const std::string name = line.substr(0, t1);
            const std::string hex = line.substr(t1 + 1, t2 - t1 - 1);
            if (name.empty() || hex.size() != 64) {
                parse_ok = false;
                break;
            }
            manifest_entry e;
            e.nbytes = (size_t) strtoull(line.c_str() + t1 + 1 + 65, nullptr, 10);
            for (int i = 0; i < 32; i++) {
                const char hi = hex[2*i];
                const char lo = hex[2*i + 1];
                auto nib = [](char c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                    return -1;
                };
                const int h = nib(hi);
                const int l = nib(lo);
                if (h < 0 || l < 0) {
                    parse_ok = false;
                    break;
                }
                e.digest[i] = (uint8_t) ((h << 4) | l);
            }
            if (!parse_ok || e.nbytes == 0) {
                parse_ok = false;
                break;
            }
            parsed[name] = std::move(e);
        }

        if (!parse_ok || parsed.empty()) {
            LLAMA_LOG_WARN("%s: %s is truncated; will rewrite it at the end of the load\n",
                           __func__, manifest_path.c_str());
            return;
        }

        uint64_t total_bytes = 0;
        uint64_t max_mtime = 0;
        if (!manifest_stat_state(gguf_paths, total_bytes, max_mtime)) {
            LLAMA_LOG_WARN("%s: model weight files are not all readable; sharing is disabled for this load\n",
                           __func__);
            return;
        }

        uint64_t m_total = 0;
        uint64_t m_mtime = 0;
        {
            // header line was validated above; pull the two numbers back out of the file
            FILE * f = fopen(manifest_path.c_str(), "r");
            if (f == nullptr || fscanf(f, "#gguf-share-manifest\t%" SCNu64 "\t%" SCNu64 "\n", &m_total, &m_mtime) != 2) {
                if (f != nullptr) {
                    fclose(f);
                }
                LLAMA_LOG_WARN("%s: %s has no readable header; ignoring it\n", __func__, manifest_path.c_str());
                return;
            }
            fclose(f);
        }

        if (m_total != total_bytes || m_mtime != max_mtime) {
            LLAMA_LOG_INFO("%s: manifest is stale (weight files changed); ignoring it and republishing\n",
                           __func__);
            return;
        }

        manifest = std::move(parsed);
        manifest_fresh = true;
        LLAMA_LOG_DEBUG("%s: loaded manifest %s with %zu entries\n", __func__, manifest_path.c_str(), manifest.size());
    }

    // full registry round trip: connect, send request (optionally with an fd),
    // read the reply; returns false on any transport-level problem
    bool registry_roundtrip(llama_share_hdr & req, const std::string & trailing,
                            int req_fd, llama_share_hdr & rep, std::string & rep_name,
                            std::string & rep_bus, int * rep_fd) {
        const int sfd = unix_connect(ipc_path);
        if (sfd < 0) {
            return false;
        }
        if (!send_frame(sfd, req, trailing, req_fd >= 0, req_fd)) {
            close(sfd);
            return false;
        }
        const bool ok = recv_frame(sfd, rep, rep_name, rep_bus, rep_fd) &&
                        rep.magic == LLAMA_SHARE_MAGIC && rep.version == LLAMA_SHARE_VERSION;
        close(sfd);
        return ok;
    }

    bool registry_query(const uint8_t * digest, const std::string & bus,
                        int * out_fd, uint64_t * out_seg) {
        *out_fd = -1;
        *out_seg = 0;

        llama_share_hdr req = {};
        req.magic   = LLAMA_SHARE_MAGIC;
        req.version = LLAMA_SHARE_VERSION;
        req.type    = LLAMA_SHARE_QUERY;
        req.bus_len = (uint32_t) bus.size() + 1;
        memcpy(req.digest, digest, 32);

        llama_share_hdr rep;
        std::string rep_name;
        std::string rep_bus;
        int fd = -1;
        if (!registry_roundtrip(req, bus + std::string(1, '\0'), -1, rep, rep_name, rep_bus, &fd)) {
            return false;
        }
        if (rep.type != LLAMA_SHARE_QUERY || rep.status != LLAMA_SHARE_STATUS_HIT || fd < 0) {
            return false;
        }

        *out_fd = fd;
        *out_seg = rep.seg_size;
        return true;
    }

    bool registry_register(const uint8_t * digest, const std::string & bus,
                           uint64_t seg_size, const char * name, int fd) {
        llama_share_hdr req = {};
        req.magic    = LLAMA_SHARE_MAGIC;
        req.version = LLAMA_SHARE_VERSION;
        req.type     = LLAMA_SHARE_REGISTER;
        req.name_len = (uint32_t) strlen(name) + 1;
        req.bus_len  = (uint32_t) bus.size() + 1;
        req.seg_size = seg_size;
        memcpy(req.digest, digest, 32);

        std::string trailing(name, strlen(name) + 1);
        trailing += bus + std::string(1, '\0');

        llama_share_hdr rep;
        std::string rep_name;
        std::string rep_bus;
        int rep_fd = -1;
        if (!registry_roundtrip(req, trailing, fd, rep, rep_name, rep_bus, &rep_fd) ||
                rep.type != LLAMA_SHARE_REGISTER || rep.status != LLAMA_SHARE_STATUS_OK) {
            // the server may or may not have received the fd across a failed
            // exchange; closing ours is always safe (it holds a dup either way)
            close(fd);
            return false;
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// public interface
// ---------------------------------------------------------------------------

llama_share::llama_share(const std::string & ipc_path, const std::string & manifest_path,
                         const std::vector<std::string> & gguf_paths)
    : pimpl(new impl) {
    pimpl->ipc_path     = ipc_path;
    pimpl->manifest_path = manifest_path;
    pimpl->gguf_paths   = gguf_paths;

    if (getenv("LLAMA_SHARE_DISABLE") != nullptr) {
        LLAMA_LOG_INFO("%s: sharing disabled via LLAMA_SHARE_DISABLE\n", __func__);
        return;
    }

    pimpl->manifest_load();
}

llama_share::~llama_share() = default;

bool llama_share::enable() {
    if (pimpl->enabled) {
        return true;
    }

    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (reg == nullptr) {
        LLAMA_LOG_INFO("%s: no CUDA backend registered; weight sharing is disabled\n", __func__);
        return false;
    }

    const std::pair<const char *, void **> procs[] = {
        { "ggml_backend_cuda_buffer_is_shareable",    (void **) &pimpl->fn_is_shareable   },
        { "ggml_backend_cuda_buffer_export_shareable", (void **) &pimpl->fn_export        },
        { "ggml_backend_cuda_buffer_is_attached_shareable", (void **) &pimpl->fn_is_attached },
        { "ggml_backend_cuda_shareable_buffer_type",  (void **) &pimpl->fn_shareable_buft },
        { "ggml_backend_cuda_shareable_buffer_type_attach_spec", (void **) &pimpl->fn_attach_spec },
        { "ggml_backend_cuda_shareable_device_bus_id", (void **) &pimpl->fn_bus_id        },
    };
    for (const auto & p : procs) {
        void * fn = ggml_backend_reg_get_proc_address(reg, p.first);
        if (fn == nullptr) {
            LLAMA_LOG_INFO("%s: CUDA backend lacks %s; weight sharing is disabled\n",
                           __func__, p.first);
            return false;
        }
        *p.second = fn;
    }

    pimpl->enabled = true;
    return true;
}

bool llama_share::is_enabled() const {
    return pimpl->enabled && !pimpl->ipc_path.empty();
}

ggml_backend_buffer_type_t llama_share::shareable_buft(ggml_backend_dev_t dev) {
    if (!is_enabled()) {
        return nullptr;
    }

    // device index within the CUDA registry is what every proc-addressed call takes
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    int idx = -1;
    const size_t n_devs = ggml_backend_reg_dev_count(reg);
    for (size_t i = 0; i < n_devs; i++) {
        if (ggml_backend_reg_dev_get(reg, i) == dev) {
            idx = (int) i;
            break;
        }
    }
    if (idx < 0) {
        return nullptr;
    }

    char bus[64] = {};
    if (pimpl->fn_bus_id(idx, bus, sizeof(bus)) == nullptr) {
        return nullptr; // device has no VMM (or bus id lookup failed)
    }

    ggml_backend_buffer_type_t buft = pimpl->fn_shareable_buft(idx);
    if (buft == nullptr) {
        return nullptr;
    }

    pimpl->buft_bus[buft] = bus;
    return buft;
}

bool llama_share::is_shareable_buft(ggml_backend_buffer_type_t buft) const {
    return pimpl->enabled && pimpl->buft_bus.count(buft) != 0;
}

void llama_share::attach_specs_for_ctx(ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    if (!is_enabled()) {
        return;
    }

    auto it = pimpl->buft_bus.find(buft);
    if (it == pimpl->buft_bus.end()) {
        LLAMA_LOG_WARN("%s: buffer type is not a known shareable buffer type\n", __func__);
        return;
    }
    const std::string & bus = it->second;

    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        auto mit = pimpl->manifest.find(t->name);
        if (mit == pimpl->manifest.end()) {
            continue;
        }
        const size_t nbytes = ggml_nbytes(t);
        if (mit->second.nbytes != nbytes) {
            continue; // same name, different shape/quant: no sharing
        }

        int fd = -1;
        uint64_t seg_size = 0;
        if (!pimpl->registry_query(mit->second.digest.data(), bus, &fd, &seg_size)) {
            continue; // not resident anywhere (or registry down): fill from file instead
        }

        if (!pimpl->fn_attach_spec(buft, t->name, fd, seg_size)) {
            close(fd); // on success the buffer type takes ownership of the fd
            continue;
        }

        // remember the entry so finish() can republish the manifest
        pimpl->new_entries[t->name] = { mit->second.digest, nbytes };
        pimpl->n_attached++;
        pimpl->n_skipped_bytes += nbytes;
        LLAMA_LOG_DEBUG("%s: will attach '%s' (%zu bytes) from the registry\n", __func__, t->name, nbytes);
    }
}

bool llama_share::begin_tensor(const ggml_tensor * cur) {
    pimpl->hashing = false;
    if (!is_enabled() || cur->buffer == nullptr) {
        return false;
    }

    if (!pimpl->fn_is_shareable(cur->buffer)) {
        return false; // tensor lives on a non-shareable buffer: fill normally, no digest
    }

    if (pimpl->fn_is_attached(cur->buffer)) {
        return true; // tensor is already resident via an attached foreign segment
    }

    // owned shareable allocation: digest the bytes that get written
    pimpl->hasher = sha256();
    pimpl->hashing = true;
    return false;
}

void llama_share::on_bytes(const void * ptr, size_t size) {
    if (pimpl->hashing && ptr != nullptr && size > 0) {
        pimpl->hasher.update(ptr, size);
    }
}

void llama_share::end_tensor(ggml_tensor * cur) {
    if (!pimpl->hashing) {
        return;
    }
    pimpl->hashing = false;

    const auto digest = pimpl->hasher.final();
    const size_t nbytes = ggml_nbytes(cur);

    // the register call needs the segment's owning device bus id; resolve it from
    // the buffer type that produced this tensor's buffer
    auto it = pimpl->buft_bus.find(ggml_backend_buffer_get_type(cur->buffer));
    if (it == pimpl->buft_bus.end()) {
        LLAMA_LOG_WARN("%s: cannot register '%s': unknown shareable buffer type\n", __func__, cur->name);
        return;
    }

    int fd = -1;
    size_t seg_size = 0;
    if (pimpl->fn_export(cur->buffer, &fd, &seg_size) != 0) {
        LLAMA_LOG_DEBUG("%s: could not export segment for '%s'\n", __func__, cur->name);
        return;
    }

    if (!pimpl->registry_register(digest.data(), it->second, seg_size, cur->name, fd)) {
        // registry_register closed the fd
        return;
    }

    pimpl->new_entries[cur->name] = { digest, nbytes };
    pimpl->n_registered++;
    LLAMA_LOG_DEBUG("%s: registered '%s' (%zu bytes, segment %zu)\n", __func__, cur->name, nbytes, seg_size);
}

void llama_share::finish(bool success) {
    pimpl->hashing = false;

    if (!pimpl->enabled || !success || pimpl->new_entries.empty()) {
        return;
    }

    if (pimpl->gguf_paths.empty()) {
        LLAMA_LOG_WARN("%s: no weight files given; not writing manifest %s\n",
                       __func__, pimpl->manifest_path.c_str());
        return;
    }

    uint64_t total_bytes = 0;
    uint64_t max_mtime = 0;
    if (!manifest_stat_state(pimpl->gguf_paths, total_bytes, max_mtime)) {
        LLAMA_LOG_WARN("%s: no longer able to stat the weight files; not writing manifest\n", __func__);
        return;
    }

    LLAMA_LOG_INFO("%s: sharing summary - %llu tensors attached (%llu bytes), %llu registered\n",
                   __func__,
                   (unsigned long long) pimpl->n_attached,
                   (unsigned long long) pimpl->n_skipped_bytes,
                   (unsigned long long) pimpl->n_registered);

    std::string out = "#gguf-share-manifest\t" + std::to_string(total_bytes) +
                      "\t" + std::to_string(max_mtime) + "\n";
    for (const auto & kv : pimpl->new_entries) {
        out += kv.first;
        out += '\t';
        out += digest_to_hex(kv.second.digest);
        out += '\t';
        out += std::to_string(kv.second.nbytes);
        out += '\n';
    }

    const std::string tmp = pimpl->manifest_path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "w");
    if (f == nullptr) {
        LLAMA_LOG_WARN("%s: cannot write %s: %s (errno %d)\n", __func__, tmp.c_str(), strerror(errno), errno);
        return;
    }
    const bool ok = fwrite(out.data(), 1, out.size(), f) == out.size();
    if (fflush(f) != 0) {
        fclose(f);
        LLAMA_LOG_WARN("%s: flushing %s failed\n", __func__, tmp.c_str());
        return;
    }
    int fd = fileno(f);
    if (fd >= 0) {
        fsync(fd);
    }
    fclose(f);
    if (!ok || rename(tmp.c_str(), pimpl->manifest_path.c_str()) != 0) {
        LLAMA_LOG_WARN("%s: publishing manifest %s failed\n", __func__, pimpl->manifest_path.c_str());
        return;
    }

    LLAMA_LOG_INFO("%s: wrote manifest %s with %zu entries\n",
                   __func__, pimpl->manifest_path.c_str(), pimpl->new_entries.size());
}