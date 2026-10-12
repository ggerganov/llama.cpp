#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

// Cross-process sharing of weight tensors over CUDA VMM segments (Linux, requires the
// ggml CUDA backend built with shareable-buffer support).
//
// Every process loading a model with share_ipc set behaves symmetrically, so the first
// instance produces and later instances consume with zero configuration:
//   - before allocating a context on a shareable buffer type, every tensor is looked up
//     in the manifest (name -> digest, only when sizes match) and queried against the
//     registry; a hit attaches the registered segment and the load skips reading that
//     tensor's bytes altogether
//   - after a tensor (on a shareable buffer) is filled from weight data, its bytes are
//     hashed and the buffer's segment is exported and registered under the digest
// The registry is a small unix-socket daemon (llama-share-registry, no CUDA needed) that
// keeps the exported fds open, anchoring the physical allocations even when a producer
// exits. A missing registry, stale manifests and changed model files all degrade to an
// unshared load with a warning; nothing else changes.

// The registry daemon (tools/llama-share-registry) is separate from the llama library;
// this header describes the wire format it shares with llama-share.cpp so both stay in
// sync.

#define LLAMA_SHARE_MAGIC   0x50485347 // "GSHP" little-endian
#define LLAMA_SHARE_VERSION 1

enum llama_share_msg_type : uint32_t {
    LLAMA_SHARE_QUERY    = 1, // request: digest (and bus id); reply: HIT (fd + bus id + seg size) or MISS
    LLAMA_SHARE_REGISTER = 2, // request: digest + bus id + seg size + name + fd; reply: OK or ERR
};

enum llama_share_msg_status : uint32_t {
    LLAMA_SHARE_STATUS_OK   = 0,
    LLAMA_SHARE_STATUS_MISS = 1,
    LLAMA_SHARE_STATUS_HIT  = 2,
    LLAMA_SHARE_STATUS_ERR  = 3,
};

struct llama_share_hdr { // packed; one message per connection
    uint32_t magic;    // LLAMA_SHARE_MAGIC
    uint32_t version;  // LLAMA_SHARE_VERSION
    uint32_t type;     // llama_share_msg_type; replies echo the request's type (0 on transport-level error replies)
    uint32_t status;   // llama_share_msg_status; 0 in requests
    uint32_t name_len; // trailing name bytes (REGISTER request; 0 otherwise)
    uint32_t bus_len;  // trailing bus-id bytes; QUERY filters by it, REGISTER stores it, HIT reply echoes the registrant
    uint64_t seg_size; // REGISTER request and HIT reply
    uint8_t  digest[32];
} __attribute__((packed));

struct llama_share {
    // gguf_paths: the model's weight files (used to stat the gguf state that the manifest
    // was generated from; pass empty to skip manifest freshness gating entirely).
    llama_share(const std::string & ipc_path, const std::string & manifest_path,
                const std::vector<std::string> & gguf_paths);
    ~llama_share();

    llama_share(const llama_share &) = delete;
    llama_share & operator=(const llama_share &) = delete;

    // resolve the shareable-buffer functions on the CUDA backend registry; false when the
    // backend does not support sharing (the object then behaves fully disabled)
    bool enable();

    bool is_enabled() const;

    // buffer type to substitute a device's default buffer type with, or nullptr when the
    // device's CUDA context has no VMM support
    ggml_backend_buffer_type_t shareable_buft(ggml_backend_dev_t dev);
    bool is_shareable_buft(ggml_backend_buffer_type_t buft) const;

    // called before allocating ctx on buft: register attach specs for the context's
    // tensors that the registry can already supply
    void attach_specs_for_ctx(ggml_context * ctx, ggml_backend_buffer_type_t buft);

    // load_all_data hooks; one begin/on_bytes*/end sequence per tensor
    bool begin_tensor(const ggml_tensor * cur); // true = tensor already resident, skip the fill
    void on_bytes(const void * ptr, size_t size); // feed the digest
    void end_tensor(ggml_tensor * cur);

    // after loading; writes newly computed digests to the manifest file
    void finish(bool success);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};