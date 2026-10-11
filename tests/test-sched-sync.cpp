// Tests the scheduler's cross-backend zero-copy producer synchronization.
//
// The scheduler already synchronizes the backend of the *previous* split, but only when the
// current split has no copied inputs. The shared-producer synchronization is what covers the
// remaining case: a split that reads another backend's buffer directly (zero-copy) without
// going through a copy. This test builds that exact situation with dummy backends.
//
// Backends in the same "group" can map each other's buffers (as with dma-buf), so the scheduler
// keeps those tensors in place. A backend in a different group forces a copy.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct dummy_backend {
    const char * name = nullptr;
    int group = 0;
    std::vector<uint8_t> mem;
    int n_compute = 0;

    ggml_backend_buffer_i    buffer_iface{};
    ggml_backend_device      device{};
    ggml_backend             backend{};
    ggml_backend_buffer_type buffer_type{};
};

static std::vector<std::string> g_events;

static const char * dummy_buft_get_name(ggml_backend_buffer_type_t) {
    return "dummy_buft";
}

static ggml_backend_buffer_t dummy_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    dummy_backend * b = (dummy_backend *) buft->context;
    return ggml_backend_buffer_init(buft, b->buffer_iface, b, size);
}

static size_t dummy_buft_get_alignment(ggml_backend_buffer_type_t) {
    return 16;
}

static size_t dummy_buft_get_max_size(ggml_backend_buffer_type_t) {
    return SIZE_MAX;
}

static bool dummy_buft_is_host(ggml_backend_buffer_type_t) {
    return false;
}

static void dummy_buf_free_buffer(ggml_backend_buffer_t) {
}

static void * dummy_buf_get_base(ggml_backend_buffer_t buffer) {
    return ((dummy_backend *) buffer->context)->mem.data();
}

static ggml_status dummy_buf_init_tensor(ggml_backend_buffer_t, ggml_tensor *) {
    return GGML_STATUS_SUCCESS;
}

static void dummy_buf_memset_tensor(ggml_backend_buffer_t, ggml_tensor *, uint8_t, size_t, size_t) {}
static void dummy_buf_set_tensor(ggml_backend_buffer_t, ggml_tensor *, const void *, size_t, size_t) {}
static void dummy_buf_get_tensor(ggml_backend_buffer_t, const ggml_tensor *, void *, size_t, size_t) {}
static void dummy_buf_clear(ggml_backend_buffer_t, uint8_t) {}

static const char * dummy_dev_get_name(ggml_backend_dev_t dev) {
    return ((dummy_backend *) dev->context)->name;
}

static const char * dummy_dev_get_description(ggml_backend_dev_t dev) {
    return ((dummy_backend *) dev->context)->name;
}

static enum ggml_backend_dev_type dummy_dev_get_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;
}

static bool dummy_dev_supports_op(ggml_backend_dev_t, const ggml_tensor *) {
    return true;
}

static bool dummy_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ((dummy_backend *) dev->context)->group == ((dummy_backend *) buft->context)->group;
}

static const char * dummy_backend_get_name(ggml_backend_t backend) {
    return ((dummy_backend *) backend->context)->name;
}

static void dummy_synchronize(ggml_backend_t backend) {
    dummy_backend * b = (dummy_backend *) backend->context;
    g_events.push_back(std::string("sync:") + b->name);
}

static enum ggml_status dummy_graph_compute(ggml_backend_t backend, ggml_cgraph *) {
    dummy_backend * b = (dummy_backend *) backend->context;
    b->n_compute++;
    g_events.push_back(std::string("compute:") + b->name);
    return GGML_STATUS_SUCCESS;
}

static void dummy_init(dummy_backend * b, const char * name, int group) {
    b->name  = name;
    b->group = group;
    b->mem.resize(1 << 20);

    b->buffer_iface.free_buffer   = dummy_buf_free_buffer;
    b->buffer_iface.get_base      = dummy_buf_get_base;
    b->buffer_iface.init_tensor   = dummy_buf_init_tensor;
    b->buffer_iface.memset_tensor = dummy_buf_memset_tensor;
    b->buffer_iface.set_tensor    = dummy_buf_set_tensor;
    b->buffer_iface.get_tensor    = dummy_buf_get_tensor;
    b->buffer_iface.clear         = dummy_buf_clear;

    b->device.context               = b;
    b->device.iface.get_name        = dummy_dev_get_name;
    b->device.iface.get_description = dummy_dev_get_description;
    b->device.iface.get_type        = dummy_dev_get_type;
    b->device.iface.supports_op     = dummy_dev_supports_op;
    b->device.iface.supports_buft   = dummy_dev_supports_buft;

    b->backend.context             = b;
    b->backend.device              = &b->device;
    b->backend.iface.get_name      = dummy_backend_get_name;
    b->backend.iface.synchronize   = dummy_synchronize;
    b->backend.iface.graph_compute = dummy_graph_compute;

    b->buffer_type.device              = &b->device;
    b->buffer_type.context             = b;
    b->buffer_type.iface.get_name      = dummy_buft_get_name;
    b->buffer_type.iface.alloc_buffer  = dummy_buft_alloc_buffer;
    b->buffer_type.iface.get_alignment = dummy_buft_get_alignment;
    b->buffer_type.iface.get_max_size  = dummy_buft_get_max_size;
    b->buffer_type.iface.is_host       = dummy_buft_is_host;
}

static int find_event(const char * e) {
    for (size_t i = 0; i < g_events.size(); i++) {
        if (g_events[i] == e) {
            return (int) i;
        }
    }
    return -1;
}

static int last_event_before(const char * e, int before) {
    int idx = -1;
    for (int i = 0; i < before && i < (int) g_events.size(); i++) {
        if (g_events[i] == e) {
            idx = i;
        }
    }
    return idx;
}

// one producer split on a, one consumer split on b that reads a's output zero-copy and also
// copies an input from d (so the consumer split has inputs, which disables the scheduler's
// "previous split" synchronization)
static void test_zero_copy_sync(dummy_backend * a, dummy_backend * b, dummy_backend * d) {
    g_events.clear();

    ggml_context * ctx_static = ggml_init({ 4 * ggml_tensor_overhead(), nullptr, true });
    ggml_tensor * w = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 4);
    ggml_set_name(w, "w");
    ggml_backend_buffer_t w_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_static, &d->buffer_type);
    GGML_ASSERT(w_buf != nullptr);

    ggml_context * ctx = ggml_init({ 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true });
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(x);
    ggml_set_name(x, "x");

    ggml_tensor * t1 = ggml_scale(ctx, x, 2.0f);      // producer, on a
    ggml_set_name(t1, "t1");
    ggml_tensor * t2 = ggml_add(ctx, t1, w);          // consumer, on b, reads t1 zero-copy
    ggml_set_name(t2, "t2");

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_set_output(t2);
    ggml_build_forward_expand(graph, t2);

    ggml_backend_t backends[3] = { &a->backend, &b->backend, &d->backend };
    ggml_backend_buffer_type_t bufts[3] = { &a->buffer_type, &b->buffer_type, &d->buffer_type };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 3, GGML_DEFAULT_GRAPH_SIZE, false, true);
    GGML_ASSERT(sched != nullptr);

    ggml_backend_sched_set_tensor_backend(sched, t1, &a->backend);
    ggml_backend_sched_set_tensor_backend(sched, t2, &b->backend);

    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);

    // a produced t1 zero-copy, so b must synchronize a after a has computed and before b computes.
    // counters alone are not enough, allocating the graph synchronizes every backend once
    const int i_compute_b = find_event("compute:b");
    const int i_compute_a = last_event_before("compute:a", i_compute_b);
    const int i_sync_a    = last_event_before("sync:a", i_compute_b);

    GGML_ASSERT(a->n_compute == 1);
    GGML_ASSERT(b->n_compute == 1);
    GGML_ASSERT(d->n_compute == 0);
    GGML_ASSERT(i_compute_b >= 0);
    GGML_ASSERT(i_compute_a >= 0);
    GGML_ASSERT(i_sync_a > i_compute_a);

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(w_buf);
    ggml_free(ctx_static);
    ggml_free(ctx);
}

int main() {
    ggml_log_set(nullptr, nullptr);

    // a and b share buffers (zero-copy); d does not, so its tensor is copied
    dummy_backend a;
    dummy_backend b;
    dummy_backend d;
    dummy_init(&a, "a", 0);
    dummy_init(&b, "b", 0);
    dummy_init(&d, "d", 1);

    test_zero_copy_sync(&a, &b, &d);

    printf("PASSED\n");
    return 0;
}
