#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

static bool run_tensor_n_case(
        std::vector<ggml_tensor *> & tensors,
        const std::vector<size_t> & offsets,
        const std::vector<size_t> & sizes,
        const std::vector<size_t> & data_offsets) {
    const size_t n = tensors.size();

    std::vector<const ggml_tensor *> const_tensors(tensors.begin(), tensors.end());
    std::vector<uint8_t> data(256, 0);
    std::vector<uint8_t> out(256, 0);

    for (size_t i = 0; i < n; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        memset(data.data() + data_offsets[i], (int) (0x40 + i*0x10), sizes[i]);
    }

    ggml_backend_tensor_set_n(const_tensors.data(), data.data(), offsets.data(), sizes.data(), data_offsets.data(), n);

    ggml_backend_tensor_get_n(const_tensors.data(), out.data(), offsets.data(), sizes.data(), data_offsets.data(), n);

    for (size_t i = 0; i < n; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        if (memcmp(data.data() + data_offsets[i], out.data() + data_offsets[i], sizes[i]) != 0) {
            return false;
        }
    }

    std::vector<uint8_t> slice(256, 0);
    for (size_t i = 0; i < n; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        memset(slice.data(), 0, slice.size());
        ggml_backend_tensor_get(tensors[i], slice.data(), offsets[i], sizes[i]);
        if (memcmp(slice.data(), data.data() + data_offsets[i], sizes[i]) != 0) {
            return false;
        }
    }

    return true;
}

static size_t align_up(size_t x, size_t align) {
    return (x + align - 1) / align * align;
}

static bool test_backend_tensor_n_mixed_buft(ggml_backend_t backend) {
    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead()*16 + 2048;
    params.no_alloc = true;

    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    ggml_tensor * t0 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
    ggml_tensor * t1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 2);
    ggml_tensor * t2 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor * s0 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
    ggml_tensor * s1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 2);
    ggml_tensor * s2 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_buffer_type_t buft0 = ggml_backend_dev_buffer_type(dev);

    const size_t align0 = ggml_backend_buft_get_alignment(buft0);
    const size_t sz_t0 = ggml_backend_buft_get_alloc_size(buft0, t0);
    const size_t sz_t2 = ggml_backend_buft_get_alloc_size(buft0, t2);
    const size_t buf0_size = align_up(sz_t0, align0) + sz_t2;

    ggml_backend_buffer_ptr buf0(ggml_backend_buft_alloc_buffer(buft0, buf0_size));
    if (!buf0) {
        return false;
    }

    char * base0 = (char *) ggml_backend_buffer_get_base(buf0.get());
    if (ggml_backend_tensor_alloc(buf0.get(), t0, base0) != GGML_STATUS_SUCCESS) {
        return false;
    }
    char * p2 = base0 + align_up(sz_t0, align0);
    if (ggml_backend_tensor_alloc(buf0.get(), t2, p2) != GGML_STATUS_SUCCESS) {
        return false;
    }

    const size_t host_align = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
    std::vector<uint8_t> host(1024 + host_align, 0);
    uint8_t * host_ptr = (uint8_t *)(((uintptr_t)host.data() + host_align - 1) / host_align * host_align);
    ggml_backend_buffer_ptr buf1(ggml_backend_cpu_buffer_from_ptr(host_ptr, 1024));
    if (!buf1) {
        return false;
    }

    ggml_backend_buffer_type_t buft1 = ggml_backend_buffer_get_type(buf1.get());
    const size_t align1 = ggml_backend_buft_get_alignment(buft1);
    const size_t sz_t1 = ggml_backend_buft_get_alloc_size(buft1, t1);
    const size_t sz_s0 = ggml_backend_buft_get_alloc_size(buft1, s0);
    const size_t sz_s1 = ggml_backend_buft_get_alloc_size(buft1, s1);

    char * base1 = (char *) ggml_backend_buffer_get_base(buf1.get());
    char * p = base1;
    if (ggml_backend_tensor_alloc(buf1.get(), t1, p) != GGML_STATUS_SUCCESS) {
        return false;
    }
    p += align_up(sz_t1, align1);
    if (ggml_backend_tensor_alloc(buf1.get(), s0, p) != GGML_STATUS_SUCCESS) {
        return false;
    }
    p += align_up(sz_s0, align1);
    if (ggml_backend_tensor_alloc(buf1.get(), s1, p) != GGML_STATUS_SUCCESS) {
        return false;
    }
    p += align_up(sz_s1, align1);
    if (ggml_backend_tensor_alloc(buf1.get(), s2, p) != GGML_STATUS_SUCCESS) {
        return false;
    }

    uint8_t src0[12], src1[8], src2[16];
    memset(src0, 0xA1, sizeof(src0));
    memset(src1, 0xB2, sizeof(src1));
    memset(src2, 0xC3, sizeof(src2));

    ggml_backend_tensor_set(s0, src0, 0, sizeof(src0));
    ggml_backend_tensor_set(s1, src1, 0, sizeof(src1));
    ggml_backend_tensor_set(s2, src2, 0, sizeof(src2));

    {
        std::vector<ggml_tensor *> tensors = { t0, t1, t2 };
        std::vector<size_t> offsets = { 0, 0, 0 };
        std::vector<size_t> sizes = { sizeof(src0), sizeof(src1), sizeof(src2) };
        std::vector<size_t> data_offsets = { 0, sizeof(src0), sizeof(src0) + sizeof(src1) };

        if (!run_tensor_n_case(tensors, offsets, sizes, data_offsets)) {
            return false;
        }
    }

    {
        std::vector<const ggml_tensor *> srcs = { s0, s1, s2 };
        std::vector<ggml_tensor *> dsts = { t0, t1, t2 };
        ggml_backend_tensor_copy_n(srcs.data(), dsts.data(), srcs.size());

        const uint8_t * expected[3] = { src0, src1, src2 };
        const size_t sizes[3] = { sizeof(src0), sizeof(src1), sizeof(src2) };
        std::vector<uint8_t> out(64, 0);
        for (size_t i = 0; i < 3; i++) {
            memset(out.data(), 0, out.size());
            ggml_backend_tensor_get(dsts[i], out.data(), 0, sizes[i]);
            if (memcmp(out.data(), expected[i], sizes[i]) != 0) {
                return false;
            }
        }
    }

    return true;
}

static bool test_backend_tensor_n(ggml_backend_t backend) {
    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead()*32 + 1024;
    params.no_alloc = true;

    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    ggml_tensor * t0 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
    ggml_tensor * t1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 2);
    ggml_tensor * t2 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor * t3 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
    ggml_tensor * t_big = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 6);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) {
        return false;
    }

    {
        std::vector<ggml_tensor *> tensors = { t0, t1, t2, t0 };
        std::vector<size_t> offsets = { 0, 0, 0, 0 };
        std::vector<size_t> sizes = { 3*4, 2*4, 4*4, 0 };
        std::vector<size_t> data_offsets = { 0, 3*4, 5*4, 64 };

        if (!run_tensor_n_case(tensors, offsets, sizes, data_offsets)) {
            return false;
        }
    }

    {
        std::vector<ggml_tensor *> tensors = { t_big, t_big, t_big };
        std::vector<size_t> offsets = { 0, 8, 16 };
        std::vector<size_t> sizes = { 8, 8, 8 };
        std::vector<size_t> data_offsets = { 0, 8, 16 };

        if (!run_tensor_n_case(tensors, offsets, sizes, data_offsets)) {
            return false;
        }
    }

    {
        std::vector<ggml_tensor *> tensors = { t0, t3 };
        std::vector<size_t> offsets = { 0, 0 };
        std::vector<size_t> sizes = { 3*4, 3*4 };
        std::vector<size_t> data_offsets = { 0, 3*4 };

        if (!run_tensor_n_case(tensors, offsets, sizes, data_offsets)) {
            return false;
        }
    }

    return true;
}

struct batch_read {
    ggml_tensor * tensor;
    const std::vector<unsigned char> * source;
    size_t base;
    size_t offset;
    size_t size;
    size_t destination;
};

static bool run_tensor_get_n_case(const std::vector<batch_read> & reads, size_t size) {
    std::vector<unsigned char> actual(size, 0xa5);
    std::vector<unsigned char> expected(size, 0xa5);
    std::vector<const ggml_tensor *> tensors;
    std::vector<size_t> offsets, sizes, data_offsets;
    for (const auto & read : reads) {
        tensors.push_back(read.tensor);
        offsets.push_back(read.offset);
        sizes.push_back(read.size);
        data_offsets.push_back(read.destination);
        std::memcpy(expected.data() + read.destination,
                    read.source->data() + read.base + read.offset, read.size);
    }
    ggml_backend_tensor_get_n(tensors.data(), actual.data(), offsets.data(), sizes.data(), data_offsets.data(), reads.size());
    if (actual != expected) {
        for (size_t i = 0; i < size; ++i) {
            if (actual[i] != expected[i]) {
                std::fprintf(stderr, "byte mismatch at %zu: %u != %u\n", i, actual[i], expected[i]);
                break;
            }
        }
        return false;
    }
    return true;
}

static void fill_bytes(std::vector<unsigned char> & bytes, unsigned int seed) {
    for (auto & byte : bytes) {
        seed = seed * 1664525u + 1013904223u;
        byte = (unsigned char) (seed >> 24);
    }
}

static bool test_type(ggml_backend_t backend, ggml_type type) {
    ggml_init_params params = {8 * ggml_tensor_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, 512, 129);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, type, 512, 129);
    const size_t row = a->nb[1];
    ggml_tensor * view = ggml_view_2d(ctx, a, 256, 31, row, 7 * row);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    std::vector<unsigned char> source_a(ggml_nbytes(a));
    std::vector<unsigned char> source_b(ggml_nbytes(b));
    fill_bytes(source_a, 17);
    fill_bytes(source_b, 29);
    ggml_backend_tensor_set(a, source_a.data(), 0, source_a.size());
    ggml_backend_tensor_set(b, source_b.data(), 0, source_b.size());

    std::vector<batch_read> reads;
    size_t output_size = 13;
    auto add = [&](ggml_tensor * tensor, const std::vector<unsigned char> & source,
                   size_t base, size_t offset, size_t size) {
        reads.push_back({tensor, &source, base, offset, size, output_size});
        output_size += size + 11;
    };
    // Unsorted and repeated source rows, multiple tensors, and destination guard gaps.
    for (size_t index : {127u, 3u, 89u, 3u, 0u, 128u, 61u}) {
        add(a, source_a, 0, index * row, row);
        add(b, source_b, 0, index * row, row);
    }
    add(a, source_a, 0, 12 * row, 8 * row);
    add(view, source_a, 7 * row, 3 * row, ggml_row_size(type, 256));
    add(view, source_a, 7 * row, 1, 7);
    add(a, source_a, 0, source_a.size(), 0);
    for (size_t src_alignment = 0; src_alignment < 16; ++src_alignment) {
        for (size_t dst_alignment = 0; dst_alignment < 16; ++dst_alignment) {
            output_size = (output_size + 15) / 16 * 16 + dst_alignment;
            add(a, source_a, 0, src_alignment, 65);
        }
    }
    // Many short reads with unaligned offsets and untouched destination bytes.
    for (size_t i = 0; i < 65537; ++i) {
        add(a, source_a, 0, (i * 37) % (source_a.size() - 2), 2);
    }
    const bool ok = run_tensor_get_n_case(reads, output_size);
    std::printf("%s raw bytes, views, fragmentation, 65537 tiny reads: %s\n", ggml_type_name(type), ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

static bool test_column_ranges(ggml_backend_t backend, ggml_type type) {
    ggml_init_params params = {ggml_tensor_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = ggml_new_tensor_2d(ctx, type, 65, 37);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    std::vector<unsigned char> source(ggml_nbytes(tensor));
    fill_bytes(source, 97);
    ggml_backend_tensor_set(tensor, source.data(), 0, source.size());
    std::vector<batch_read> reads;
    size_t output_size = 19;
    const size_t element = ggml_type_size(type);
    for (size_t row = 0; row < 37; ++row) {
        for (size_t column : {64u, 1u, 31u, 1u, 0u}) {
            reads.push_back({tensor, &source, 0, (row * 65 + column) * element, element, output_size});
            output_size += element + 3;
        }
    }
    const bool ok = run_tensor_get_n_case(reads, output_size);
    std::printf("%s selected columns with destination gaps: %s\n", ggml_type_name(type), ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

static bool test_large(ggml_backend_t backend) {
    ggml_init_params params = {ggml_tensor_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    const size_t size = 256 * 1024 * 1024 + 501;
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, size);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    std::vector<unsigned char> source(size);
    fill_bytes(source, 131);
    ggml_backend_tensor_set(tensor, source.data(), 0, source.size());
    std::vector<batch_read> reads = {{tensor, &source, 0, 0, size, 19}};
    const bool ok = run_tensor_get_n_case(reads, size + 41) && run_tensor_get_n_case(reads, size + 41);
    std::printf("large byte range and repeated reads: %s\n", ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

static bool test_range_boundaries(ggml_backend_t backend) {
    ggml_init_params params = {ggml_tensor_overhead(), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, 128 * 1024);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer);
    std::vector<unsigned char> source(ggml_nbytes(tensor));
    fill_bytes(source, 113);
    ggml_backend_tensor_set(tensor, source.data(), 0, source.size());
    std::vector<batch_read> reads;
    size_t output_size = 13;
    for (size_t size : {65535u, 65536u, 65537u}) {
        reads.push_back({tensor, &source, 0, 3, size, output_size});
        output_size += size + 11;
    }
    const bool ok = run_tensor_get_n_case(reads, output_size);
    std::printf("range boundaries at 64 KiB: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_concurrent_reads(ggml_backend_t backend) {
    ggml_init_params params = {ggml_tensor_overhead(), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, 1024 * 1024);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer);
    std::vector<unsigned char> source(ggml_nbytes(tensor));
    fill_bytes(source, 149);
    ggml_backend_tensor_set(tensor, source.data(), 0, source.size());
    const size_t counts[] = {2048, 131073};
    const size_t sizes[] = {513, 9};
    const size_t strides[] = {529, 17};
    std::vector<batch_read> reads[2];
    for (size_t i = 0; i < 2; ++i) {
        reads[i].reserve(counts[i]);
        for (size_t j = 0; j < counts[i]; ++j) {
            reads[i].push_back({tensor, &source, 0, (j * 37 + i * 101) % (source.size() - sizes[i]), sizes[i], 13 + j * strides[i]});
        }
    }
    std::atomic<int> ready{0};
    bool ok[] = {true, true};
    std::thread workers[2];
    for (size_t i = 0; i < 2; ++i) {
        workers[i] = std::thread([&, i]() {
            ready.fetch_add(1);
            while (ready.load() < 2) {
                std::this_thread::yield();
            }
            for (int repeat = 0; repeat < 4; ++repeat) {
                ok[i] = run_tensor_get_n_case(reads[i], counts[i] * strides[i] + 31) && ok[i];
            }
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }
    std::printf("concurrent reads from one source buffer: %s\n", ok[0] && ok[1] ? "PASS" : "FAIL");
    return ok[0] && ok[1];
}

static bool test_overlapping_destinations(ggml_backend_t backend) {
    ggml_init_params params = {ggml_tensor_overhead(), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, 4096);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer);
    std::vector<unsigned char> source(4096);
    fill_bytes(source, 53);
    ggml_backend_tensor_set(tensor, source.data(), 0, source.size());
    const std::vector<batch_read> reads = {
        {tensor, &source, 0, 0, 64, 32},
        {tensor, &source, 0, 131, 32, 48},
        {tensor, &source, 0, 257, 64, 16},
        {tensor, &source, 0, 389, 64, 32},
    };
    const bool ok = run_tensor_get_n_case(reads, 128);
    std::printf("overlapping destinations retain copy order: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_many_ranges(ggml_backend_t backend) {
    ggml_init_params params = {ggml_tensor_overhead(), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, 4096);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer);
    std::vector<unsigned char> source(4096);
    fill_bytes(source, 83);
    ggml_backend_tensor_set(tensor, source.data(), 0, source.size());
    const size_t count = 1024 * 1024 + 1;
    std::vector<batch_read> reads;
    reads.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        reads.push_back({tensor, &source, 0, i * 37 % source.size(), 1, 13 + 2*i});
    }
    const bool ok = run_tensor_get_n_case(reads, 2*count + 31);
    std::printf("1048577 short reads with destination gaps: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_mixed_buffers(ggml_backend_t backend) {
    ggml_init_params params = {ggml_tensor_overhead(), nullptr, true};
    ggml_context_ptr ctx_a(ggml_init(params));
    ggml_context_ptr ctx_b(ggml_init(params));
    ggml_tensor * a = ggml_new_tensor_1d(ctx_a.get(), GGML_TYPE_I8, 4096);
    ggml_tensor * b = ggml_new_tensor_1d(ctx_b.get(), GGML_TYPE_I8, 4096);
    ggml_backend_buffer_ptr buffer_a(ggml_backend_alloc_ctx_tensors(ctx_a.get(), backend));
    ggml_backend_buffer_ptr buffer_b(ggml_backend_alloc_ctx_tensors(ctx_b.get(), backend));
    GGML_ASSERT(buffer_a && buffer_b);
    std::vector<unsigned char> source_a(4096), source_b(4096);
    fill_bytes(source_a, 19);
    fill_bytes(source_b, 23);
    ggml_backend_tensor_set(a, source_a.data(), 0, source_a.size());
    ggml_backend_tensor_set(b, source_b.data(), 0, source_b.size());
    const std::vector<batch_read> reads = {
        {a, &source_a, 0, 29, 71, 13},
        {b, &source_b, 0, 37, 131, 4097},
        {a, &source_a, 0, 73, 65, 113},
    };
    const bool ok = run_tensor_get_n_case(reads, 8192);
    std::printf("reads from separate source buffers: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_backend_tensor_get_n(ggml_backend_t backend, bool large) {
    bool ok = true;
    ggml_backend_tensor_get_n(nullptr, nullptr, nullptr, nullptr, nullptr, 0);
    for (ggml_type type : {GGML_TYPE_I8, GGML_TYPE_I64, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                         GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1}) {
        ok = test_type(backend, type) && ok;
    }
    for (ggml_type type : {GGML_TYPE_I8, GGML_TYPE_I64, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        ok = test_column_ranges(backend, type) && ok;
    }
    ok = test_overlapping_destinations(backend) && ok;
    ok = test_mixed_buffers(backend) && ok;
    ok = test_range_boundaries(backend) && ok;
    if (large) {
        ok = test_large(backend) && ok;
        ok = test_many_ranges(backend) && ok;
        ok = test_concurrent_reads(backend) && ok;
    }
    return ok;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    const size_t n_devs = ggml_backend_dev_count();
    size_t n_ok = 0;

    for (size_t i = 0; i < n_devs; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);

        printf("%s ", name);
        fflush(stdout);

        ggml_backend_ptr backend(ggml_backend_dev_init(dev, NULL));
        if (!backend) {
            printf("FAILED\n");
            continue;
        }

        const bool large = argc > 1 && std::strcmp(argv[1], "--large") == 0;
        const bool ok = test_backend_tensor_n(backend.get()) && test_backend_tensor_n_mixed_buft(backend.get()) && test_backend_tensor_get_n(backend.get(), large);
        printf("%s\n", ok ? "PASSED" : "FAILED");
        if (ok) {
            n_ok++;
        }
    }

    printf("%zu/%zu backends passed\n", n_ok, n_devs);

    return n_ok == n_devs ? 0 : 1;
}
