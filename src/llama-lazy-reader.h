#pragma once

#include "ggml.h"
#include "llama-mmap.h"

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

struct llama_lazy_reader;

struct llama_lazy_reader_factory {
    void add(const ggml_tensor * tensor, const llama_file & source, size_t offs);
    void add_virtual(const ggml_tensor * tensor);
    bool has(const ggml_tensor * tensor) const;
    std::unique_ptr<llama_lazy_reader> create(int n_readers) const;

private:
    friend struct llama_lazy_reader;

    struct tensor_info {
        size_t file;
        size_t offs;
        size_t rsize;
        int64_t nrows;
    };

    std::map<const ggml_tensor *, tensor_info> tensors;
    std::vector<std::unique_ptr<llama_file>> sources;
};

struct llama_lazy_reader {
    llama_lazy_reader(const llama_lazy_reader_factory & factory, int n_readers);

    llama_lazy_reader(const llama_lazy_reader &) = delete;
    llama_lazy_reader & operator=(const llama_lazy_reader &) = delete;

    ~llama_lazy_reader();

    bool has(const ggml_tensor * tensor) const { return factory.has(tensor); }
    void gather(const ggml_tensor * tensor, const int32_t * rows, int64_t n, uint8_t * dst) const;

private:
    struct run_span {
        int32_t first_row;
        int32_t n_unique_rows;
        int32_t pair_begin;
        int32_t pair_end;
    };

    void read_runs(const llama_lazy_reader_factory::tensor_info & info,
                   const std::pair<int32_t, int32_t> * pairs,
                   const run_span * runs, int64_t begin, int64_t end,
                   size_t fi, uint8_t * dst) const;

    void worker_loop(int w);

    const llama_lazy_reader_factory & factory;
    std::vector<std::vector<std::unique_ptr<llama_file>>> files;

    // persistent worker pool for parallel pread across NVMe queues
    mutable std::mutex mtx;
    mutable std::condition_variable cv_work;
    mutable std::condition_variable cv_done;
    std::vector<std::thread> workers;
    mutable std::vector<std::exception_ptr> worker_errs;

    mutable uint64_t epoch = 0;
    mutable int active_workers = 0;
    mutable int remaining_workers = 0;
    mutable bool stop = false;

    mutable const llama_lazy_reader_factory::tensor_info * cur_info = nullptr;
    mutable const std::pair<int32_t, int32_t> * cur_pairs = nullptr;
    mutable const run_span * cur_runs = nullptr;
    mutable int64_t cur_n_runs = 0;
    mutable uint8_t * cur_dst = nullptr;

    mutable std::vector<std::pair<int32_t, int32_t>> scratch_pairs;
    mutable std::vector<run_span> scratch_runs;
    mutable std::vector<std::vector<uint8_t>> worker_bounces;
};
