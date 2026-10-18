#include "llama-lazy-reader.h"

#include "llama-impl.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

void llama_lazy_reader_factory::add(const ggml_tensor * tensor, const llama_file & source, size_t offs) {
    if (tensor->type != GGML_TYPE_F32 && ggml_get_type_traits(tensor->type)->to_float == nullptr) {
        throw std::runtime_error(format("%s cannot be read row by row: %s has no F32 conversion",
                source.name().c_str(), ggml_type_name(tensor->type)));
    }

    GGML_ASSERT(tensor->ne[0] > 0 && tensor->ne[1] > 0);

    size_t file = 0;
    while (file < sources.size() && !source.same_file(*sources[file])) {
        ++file;
    }
    if (file == sources.size()) {
        const auto & name = source.name();
#if defined(__cpp_lib_char8_t)
        const auto path = std::filesystem::absolute(std::filesystem::path(std::u8string(name.begin(), name.end())));
#else
        const auto path = std::filesystem::absolute(std::filesystem::u8path(name));
#endif
        const auto u8path = path.u8string();
        const std::string abs_path(reinterpret_cast<const char *>(u8path.data()), u8path.size());
        auto handle = std::make_unique<llama_file>(abs_path.c_str(), "rb", /*use_direct_io =*/ false);
        if (!source.same_file(*handle)) {
            throw std::runtime_error(format("model file changed while opening lazy reader: %s", source.name().c_str()));
        }
        sources.emplace_back(std::move(handle));
    }

    const size_t rsize = ggml_row_size(tensor->type, tensor->ne[0]);
    const int64_t nrows = tensor->ne[1];
    sources[file]->advise_random(offs, rsize * (size_t) nrows);

    auto result = tensors.emplace(tensor, tensor_info { file, offs, rsize, nrows });
    GGML_ASSERT(result.second);
}

void llama_lazy_reader_factory::add_virtual(const ggml_tensor * tensor) {
    GGML_ASSERT(tensor->ne[0] > 0 && tensor->ne[1] > 0);
    const size_t rsize = ggml_row_size(tensor->type, tensor->ne[0]);
    const int64_t nrows = tensor->ne[1];
    auto result = tensors.emplace(tensor, tensor_info { SIZE_MAX, 0, rsize, nrows });
    GGML_ASSERT(result.second);
}

bool llama_lazy_reader_factory::has(const ggml_tensor * tensor) const {
    return tensors.find(tensor) != tensors.end();
}

std::unique_ptr<llama_lazy_reader> llama_lazy_reader_factory::create(int n_readers) const {
    return std::make_unique<llama_lazy_reader>(*this, n_readers);
}

llama_lazy_reader::llama_lazy_reader(const llama_lazy_reader_factory & factory, int n_readers) : factory(factory) {
    GGML_ASSERT(n_readers > 0);
    if (factory.sources.empty()) {
        return;
    }
    // cap persistent reader handles/threads at 16 (sufficient to saturate NVMe queue depth)
    n_readers = std::min(n_readers, 16);

    files.resize(factory.sources.size());
    for (size_t i = 0; i < factory.sources.size(); ++i) {
        auto & f_workers = files[i];
        const auto & source = *factory.sources[i];
        f_workers.reserve(n_readers);
        for (int w = 0; w < n_readers; ++w) {
            auto handle = std::make_unique<llama_file>(source.name().c_str(), "rb", /*use_direct_io =*/ false);
            if (!source.same_file(*handle)) {
                throw std::runtime_error(format("model file changed while opening lazy reader: %s", source.name().c_str()));
            }
            for (const auto & [_, info] : factory.tensors) {
                if (info.file == i) {
                    handle->advise_random(info.offs, info.rsize * (size_t) info.nrows);
                }
            }
            f_workers.emplace_back(std::move(handle));
        }
    }

    worker_errs.resize(n_readers);
    worker_bounces.resize(n_readers);
    workers.reserve(n_readers - 1);
    for (int w = 1; w < n_readers; ++w) {
        workers.emplace_back(&llama_lazy_reader::worker_loop, this, w);
    }
}

llama_lazy_reader::~llama_lazy_reader() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        stop = true;
    }
    cv_work.notify_all();
    for (auto & t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }
}

void llama_lazy_reader::worker_loop(int w) {
    uint64_t seen_epoch = 0;
    while (true) {
        int n_active = 0;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv_work.wait(lock, [&]() { return stop || epoch != seen_epoch; });
            if (stop) {
                return;
            }
            seen_epoch = epoch;
            n_active = active_workers;
            if (w >= n_active) {
                continue;
            }
        }

        std::exception_ptr err;
        try {
            const int64_t r_begin = cur_n_runs * w / n_active;
            const int64_t r_end   = cur_n_runs * (w + 1) / n_active;
            if (r_begin < r_end) {
                read_runs(*cur_info, cur_pairs, cur_runs, r_begin, r_end, (size_t) w, cur_dst);
            }
        } catch (...) {
            err = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lock(mtx);
            worker_errs[w] = err;
            if (--remaining_workers == 0) {
                cv_done.notify_one();
            }
        }
    }
}

void llama_lazy_reader::read_runs(const llama_lazy_reader_factory::tensor_info & info,
                                  const std::pair<int32_t, int32_t> * pairs,
                                  const run_span * runs, int64_t begin, int64_t end,
                                  size_t fi, uint8_t * dst) const {
    auto & bounce = worker_bounces[fi];

    for (int64_t r = begin; r < end; ++r) {
        const auto & run = runs[r];
        const size_t file_off = info.offs + (size_t) run.first_row * info.rsize;

        if (run.n_unique_rows == 1) {
            // Read directly into the first destination slot, then memcpy to any duplicates
            uint8_t * first_dst = dst + (size_t) pairs[run.pair_begin].second * info.rsize;
            files[info.file][fi]->read_at(file_off, first_dst, info.rsize);
            for (int32_t k = run.pair_begin + 1; k < run.pair_end; ++k) {
                memcpy(dst + (size_t) pairs[k].second * info.rsize, first_dst, info.rsize);
            }
        } else {
            // Coalesced read of consecutive rows in a single pread call
            const size_t total_bytes = (size_t) run.n_unique_rows * info.rsize;
            if (bounce.size() < total_bytes) {
                bounce.resize(total_bytes);
            }
            files[info.file][fi]->read_at(file_off, bounce.data(), total_bytes);
            for (int32_t k = run.pair_begin; k < run.pair_end; ++k) {
                const size_t rel_row = (size_t) (pairs[k].first - run.first_row);
                memcpy(dst + (size_t) pairs[k].second * info.rsize,
                       bounce.data() + rel_row * info.rsize,
                       info.rsize);
            }
        }
    }
}

void llama_lazy_reader::gather(const ggml_tensor * tensor, const int32_t * rows, int64_t n, uint8_t * dst) const {
    if (n <= 0) {
        return;
    }

    const auto it = factory.tensors.find(tensor);
    GGML_ASSERT(it != factory.tensors.end());
    const auto & info = it->second;

    if (n == 1) {
        GGML_ASSERT(rows[0] >= 0 && (int64_t) rows[0] < info.nrows);
        files[info.file][0]->read_at(info.offs + (size_t) rows[0] * info.rsize, dst, info.rsize);
        return;
    }

    auto & pairs = scratch_pairs;
    pairs.clear();
    if ((int64_t) pairs.capacity() < n) {
        pairs.reserve(n);
    }
    for (int64_t i = 0; i < n; ++i) {
        GGML_ASSERT(rows[i] >= 0 && (int64_t) rows[i] < info.nrows);
        pairs.emplace_back(rows[i], (int32_t) i);
    }

    std::sort(pairs.begin(), pairs.end());

    // Build deduplicated & consecutive-row-coalesced runs
    auto & runs = scratch_runs;
    runs.clear();
    if ((int64_t) runs.capacity() < n) {
        runs.reserve(n);
    }
    for (int64_t i = 0; i < n; ) {
        const int32_t first_row = pairs[i].first;
        int32_t cur_row = first_row;
        int64_t j = i + 1;
        while (j < n) {
            if (pairs[j].first == cur_row) {
                ++j;
            } else if (pairs[j].first == cur_row + 1 && (cur_row + 1 - first_row) < 64) {
                cur_row = pairs[j].first;
                ++j;
            } else {
                break;
            }
        }
        runs.push_back({ first_row, cur_row - first_row + 1, (int32_t) i, (int32_t) j });
        i = j;
    }

    const int64_t n_runs = (int64_t) runs.size();
    const int max_workers = (int) files[info.file].size();
    // Use multiple persistent workers even for decode (e.g. 32 PLE heads -> 8 workers * 4 runs each)
    const int n_workers = (int) std::min<int64_t>(max_workers, std::max<int64_t>(1, (n_runs + 3) / 4));

    if (n_workers <= 1) {
        read_runs(info, pairs.data(), runs.data(), 0, n_runs, 0, dst);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        cur_info = &info;
        cur_pairs = pairs.data();
        cur_runs = runs.data();
        cur_n_runs = n_runs;
        cur_dst = dst;
        active_workers = n_workers;
        remaining_workers = n_workers - 1;
        for (int w = 0; w < n_workers; ++w) {
            worker_errs[w] = nullptr;
        }
        ++epoch;
    }
    cv_work.notify_all();

    try {
        read_runs(info, pairs.data(), runs.data(), 0, n_runs / n_workers, 0, dst);
    } catch (...) {
        worker_errs[0] = std::current_exception();
    }

    {
        std::unique_lock<std::mutex> lock(mtx);
        cv_done.wait(lock, [&]() { return remaining_workers == 0; });
    }

    for (int w = 0; w < n_workers; ++w) {
        if (worker_errs[w]) {
            std::rethrow_exception(worker_errs[w]);
        }
    }
}
