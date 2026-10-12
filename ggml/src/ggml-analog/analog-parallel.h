#pragma once

// Parallel execution of the array kernels.
//
// The passes of the array model are element-wise: every item of a pass has its own input range and
// its own output range, which is what makes the parallel result identical to the serial one. The
// helper below hands the items out to the threads of the build, the OpenMP runtime when the build
// has one (its threads stay alive between regions, and a model evaluation enters these kernels once
// per matrix multiplication), plain threads otherwise.

#include <algorithm>
#include <thread>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

// run fn(i0, i1) over n independent items, the ranges are disjoint and cover [0, n)
template <typename F>
inline void ggml_analog_parallel(int64_t n, int n_threads, F && fn) {
    if (n <= 0) {
        return;
    }

    if (n_threads <= 1 || n < 2) {
        fn((int64_t) 0, n);
        return;
    }

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(n_threads) if (n_threads > 1 && n >= 2*n_threads)
    for (int64_t i = 0; i < n; ++i) {
        fn(i, i + 1);
    }
#else
    const int64_t chunk = (n + n_threads - 1)/n_threads;

    std::vector<std::thread> workers;
    workers.reserve(n_threads - 1);

    for (int t = 0; t < n_threads; ++t) {
        const int64_t i0 = std::min<int64_t>(t*chunk, n);
        const int64_t i1 = std::min<int64_t>(i0 + chunk, n);

        if (i0 >= n) {
            break;
        }

        if (t == n_threads - 1) {
            fn(i0, i1);
        } else {
            workers.emplace_back([&fn, i0, i1]() { fn(i0, i1); });
        }
    }

    for (auto & w : workers) {
        w.join();
    }
#endif
}
