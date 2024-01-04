#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
// typedef sycl::half ggml_fp16_t;

#define CHECK_TRY_ERROR(expr)                                                  \
  [&]() {                                                                      \
    try {                                                                      \
      expr;                                                                    \
      return dpct::success;                                                    \
    } catch (std::exception const &e) {                                        \
      std::cerr << e.what()<< "\nException caught at file:" << __FILE__        \
        << ", line:" << __LINE__ <<", func:"<<__func__<< std::endl;            \
      return dpct::default_error;                                              \
    }                                                                          \
  }()

#define DEBUG_CUDA_MALLOC