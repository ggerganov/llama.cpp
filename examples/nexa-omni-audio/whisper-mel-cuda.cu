#define CUB_IGNORE_DEPRECATED_CPP_DIALECT
#include "whisper-mel-cuda.hpp"
#include "whisper.h"

#include "common.cuh"
#include <ggml-backend.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cub/device/device_reduce.cuh>
#include <device_launch_parameters.h>

#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(disable: 4324) // added padding
#endif

namespace {

static const char* cufftGetErrorString(cufftResult_t res) {
    switch (res) {
    case CUFFT_SUCCESS: return "The cuFFT operation was successful";
    case CUFFT_INVALID_PLAN: return "cuFFT was passed an invalid plan handle";
    case CUFFT_ALLOC_FAILED: return "cuFFT failed to allocate GPU or CPU memory";
    case CUFFT_INVALID_TYPE: return "No longer used";
    case CUFFT_INVALID_VALUE: return "User specified an invalid pointer or parameter";
    case CUFFT_INTERNAL_ERROR: return "Driver or internal cuFFT library error";
    case CUFFT_EXEC_FAILED: return "Failed to execute an FFT on the GPU";
    case CUFFT_SETUP_FAILED: return "The cuFFT library failed to initialize";
    case CUFFT_INVALID_SIZE: return "User specified an invalid transform size";
    case CUFFT_UNALIGNED_DATA: return "No longer used";
    case CUFFT_INCOMPLETE_PARAMETER_LIST: return "Missing parameters in call";
    case CUFFT_INVALID_DEVICE: return "Execution of a plan was on different GPU than plan creation";
    case CUFFT_PARSE_ERROR: return "Internal plan database error";
    case CUFFT_NO_WORKSPACE: return "No workspace has been provided prior to plan execution";
    case CUFFT_NOT_IMPLEMENTED: return "Function does not implement functionality for parameters given.";
    case CUFFT_LICENSE_ERROR: return "Used in previous versions.";
    case CUFFT_NOT_SUPPORTED: return "Operation is not supported for parameters given.";
    default: return "Unknown error";
    }
}

#define CUFFT_CHECK(err) CUDA_CHECK_GEN(err, CUFFT_SUCCESS, cufftGetErrorString)

__global__ void k_fill_stft_input(
    const float * padded_samples,
    const int n_frames,
    const float * hann_window,
    float * stft_in
) {
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    // if (y >= n_frames) return;
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    // if (x >= WHISPER_N_FFT) return;

    auto line = padded_samples + y * WHISPER_HOP_LENGTH;
    auto outLine = stft_in + y * WHISPER_N_FFT;

    outLine[x] = line[x] * hann_window[x];
}

__global__ void k_calc_magnitudes(
    const cuComplex * stft_out,
    const int n_frames,
    float * magnitudes
) {
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    // if (y >= n_frames) return;
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    // if (x >= WHISPER_N_FFT_HALF) return;

    auto idx = y * WHISPER_N_FFT_HALF + x;

    auto r = stft_out[idx].x;
    auto i = stft_out[idx].y;
    magnitudes[idx] = r * r + i * i;
}

__global__ void k_calc_log_mel(
    const float * mel_data,
    const int n_mel,
    const float * max_val,
    float * log_mel
) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n_mel) return;

    float val = mel_data[x];

    constexpr float e = 1e-10f;
    if (val < e) val = e;

    val = log10(val);

    const float max = log10(*max_val) - 8.f;
    if (val < max) val = max;

    log_mel[x] = (val + 4) / 4;
}

static void fill_stft_input(
    const float * padded_samples,
    int n_frames,
    const float * hann_window,
    float * stft_in,
    cudaStream_t stream
) {
    dim3 block(WHISPER_N_FFT, 1);
    dim3 grid(1, n_frames);

    k_fill_stft_input<<<grid, block, 0, stream>>>(padded_samples, n_frames, hann_window, stft_in);
}

static void calc_magnitudes(
    const cuComplex * stft_out,
    int n_frames,
    float * magnitudes,
    cudaStream_t stream
) {
    dim3 block(WHISPER_N_FFT_HALF, 1);
    dim3 grid(1, n_frames);
    k_calc_magnitudes<<<grid, block, 0, stream>>>(stft_out, n_frames, magnitudes);
}

constexpr auto LOG_MEL_PREFIX_SIZE = 256;

static void calc_log_mel(
    const float * mel_data,
    int n_mel,
    void * tempStorage,
    int tempStorageSize,
    float * log_mel,
    cudaStream_t stream
) {
    float * max_val = reinterpret_cast<float *>(tempStorage);
    void * maxTemp = reinterpret_cast<char*>(tempStorage) + LOG_MEL_PREFIX_SIZE;

    size_t nbytes = size_t(tempStorageSize - LOG_MEL_PREFIX_SIZE);
    cub::DeviceReduce::Max(maxTemp, nbytes, mel_data, max_val, n_mel, stream);

    int block = 256;
    int grid = (n_mel + block - 1) / block;

    k_calc_log_mel<<<grid, block, 0, stream>>>(mel_data, n_mel, max_val, log_mel);
}

class mel_calc_cuda : public whisper_mel_calc {
    const int m_n_mel;

    ggml_backend_t m_backend = nullptr;
    int m_device = -1;

    cudaStream_t m_stream = nullptr;
    cublasHandle_t m_cublas_handle = nullptr;

    float * m_hann_window = nullptr;

    float * m_filters = nullptr;

    // max samples for which we have allocated memory for the temp working areas below (cufft, log_mel)
    int m_n_max_samples = 0;

    size_t m_cufft_workspace_size = 0;
    void * m_cufft_workspace = nullptr;

    size_t m_log_mel_temp_storage_size = 0;
    void * m_log_mel_temp_storage = nullptr;
public:
    mel_calc_cuda(ggml_backend_t backend, const whisper_filters & filters)
        : m_n_mel(filters.n_mel)
        , m_backend(backend)
    {
        ggml_backend_cuda_context* cuda_ctx = (ggml_backend_cuda_context*) m_backend->context;
        m_device = cuda_ctx->device;

        if (ggml_cuda_info().devices[m_device].cc < 600) {
            // we've only tesed on 6.0 and higher and we've had reports of crashes on 5.0:
            // https://github.com/ggerganov/whisper.cpp/issues/2230
            // to be safe forbid anything below 6.0
            throw std::runtime_error("CUDA compute capability 6.0 or higher is required");
        }

        ggml_cuda_set_device(m_device);

        if (filters.n_fft != WHISPER_N_FFT_HALF) {
            throw std::invalid_argument("MelFilters n_frames must be WHISPER_N_FFT_HALF");
        }
        assert(filters.data.size() == filters.n_mel * WHISPER_N_FFT_HALF);

        CUDA_CHECK(cudaStreamCreate(&m_stream));
        CUBLAS_CHECK(cublasCreate(&m_cublas_handle));
        CUBLAS_CHECK(cublasSetMathMode(m_cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
        CUBLAS_CHECK(cublasSetStream(m_cublas_handle, m_stream));

        // create Hann window
        {
            auto hw = whisper_mel_calc::hann_window();
            CUDA_CHECK(cudaMallocAsync(&m_hann_window, hw.len * sizeof(float), m_stream));
            CUDA_CHECK(cudaMemcpyAsync(m_hann_window, hw.data, hw.len * sizeof(float), cudaMemcpyHostToDevice, m_stream));
        }

        // fill filters
        {
            auto& f = filters.data;
            CUDA_CHECK(cudaMallocAsync(&m_filters, f.size() * sizeof(float), m_stream));
            CUDA_CHECK(cudaMemcpyAsync(m_filters, f.data(), f.size() * sizeof(float), cudaMemcpyHostToDevice, m_stream));
        }

        // preallocate working areas enough for the most common cases (<= 30s)
        ensure_working_areas(WHISPER_N_SAMPLES);
    }

    ~mel_calc_cuda() {
        ggml_cuda_set_device(m_device);
        CUDA_CHECK(cudaStreamSynchronize(m_stream));
        CUDA_CHECK(cudaStreamDestroy(m_stream));
        CUDA_CHECK(cudaFree(m_hann_window));
        CUDA_CHECK(cudaFree(m_cufft_workspace));
        CUDA_CHECK(cudaFree(m_filters));
        CUDA_CHECK(cudaFree(m_log_mel_temp_storage));
    }

    void ensure_working_areas(int n_samples) {
        if (n_samples <= m_n_max_samples) {
            return;
        }

        const auto max_padded_samples = n_samples + WHISPER_N_SAMPLES + WHISPER_N_FFT;
        const auto max_frames = 1 + (max_padded_samples - WHISPER_N_FFT) / WHISPER_HOP_LENGTH;

        // cufft workspace
        {
            if (m_cufft_workspace) {
                CUDA_CHECK(cudaFree(m_cufft_workspace));
                m_cufft_workspace_size = 0;
                m_cufft_workspace = nullptr;
            }
            CUFFT_CHECK(cufftEstimate1d(WHISPER_N_FFT, CUFFT_R2C, max_frames, &m_cufft_workspace_size));
            CUDA_CHECK(cudaMallocAsync(&m_cufft_workspace, m_cufft_workspace_size, m_stream));
        }

        // device reduce working area
        {
            if (m_log_mel_temp_storage) {
                CUDA_CHECK(cudaFree(m_log_mel_temp_storage));
                m_log_mel_temp_storage_size = 0;
                m_log_mel_temp_storage = nullptr;
            }

            const auto max_mels = 160;

            size_t nbytes = 0;
            float* temp = nullptr;
            cub::DeviceReduce::Max(nullptr, nbytes, temp, temp, max_frames * max_mels);
            m_log_mel_temp_storage_size = nbytes + LOG_MEL_PREFIX_SIZE;

            CUDA_CHECK(cudaMallocAsync(&m_log_mel_temp_storage, m_log_mel_temp_storage_size, m_stream));
        }

        m_n_max_samples = n_samples;
    }

    virtual whisper_mel calculate(whisper_span<const float> samples, int /*n_threads*/) override {
        ggml_cuda_set_device(m_device);
        ensure_working_areas(samples.len);

        const size_t mirror_pad = WHISPER_N_FFT / 2;
        const size_t padded_size = samples.len + WHISPER_N_SAMPLES + WHISPER_N_FFT;

        // pad
        std::vector<float> padded_samples(padded_size);
        std::reverse_copy(samples.data + 1, samples.data + 1 + mirror_pad, padded_samples.begin()); // reflect
        std::copy(samples.data, samples.data + samples.len, padded_samples.begin() + mirror_pad); // copy

        // fill the rest of the data
        // it should canonically be mirrored at the end as well,
        // but we just assume the last MEL_FRAME_SIZE/2 samples are zeros
        std::fill(padded_samples.begin() + mirror_pad + samples.len, padded_samples.end(), 0.f);

        const auto n_frames = 1 + (padded_samples.size() - WHISPER_N_FFT) / WHISPER_HOP_LENGTH;

        float * cu_padded_samples = nullptr;
        CUDA_CHECK(cudaMallocAsync(&cu_padded_samples, padded_samples.size() * sizeof(float), m_stream));
        CUDA_CHECK(cudaMemcpyAsync(cu_padded_samples, padded_samples.data(), padded_samples.size() * sizeof(float), cudaMemcpyHostToDevice, m_stream));

        float * stft_in = nullptr; // contiguous buffer for stft input
        CUDA_CHECK(cudaMallocAsync(&stft_in, n_frames * WHISPER_N_FFT * sizeof(float), m_stream));

        fill_stft_input(cu_padded_samples, int(n_frames), m_hann_window, stft_in, m_stream);

        cufftComplex* stft_out;
        CUDA_CHECK(cudaMallocAsync(&stft_out, n_frames * WHISPER_N_FFT_HALF * sizeof(cufftComplex), m_stream));

        cufftHandle plan;
        CUFFT_CHECK(cufftCreate(&plan));
        CUFFT_CHECK(cufftSetAutoAllocation(plan, 0));
        {
            size_t waSize;
            CUFFT_CHECK(cufftMakePlan1d(plan, WHISPER_N_FFT, CUFFT_R2C, int(n_frames), &waSize));
            assert(waSize <= m_cufft_workspace_size);
            CUFFT_CHECK(cufftSetWorkArea(plan, m_cufft_workspace));
            CUFFT_CHECK(cufftSetStream(plan, m_stream));
        }
        CUFFT_CHECK(cufftExecR2C(plan, stft_in, stft_out));

        const auto n_mag_frames = n_frames - 1; // drop last frame
        float * magnitudes;
        CUDA_CHECK(cudaMallocAsync(&magnitudes, n_mag_frames * WHISPER_N_FFT_HALF * sizeof(float), m_stream));
        calc_magnitudes(stft_out, int(n_mag_frames), magnitudes, m_stream);

        float * mel_data = nullptr;
        CUDA_CHECK(cudaMallocAsync(&mel_data, m_n_mel * n_mag_frames * sizeof(float), m_stream));

        const float fone = 1.0f, fzero = 0.0f;
        CUBLAS_CHECK(cublasSgemm(m_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            int(n_mag_frames), m_n_mel, WHISPER_N_FFT_HALF,
            &fone,
            magnitudes, WHISPER_N_FFT_HALF,
            m_filters, WHISPER_N_FFT_HALF,
            &fzero,
            mel_data, int(n_mag_frames)));

        whisper_mel ret;
        // Calculate semi-padded sample length to ensure compatibility
        int n_len_org = 1 + int(samples.len + mirror_pad - WHISPER_N_FFT) / WHISPER_HOP_LENGTH;
        whisper_mel_init(ret, m_backend, int(n_mag_frames), n_len_org, m_n_mel);
        assert(ggml_nbytes(ret.tensor) == m_n_mel * n_mag_frames * sizeof(float));

        float* log_mels = reinterpret_cast<float*>(ret.tensor->data);

        calc_log_mel(
            mel_data, int(m_n_mel * n_mag_frames),
            m_log_mel_temp_storage , int(m_log_mel_temp_storage_size),
            log_mels, m_stream);

        CUDA_CHECK(cudaStreamSynchronize(m_stream));

        // cleanup
        CUFFT_CHECK(cufftDestroy(plan));
        CUDA_CHECK(cudaFreeAsync(mel_data, m_stream));
        CUDA_CHECK(cudaFreeAsync(magnitudes, m_stream));
        CUDA_CHECK(cudaFreeAsync(stft_out, m_stream));
        CUDA_CHECK(cudaFreeAsync(stft_in, m_stream));
        CUDA_CHECK(cudaFreeAsync(cu_padded_samples, m_stream));

        return ret;
    }
};

}

whisper_mel_calc * whisper_mel_calc_create_cuda(ggml_backend_t backend, const whisper_filters & filters) {
    try {
        return new mel_calc_cuda(backend, filters);
    }
    catch (...) {
        // TODO: log error (but for this we would have to expose the log state to be accessible here)
        return nullptr;
    }
}
