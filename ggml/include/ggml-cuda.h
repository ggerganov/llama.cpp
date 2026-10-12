#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// shareable buffers (CUDA VMM on Linux): allocations are physical segments with
// POSIX-fd handles that another CUDA process can import and map read-only, so
// several processes can share identical tensor data (e.g. the same weights in
// multiple model instances) with a single resident copy.
//
// A shareable buffer of size N spans exactly one segment of round_up(N, granularity)
// bytes. Buffers produced by the shareable buffer type are regular CUDA device
// buffers for all practical purposes (copies, tensor access); only their
// teardown and export paths differ.
//
// This feature requires GGML_USE_VMM and is only implemented for CUDA (not
// HIP/MUSA); on other platforms the calls below are safe no-ops/NULL.
//
// query whether a buffer came from the shareable buffer type and can be exported
GGML_BACKEND_API bool ggml_backend_cuda_buffer_is_shareable(ggml_backend_buffer_t buffer);

// export the POSIX fd and size (in bytes, granularity-rounded) of the segment backing a
// shareable buffer. the fd is owned by the caller and must be closed (or passed to
// ggml_backend_cuda_buffer_attach_shareable) afterwards. returns 0 on success.
GGML_BACKEND_API int ggml_backend_cuda_buffer_export_shareable(ggml_backend_buffer_t buffer, int * fd, size_t * segment_size);

// import a fd received from another process (e.g. over SCM_RIGHTS) and map it read-only
// on `device`. the fd is consumed on success. `segment_size` must match the size the
// exporter reported; `buffer_size` (<= segment_size) is the size reported by the returned
// buffer. returns NULL when the import or mapping fails.
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_cuda_buffer_attach_shareable(int device, int fd, size_t segment_size, size_t buffer_size);

// buffer type whose allocations share the interface of ggml_backend_cuda_buffer_type
// but allocate exportable one-segment-per-tensor buffers. tensors for which
// ggml_backend_cuda_shareable_buffer_type_attach_spec was called are backed by the
// foreign segment instead of a new allocation.
// returns NULL when VMM is unavailable (GGML_CUDA_NO_VMM builds) or the device has no
// virtual memory management support.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_shareable_buffer_type(int device);

// register an attach spec: when the shareable buffer type allocates the tensor named
// `tensor_name`, it imports and maps `fd` (segment_size as reported by the exporter)
// read-only instead of allocating. `tensor_name` is copied. returns false when `buft`
// is not a CUDA shareable buffer type.
GGML_BACKEND_API bool ggml_backend_cuda_shareable_buffer_type_attach_spec(ggml_backend_buffer_type_t buft, const char * tensor_name, int fd, size_t segment_size);

// PCI bus id of the physical GPU backing `device` (e.g. "0000:61:00.0"), written into
// `buf` and returned; NULL on failure. consumers of attachable segments compare it
// against the segment's registration token to refuse segments anchored on a different
// GPU (importing and mapping a foreign segment succeeds even cross-device, but reads
// then return garbage).
GGML_BACKEND_API const char * ggml_backend_cuda_shareable_device_bus_id(int device, char * buf, size_t buf_size);

// query whether `buffer` is backed by a foreign segment attached read-only via the
// attach path (as opposed to an owned allocation); false when the buffer is not a
// CUDA shareable buffer
GGML_BACKEND_API bool ggml_backend_cuda_buffer_is_attached_shareable(ggml_backend_buffer_t buffer);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
