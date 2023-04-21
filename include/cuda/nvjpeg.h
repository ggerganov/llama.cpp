/*
 * Copyright 2009-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

  
#ifndef NV_JPEG_HEADER
#define NV_JPEG_HEADER

#define NVJPEGAPI


#include "cuda_runtime_api.h"
#include "library_types.h"

#include "stdint.h"

#if defined(__cplusplus)
  extern "C" {
#endif

// Maximum number of channels nvjpeg decoder supports
#define NVJPEG_MAX_COMPONENT 4

// nvjpeg version information
#define NVJPEG_VER_MAJOR 11
#define NVJPEG_VER_MINOR 6
#define NVJPEG_VER_PATCH 2
#define NVJPEG_VER_BUILD 124

/* nvJPEG status enums, returned by nvJPEG API */
typedef enum
{
    NVJPEG_STATUS_SUCCESS                       = 0,
    NVJPEG_STATUS_NOT_INITIALIZED               = 1,
    NVJPEG_STATUS_INVALID_PARAMETER             = 2,
    NVJPEG_STATUS_BAD_JPEG                      = 3,
    NVJPEG_STATUS_JPEG_NOT_SUPPORTED            = 4,
    NVJPEG_STATUS_ALLOCATOR_FAILURE             = 5,
    NVJPEG_STATUS_EXECUTION_FAILED              = 6,
    NVJPEG_STATUS_ARCH_MISMATCH                 = 7,
    NVJPEG_STATUS_INTERNAL_ERROR                = 8,
    NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED  = 9,
    NVJPEG_STATUS_INCOMPLETE_BITSTREAM          = 10
} nvjpegStatus_t;


// Enums for EXIF Orientation
typedef enum nvjpegExifOrientation
{
    NVJPEG_ORIENTATION_UNKNOWN = 0,
    NVJPEG_ORIENTATION_NORMAL = 1,
    NVJPEG_ORIENTATION_FLIP_HORIZONTAL = 2,
    NVJPEG_ORIENTATION_ROTATE_180 = 3,
    NVJPEG_ORIENTATION_FLIP_VERTICAL = 4,
    NVJPEG_ORIENTATION_TRANSPOSE = 5,
    NVJPEG_ORIENTATION_ROTATE_90 = 6,
    NVJPEG_ORIENTATION_TRANSVERSE = 7,
    NVJPEG_ORIENTATION_ROTATE_270 = 8
} nvjpegExifOrientation_t;

// Enum identifies image chroma subsampling values stored inside JPEG input stream
// In the case of NVJPEG_CSS_GRAY only 1 luminance channel is encoded in JPEG input stream
// Otherwise both chroma planes are present
typedef enum
{
    NVJPEG_CSS_444 = 0,
    NVJPEG_CSS_422 = 1,
    NVJPEG_CSS_420 = 2,
    NVJPEG_CSS_440 = 3,
    NVJPEG_CSS_411 = 4,
    NVJPEG_CSS_410 = 5,
    NVJPEG_CSS_GRAY = 6,
    NVJPEG_CSS_410V = 7,
    NVJPEG_CSS_UNKNOWN = -1
} nvjpegChromaSubsampling_t;

// Parameter of this type specifies what type of output user wants for image decoding
typedef enum
{
    // return decompressed image as it is - write planar output
    NVJPEG_OUTPUT_UNCHANGED   = 0,
    // return planar luma and chroma, assuming YCbCr colorspace
    NVJPEG_OUTPUT_YUV         = 1, 
    // return luma component only, if YCbCr colorspace, 
    // or try to convert to grayscale,
    // writes to 1-st channel of nvjpegImage_t
    NVJPEG_OUTPUT_Y           = 2,
    // convert to planar RGB 
    NVJPEG_OUTPUT_RGB         = 3,
    // convert to planar BGR
    NVJPEG_OUTPUT_BGR         = 4, 
    // convert to interleaved RGB and write to 1-st channel of nvjpegImage_t
    NVJPEG_OUTPUT_RGBI        = 5, 
    // convert to interleaved BGR and write to 1-st channel of nvjpegImage_t
    NVJPEG_OUTPUT_BGRI        = 6,
    // maximum allowed value
    NVJPEG_OUTPUT_FORMAT_MAX  = 6  
} nvjpegOutputFormat_t;

// Parameter of this type specifies what type of input user provides for encoding
typedef enum
{
    NVJPEG_INPUT_RGB         = 3, // Input is RGB - will be converted to YCbCr before encoding
    NVJPEG_INPUT_BGR         = 4, // Input is RGB - will be converted to YCbCr before encoding
    NVJPEG_INPUT_RGBI        = 5, // Input is interleaved RGB - will be converted to YCbCr before encoding
    NVJPEG_INPUT_BGRI        = 6  // Input is interleaved RGB - will be converted to YCbCr before encoding
} nvjpegInputFormat_t;

// Implementation
// NVJPEG_BACKEND_DEFAULT    : default value
// NVJPEG_BACKEND_HYBRID     : uses CPU for Huffman decode
// NVJPEG_BACKEND_GPU_HYBRID : uses GPU assisted Huffman decode. nvjpegDecodeBatched will use GPU decoding for baseline JPEG bitstreams with
//                             interleaved scan when batch size is bigger than 100
// NVJPEG_BACKEND_HARDWARE   : supports baseline JPEG bitstream with single scan. 410 and 411 sub-samplings are not supported
// NVJPEG_BACKEND_GPU_HYBRID_DEVICE : nvjpegDecodeBatched will support bitstream input on device memory
// NVJPEG_BACKEND_HARDWARE_DEVICE   : nvjpegDecodeBatched will support bitstream input on device memory
typedef enum 
{
    NVJPEG_BACKEND_DEFAULT = 0,
    NVJPEG_BACKEND_HYBRID  = 1,
    NVJPEG_BACKEND_GPU_HYBRID = 2,
    NVJPEG_BACKEND_HARDWARE = 3,
    NVJPEG_BACKEND_GPU_HYBRID_DEVICE = 4,
    NVJPEG_BACKEND_HARDWARE_DEVICE = 5
} nvjpegBackend_t;

// Currently parseable JPEG encodings (SOF markers)
typedef enum
{
    NVJPEG_ENCODING_UNKNOWN                                 = 0x0,

    NVJPEG_ENCODING_BASELINE_DCT                            = 0xc0,
    NVJPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN         = 0xc1,
    NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN                 = 0xc2

} nvjpegJpegEncoding_t;

typedef enum 
{
    NVJPEG_SCALE_NONE = 0, // decoded output is not scaled 
    NVJPEG_SCALE_1_BY_2 = 1, // decoded output width and height is scaled by a factor of 1/2
    NVJPEG_SCALE_1_BY_4 = 2, // decoded output width and height is scaled by a factor of 1/4
    NVJPEG_SCALE_1_BY_8 = 3, // decoded output width and height is scaled by a factor of 1/8
} nvjpegScaleFactor_t;

#define NVJPEG_FLAGS_DEFAULT 0
#define NVJPEG_FLAGS_HW_DECODE_NO_PIPELINE 1
#define NVJPEG_FLAGS_ENABLE_MEMORY_POOLS   1<<1
#define NVJPEG_FLAGS_BITSTREAM_STRICT  1<<2

// Output descriptor.
// Data that is written to planes depends on output format
typedef struct
{
    unsigned char * channel[NVJPEG_MAX_COMPONENT];
    size_t    pitch[NVJPEG_MAX_COMPONENT];
} nvjpegImage_t;

// Prototype for device memory allocation, modelled after cudaMalloc()
typedef int (*tDevMalloc)(void**, size_t);
// Prototype for device memory release
typedef int (*tDevFree)(void*);

// Prototype for pinned memory allocation, modelled after cudaHostAlloc()
typedef int (*tPinnedMalloc)(void**, size_t, unsigned int flags);
// Prototype for device memory release
typedef int (*tPinnedFree)(void*);

// Memory allocator using mentioned prototypes, provided to nvjpegCreateEx
// This allocator will be used for all device memory allocations inside library
// In any way library is doing smart allocations (reallocates memory only if needed)
typedef struct 
{
    tDevMalloc dev_malloc;
    tDevFree dev_free;
} nvjpegDevAllocator_t;

// Pinned memory allocator using mentioned prototypes, provided to nvjpegCreate
// This allocator will be used for all pinned host memory allocations inside library
// In any way library is doing smart allocations (reallocates memory only if needed)
typedef struct 
{
    tPinnedMalloc pinned_malloc;
    tPinnedFree pinned_free;
} nvjpegPinnedAllocator_t;

// Opaque library handle identifier.
struct nvjpegHandle;
typedef struct nvjpegHandle* nvjpegHandle_t;

// Opaque jpeg decoding state handle identifier - used to store intermediate information between deccding phases
struct nvjpegJpegState;
typedef struct nvjpegJpegState* nvjpegJpegState_t;

// returns library's property values, such as MAJOR_VERSION, MINOR_VERSION or PATCH_LEVEL
nvjpegStatus_t NVJPEGAPI nvjpegGetProperty(libraryPropertyType type, int *value);
// returns CUDA Toolkit property values that was used for building library, 
// such as MAJOR_VERSION, MINOR_VERSION or PATCH_LEVEL
nvjpegStatus_t NVJPEGAPI nvjpegGetCudartProperty(libraryPropertyType type, int *value);

// Initalization of nvjpeg handle. This handle is used for all consecutive calls
// IN         backend       : Backend to use. Currently Default or Hybrid (which is the same at the moment) is supported.
// IN         allocator     : Pointer to nvjpegDevAllocator. If NULL - use default cuda calls (cudaMalloc/cudaFree)
// INT/OUT    handle        : Codec instance, use for other calls
nvjpegStatus_t NVJPEGAPI nvjpegCreate(nvjpegBackend_t backend, nvjpegDevAllocator_t *dev_allocator, nvjpegHandle_t *handle);

// Initalization of nvjpeg handle with default backend and default memory allocators.
// INT/OUT    handle        : Codec instance, use for other calls
nvjpegStatus_t NVJPEGAPI nvjpegCreateSimple(nvjpegHandle_t *handle);

// Initalization of nvjpeg handle with additional parameters. This handle is used for all consecutive nvjpeg calls
// IN         backend       : Backend to use. Currently Default or Hybrid (which is the same at the moment) is supported.
// IN         dev_allocator : Pointer to nvjpegDevAllocator. If NULL - use default cuda calls (cudaMalloc/cudaFree)
// IN         pinned_allocator : Pointer to nvjpegPinnedAllocator. If NULL - use default cuda calls (cudaHostAlloc/cudaFreeHost)
// IN         flags         : Parameters for the operation. Must be 0.
// INT/OUT    handle        : Codec instance, use for other calls
nvjpegStatus_t NVJPEGAPI nvjpegCreateEx(nvjpegBackend_t backend, 
        nvjpegDevAllocator_t *dev_allocator, 
        nvjpegPinnedAllocator_t *pinned_allocator, 
        unsigned int flags,
        nvjpegHandle_t *handle);

// Release the handle and resources.
// IN/OUT     handle: instance handle to release 
nvjpegStatus_t NVJPEGAPI nvjpegDestroy(nvjpegHandle_t handle);

// Sets padding for device memory allocations. After success on this call any device memory allocation
// would be padded to the multiple of specified number of bytes. 
// IN         padding: padding size
// IN/OUT     handle: instance handle to release 
nvjpegStatus_t NVJPEGAPI nvjpegSetDeviceMemoryPadding(size_t padding, nvjpegHandle_t handle);

// Retrieves padding for device memory allocations
// IN/OUT     padding: padding size currently used in handle.
// IN/OUT     handle: instance handle to release 
nvjpegStatus_t NVJPEGAPI nvjpegGetDeviceMemoryPadding(size_t *padding, nvjpegHandle_t handle);

// Sets padding for pinned host memory allocations. After success on this call any pinned host memory allocation
// would be padded to the multiple of specified number of bytes. 
// IN         padding: padding size
// IN/OUT     handle: instance handle to release 
nvjpegStatus_t NVJPEGAPI nvjpegSetPinnedMemoryPadding(size_t padding, nvjpegHandle_t handle);

// Retrieves padding for pinned host memory allocations
// IN/OUT     padding: padding size currently used in handle.
// IN/OUT     handle: instance handle to release 
nvjpegStatus_t NVJPEGAPI nvjpegGetPinnedMemoryPadding(size_t *padding, nvjpegHandle_t handle);



// Initalization of decoding state
// IN         handle        : Library handle
// INT/OUT    jpeg_handle   : Decoded jpeg image state handle
nvjpegStatus_t NVJPEGAPI nvjpegJpegStateCreate(nvjpegHandle_t handle, nvjpegJpegState_t *jpeg_handle);

// Release the jpeg image handle.
// INT/OUT    jpeg_handle   : Decoded jpeg image state handle
nvjpegStatus_t NVJPEGAPI nvjpegJpegStateDestroy(nvjpegJpegState_t jpeg_handle);
// 
// Retrieve the image info, including channel, width and height of each component, and chroma subsampling.
// If less than NVJPEG_MAX_COMPONENT channels are encoded, then zeros would be set to absent channels information
// If the image is 3-channel, all three groups are valid.
// This function is thread safe.
// IN         handle      : Library handle
// IN         data        : Pointer to the buffer containing the jpeg stream data to be decoded. 
// IN         length      : Length of the jpeg image buffer.
// OUT        nComponent  : Number of componenets of the image, currently only supports 1-channel (grayscale) or 3-channel.
// OUT        subsampling : Chroma subsampling used in this JPEG, see nvjpegChromaSubsampling_t
// OUT        widths      : pointer to NVJPEG_MAX_COMPONENT of ints, returns width of each channel. 0 if channel is not encoded  
// OUT        heights     : pointer to NVJPEG_MAX_COMPONENT of ints, returns height of each channel. 0 if channel is not encoded 
nvjpegStatus_t NVJPEGAPI nvjpegGetImageInfo(
        nvjpegHandle_t handle,
        const unsigned char *data, 
        size_t length,
        int *nComponents, 
        nvjpegChromaSubsampling_t *subsampling,
        int *widths,
        int *heights);
                   

// Decodes single image. The API is back-end agnostic. It will decide on which implementation to use internally
// Destination buffers should be large enough to be able to store  output of specified format.
// For each color plane sizes could be retrieved for image using nvjpegGetImageInfo()
// and minimum required memory buffer for each plane is nPlaneHeight*nPlanePitch where nPlanePitch >= nPlaneWidth for
// planar output formats and nPlanePitch >= nPlaneWidth*nOutputComponents for interleaved output format.
// 
// IN/OUT     handle        : Library handle
// INT/OUT    jpeg_handle   : Decoded jpeg image state handle
// IN         data          : Pointer to the buffer containing the jpeg image to be decoded. 
// IN         length        : Length of the jpeg image buffer.
// IN         output_format : Output data format. See nvjpegOutputFormat_t for description
// IN/OUT     destination   : Pointer to structure with information about output buffers. See nvjpegImage_t description.
// IN/OUT     stream        : CUDA stream where to submit all GPU work
// 
// \return NVJPEG_STATUS_SUCCESS if successful
nvjpegStatus_t NVJPEGAPI nvjpegDecode(
        nvjpegHandle_t handle,
        nvjpegJpegState_t jpeg_handle,
        const unsigned char *data,
        size_t length, 
        nvjpegOutputFormat_t output_format,
        nvjpegImage_t *destination,
        cudaStream_t stream);


//////////////////////////////////////////////
/////////////// Batch decoding ///////////////
//////////////////////////////////////////////

// Resets and initizlizes batch decoder for working on the batches of specified size
// Should be called once for decoding bathes of this specific size, also use to reset failed batches
// IN/OUT     handle          : Library handle
// INT/OUT    jpeg_handle     : Decoded jpeg image state handle
// IN         batch_size      : Size of the batch
// IN         max_cpu_threads : Maximum number of CPU threads that will be processing this batch
// IN         output_format   : Output data format. Will be the same for every image in batch
//
// \return NVJPEG_STATUS_SUCCESS if successful
nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedInitialize(
          nvjpegHandle_t handle,
          nvjpegJpegState_t jpeg_handle,
          int batch_size,
          int max_cpu_threads,
          nvjpegOutputFormat_t output_format);

// Decodes batch of images. Output buffers should be large enough to be able to store 
// outputs of specified format, see single image decoding description for details. Call to 
// nvjpegDecodeBatchedInitialize() is required prior to this call, batch size is expected to be the same as 
// parameter to this batch initialization function.
// 
// IN/OUT     handle        : Library handle
// INT/OUT    jpeg_handle   : Decoded jpeg image state handle
// IN         data          : Array of size batch_size of pointers to the input buffers containing the jpeg images to be decoded. 
// IN         lengths       : Array of size batch_size with lengths of the jpeg images' buffers in the batch.
// IN/OUT     destinations  : Array of size batch_size with pointers to structure with information about output buffers, 
// IN/OUT     stream        : CUDA stream where to submit all GPU work
// 
// \return NVJPEG_STATUS_SUCCESS if successful
nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatched(
          nvjpegHandle_t handle,
          nvjpegJpegState_t jpeg_handle,
          const unsigned char *const *data,
          const size_t *lengths, 
          nvjpegImage_t *destinations,
          cudaStream_t stream);

// Allocates the internal buffers as a pre-allocation step
// IN    handle          : Library handle
// IN    jpeg_handle     : Decoded jpeg image state handle
// IN    width   : frame width
// IN    height  : frame height
// IN    chroma_subsampling   : chroma subsampling of images to be decoded
// IN    output_format : out format

nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedPreAllocate(
          nvjpegHandle_t handle,
          nvjpegJpegState_t jpeg_handle,
          int batch_size,
          int width,
          int height,
          nvjpegChromaSubsampling_t chroma_subsampling,
          nvjpegOutputFormat_t output_format);


// Allocates the internal buffers as a pre-allocation step
// IN    handle          : Library handle
// IN    jpeg_handle     : Decoded jpeg image state handle
// IN    data            : jpeg bitstream containing huffman and quantization tables
// IN    length          : bitstream size in bytes

nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedParseJpegTables(
          nvjpegHandle_t handle,
          nvjpegJpegState_t jpeg_handle,
          const unsigned char *data,
          const size_t length);

/**********************************************************
*                        Compression                      *
**********************************************************/
struct nvjpegEncoderState;
typedef struct nvjpegEncoderState* nvjpegEncoderState_t;

nvjpegStatus_t NVJPEGAPI nvjpegEncoderStateCreate(
        nvjpegHandle_t handle,
        nvjpegEncoderState_t *encoder_state,
        cudaStream_t stream);

nvjpegStatus_t NVJPEGAPI nvjpegEncoderStateDestroy(nvjpegEncoderState_t encoder_state);

struct nvjpegEncoderParams;
typedef struct nvjpegEncoderParams* nvjpegEncoderParams_t;

nvjpegStatus_t NVJPEGAPI nvjpegEncoderParamsCreate(
        nvjpegHandle_t handle, 
        nvjpegEncoderParams_t *encoder_params,
        cudaStream_t stream);

nvjpegStatus_t NVJPEGAPI nvjpegEncoderParamsDestroy(nvjpegEncoderParams_t encoder_params);

nvjpegStatus_t NVJPEGAPI nvjpegEncoderParamsSetQuality(
        nvjpegEncoderParams_t encoder_params,
        const int quality,
        cudaStream_t stream);

nvjpegStatus_t NVJPEGAPI nvjpegEncoderParamsSetEncoding(
        nvjpegEncoderParams_t encoder_params,
        nvjpegJpegEncoding_t etype,
        cudaStream_t stream);

nvjpegStatus_t NVJPEGAPI nvjpegEncoderParamsSetOptimizedHuffman(
        nvjpegEncoderParams_t encoder_params,
        const int optimized,
        cudaStream_t stream);

nvjpegStatus_t NVJPEGAPI nvjpegEncoderParamsSetSamplingFactors(
        nvjpegEncoderParams_t encoder_params,
        const nvjpegChromaSubsampling_t chroma_subsampling,
        cudaStream_t stream);

nvjpegStatus_t NVJPEGAPI nvjpegEncodeGetBufferSize(
        nvjpegHandle_t handle,
        const nvjpegEncoderParams_t encoder_params,
        int image_width,
        int image_height,
        size_t *max_stream_length);

nvjpegStatus_t NVJPEGAPI nvjpegEncodeYUV(
        nvjpegHandle_t handle,
        nvjpegEncoderState_t encoder_state,
        const nvjpegEncoderParams_t encoder_params,
        const nvjpegImage_t *source,
        nvjpegChromaSubsampling_t chroma_subsampling, 
        int image_width,
        int image_height,
        cudaStream_t stream);

nvjpegStatus_t NVJPEGAPI nvjpegEncodeImage(
        nvjpegHandle_t handle,
        nvjpegEncoderState_t encoder_state,
        const nvjpegEncoderParams_t encoder_params,
        const nvjpegImage_t *source,
        nvjpegInputFormat_t input_format, 
        int image_width,
        int image_height,
        cudaStream_t stream);

nvjpegStatus_t NVJPEGAPI nvjpegEncodeRetrieveBitstreamDevice(
        nvjpegHandle_t handle,
        nvjpegEncoderState_t encoder_state,
        unsigned char *data,
        size_t *length,
        cudaStream_t stream);

nvjpegStatus_t NVJPEGAPI nvjpegEncodeRetrieveBitstream(
        nvjpegHandle_t handle,
        nvjpegEncoderState_t encoder_state,
        unsigned char *data,
        size_t *length,
        cudaStream_t stream);

///////////////////////////////////////////////////////////////////////////////////
// API v2 //
///////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////
// NVJPEG buffers //
///////////////////////////////////////////////////////////////////////////////////

struct nvjpegBufferPinned;
typedef struct nvjpegBufferPinned* nvjpegBufferPinned_t;

nvjpegStatus_t NVJPEGAPI nvjpegBufferPinnedCreate(nvjpegHandle_t handle, 
    nvjpegPinnedAllocator_t* pinned_allocator,
    nvjpegBufferPinned_t* buffer);

nvjpegStatus_t NVJPEGAPI nvjpegBufferPinnedDestroy(nvjpegBufferPinned_t buffer);

struct nvjpegBufferDevice;
typedef struct nvjpegBufferDevice* nvjpegBufferDevice_t;

nvjpegStatus_t NVJPEGAPI nvjpegBufferDeviceCreate(nvjpegHandle_t handle, 
    nvjpegDevAllocator_t* device_allocator, 
    nvjpegBufferDevice_t* buffer);

nvjpegStatus_t NVJPEGAPI nvjpegBufferDeviceDestroy(nvjpegBufferDevice_t buffer);

// retrieve buffer size and pointer - this allows reusing buffer when decode is not needed
nvjpegStatus_t NVJPEGAPI nvjpegBufferPinnedRetrieve(nvjpegBufferPinned_t buffer, size_t* size, void** ptr);

nvjpegStatus_t NVJPEGAPI nvjpegBufferDeviceRetrieve(nvjpegBufferDevice_t buffer, size_t* size, void** ptr);

// this allows attaching same memory buffers to different states, allowing to switch implementations
// without allocating extra memory
nvjpegStatus_t NVJPEGAPI nvjpegStateAttachPinnedBuffer(nvjpegJpegState_t decoder_state,
    nvjpegBufferPinned_t pinned_buffer);

nvjpegStatus_t NVJPEGAPI nvjpegStateAttachDeviceBuffer(nvjpegJpegState_t decoder_state,
    nvjpegBufferDevice_t device_buffer);

///////////////////////////////////////////////////////////////////////////////////
// JPEG stream parameters //
///////////////////////////////////////////////////////////////////////////////////

// handle that stores stream information - metadata, encoded image parameters, encoded stream parameters
// stores everything on CPU side. This allows us parse header separately from implementation
// and retrieve more information on the stream. Also can be used for transcoding and transfering 
// metadata to encoder
struct nvjpegJpegStream;
typedef struct nvjpegJpegStream* nvjpegJpegStream_t;

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamCreate(
    nvjpegHandle_t handle, 
    nvjpegJpegStream_t *jpeg_stream);

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamDestroy(nvjpegJpegStream_t jpeg_stream);

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamParse(
    nvjpegHandle_t handle,
    const unsigned char *data, 
    size_t length,
    int save_metadata,
    int save_stream,
    nvjpegJpegStream_t jpeg_stream);

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamParseHeader(
    nvjpegHandle_t handle,
    const unsigned char *data, 
    size_t length,
    nvjpegJpegStream_t jpeg_stream);

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamParseTables(
    nvjpegHandle_t handle,
    const unsigned char *data,
    size_t length,
    nvjpegJpegStream_t jpeg_stream);

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamGetJpegEncoding(
    nvjpegJpegStream_t jpeg_stream,
    nvjpegJpegEncoding_t* jpeg_encoding);

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamGetFrameDimensions(
    nvjpegJpegStream_t jpeg_stream,
    unsigned int* width,
    unsigned int* height);

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamGetComponentsNum(
    nvjpegJpegStream_t jpeg_stream,
    unsigned int* components_num);

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamGetComponentDimensions(
    nvjpegJpegStream_t jpeg_stream,
    unsigned int component,
    unsigned int* width,
    unsigned int* height);

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamGetExifOrientation(
    nvjpegJpegStream_t jpeg_stream,
    nvjpegExifOrientation_t *orientation_flag);

// if encoded is 1 color component then it assumes 4:0:0 (NVJPEG_CSS_GRAY, grayscale)
// if encoded is 3 color components it tries to assign one of the known subsamplings
//   based on the components subsampling infromation
// in case sampling factors are not stadard or number of components is different 
//   it will return NVJPEG_CSS_UNKNOWN
nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamGetChromaSubsampling(
    nvjpegJpegStream_t jpeg_stream,
    nvjpegChromaSubsampling_t* chroma_subsampling);

///////////////////////////////////////////////////////////////////////////////////
// Decode parameters //
///////////////////////////////////////////////////////////////////////////////////
// decode parameters structure. Used to set decode-related tweaks
struct nvjpegDecodeParams;
typedef struct nvjpegDecodeParams* nvjpegDecodeParams_t;

nvjpegStatus_t NVJPEGAPI nvjpegDecodeParamsCreate(
    nvjpegHandle_t handle, 
    nvjpegDecodeParams_t *decode_params);

nvjpegStatus_t NVJPEGAPI nvjpegDecodeParamsDestroy(nvjpegDecodeParams_t decode_params);

// set output pixel format - same value as in nvjpegDecode()
nvjpegStatus_t NVJPEGAPI nvjpegDecodeParamsSetOutputFormat(
    nvjpegDecodeParams_t decode_params,
    nvjpegOutputFormat_t output_format);

// set to desired ROI. set to (0, 0, -1, -1) to disable ROI decode (decode whole image)
nvjpegStatus_t NVJPEGAPI nvjpegDecodeParamsSetROI(
    nvjpegDecodeParams_t decode_params,
    int offset_x, int offset_y, int roi_width, int roi_height);

// set to true to allow conversion from CMYK to RGB or YUV that follows simple subtractive scheme
nvjpegStatus_t NVJPEGAPI nvjpegDecodeParamsSetAllowCMYK(
    nvjpegDecodeParams_t decode_params,
    int allow_cmyk);

// works only with the hardware decoder backend
nvjpegStatus_t NVJPEGAPI nvjpegDecodeParamsSetScaleFactor(
    nvjpegDecodeParams_t decode_params,
    nvjpegScaleFactor_t scale_factor);

// set the orientation flag to the decode parameters
nvjpegStatus_t NVJPEGAPI nvjpegDecodeParamsSetExifOrientation(
    nvjpegDecodeParams_t decode_params,
    nvjpegExifOrientation_t orientation);

///////////////////////////////////////////////////////////////////////////////////
// Decoder helper functions //
///////////////////////////////////////////////////////////////////////////////////

struct nvjpegJpegDecoder;
typedef struct nvjpegJpegDecoder* nvjpegJpegDecoder_t;

//creates decoder implementation
nvjpegStatus_t NVJPEGAPI nvjpegDecoderCreate(nvjpegHandle_t nvjpeg_handle, 
    nvjpegBackend_t implementation, 
    nvjpegJpegDecoder_t* decoder_handle);

nvjpegStatus_t NVJPEGAPI nvjpegDecoderDestroy(nvjpegJpegDecoder_t decoder_handle);

// on return sets is_supported value to 0 if decoder is capable to handle jpeg_stream 
// with specified decode parameters
nvjpegStatus_t NVJPEGAPI nvjpegDecoderJpegSupported(nvjpegJpegDecoder_t decoder_handle, 
    nvjpegJpegStream_t jpeg_stream,
    nvjpegDecodeParams_t decode_params,
    int* is_supported);

nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedSupported(nvjpegHandle_t handle,
    nvjpegJpegStream_t jpeg_stream,
    int* is_supported);

nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedSupportedEx(nvjpegHandle_t handle,
    nvjpegJpegStream_t jpeg_stream,
    nvjpegDecodeParams_t decode_params,
    int* is_supported);

// creates decoder state 
nvjpegStatus_t NVJPEGAPI nvjpegDecoderStateCreate(nvjpegHandle_t nvjpeg_handle,
    nvjpegJpegDecoder_t decoder_handle,
    nvjpegJpegState_t* decoder_state);

///////////////////////////////////////////////////////////////////////////////////
// Decode functions //
///////////////////////////////////////////////////////////////////////////////////
// takes parsed jpeg as input and performs decoding
nvjpegStatus_t NVJPEGAPI nvjpegDecodeJpeg(
    nvjpegHandle_t handle,
    nvjpegJpegDecoder_t decoder,
    nvjpegJpegState_t decoder_state,
    nvjpegJpegStream_t jpeg_bitstream,
    nvjpegImage_t *destination,
    nvjpegDecodeParams_t decode_params,
    cudaStream_t stream);


// starts decoding on host and save decode parameters to the state
nvjpegStatus_t NVJPEGAPI nvjpegDecodeJpegHost(
    nvjpegHandle_t handle,
    nvjpegJpegDecoder_t decoder,
    nvjpegJpegState_t decoder_state,
    nvjpegDecodeParams_t decode_params,
    nvjpegJpegStream_t jpeg_stream);

// hybrid stage of decoding image,  involves device async calls
// note that jpeg stream is a parameter here - because we still might need copy 
// parts of bytestream to device
nvjpegStatus_t NVJPEGAPI nvjpegDecodeJpegTransferToDevice(
    nvjpegHandle_t handle,
    nvjpegJpegDecoder_t decoder,
    nvjpegJpegState_t decoder_state,
    nvjpegJpegStream_t jpeg_stream,
    cudaStream_t stream);

// finishing async operations on the device
nvjpegStatus_t NVJPEGAPI nvjpegDecodeJpegDevice(
    nvjpegHandle_t handle,
    nvjpegJpegDecoder_t decoder,
    nvjpegJpegState_t decoder_state,
    nvjpegImage_t *destination,
    cudaStream_t stream);


nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedEx(
          nvjpegHandle_t handle,
          nvjpegJpegState_t jpeg_handle,
          const unsigned char *const *data,
          const size_t *lengths,
          nvjpegImage_t *destinations,
          nvjpegDecodeParams_t *decode_params,
          cudaStream_t stream);

///////////////////////////////////////////////////////////////////////////////////
// JPEG Transcoding Functions //
///////////////////////////////////////////////////////////////////////////////////

// copies metadata (JFIF, APP, EXT, COM markers) from parsed stream
nvjpegStatus_t nvjpegEncoderParamsCopyMetadata(
	nvjpegEncoderState_t encoder_state,
    nvjpegEncoderParams_t encode_params,
    nvjpegJpegStream_t jpeg_stream,
    cudaStream_t stream);

// copies quantization tables from parsed stream
nvjpegStatus_t nvjpegEncoderParamsCopyQuantizationTables(
    nvjpegEncoderParams_t encode_params,
    nvjpegJpegStream_t jpeg_stream,
    cudaStream_t stream);

// copies huffman tables from parsed stream. should require same scans structure
nvjpegStatus_t nvjpegEncoderParamsCopyHuffmanTables(
    nvjpegEncoderState_t encoder_state,
    nvjpegEncoderParams_t encode_params,
    nvjpegJpegStream_t jpeg_stream,
    cudaStream_t stream);

#if defined(__cplusplus)
  }
#endif
 
#endif /* NV_JPEG_HEADER */
