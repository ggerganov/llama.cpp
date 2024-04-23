/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "dsc.h"
#include <dstorage.h>
#include <dxgi1_4.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <winrt/base.h>
#include <windows.h>
#include <sstream>
#include <cuda.h>
#include <cuda_d3d11_interop.h>
#include <Aclapi.h> // WindowsSecurityAttributes
#include <queue>
#include <condition_variable>
#include <mutex>

#include <nvtx3/nvToolsExt.h>


class WindowsSecurityAttributes
{
protected:
  SECURITY_ATTRIBUTES m_winSecurityAttributes = {};
  SECURITY_DESCRIPTOR m_securityDescriptor = {};
  PSID pSID = 0;
  PACL pACL = 0;

public:
  WindowsSecurityAttributes()
  {
    InitializeSecurityDescriptor(&m_securityDescriptor, SECURITY_DESCRIPTOR_REVISION);

    SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority = SECURITY_WORLD_SID_AUTHORITY;
    AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0, 0, 0, 0, 0, 0, &pSID);

    EXPLICIT_ACCESS explicitAccess = {};
    explicitAccess.grfAccessPermissions = STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
    explicitAccess.grfAccessMode = SET_ACCESS;
    explicitAccess.grfInheritance = INHERIT_ONLY;
    explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
    explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
    explicitAccess.Trustee.ptstrName = reinterpret_cast<LPTSTR>(pSID);

    SetEntriesInAcl(1, &explicitAccess, NULL, &pACL);
    SetSecurityDescriptorDacl(&m_securityDescriptor, TRUE, pACL, FALSE);

    m_winSecurityAttributes.nLength = sizeof(m_winSecurityAttributes);
    m_winSecurityAttributes.lpSecurityDescriptor = &m_securityDescriptor;
    m_winSecurityAttributes.bInheritHandle = TRUE;
  }

  WindowsSecurityAttributes(WindowsSecurityAttributes const& rhs) = delete;
  WindowsSecurityAttributes(WindowsSecurityAttributes const&& rhs) = delete;

  ~WindowsSecurityAttributes() {
    if (pSID)
    {
      FreeSid(pSID);
    }
    if (pACL)
    {
      LocalFree(pACL);
    }
  }

  operator SECURITY_ATTRIBUTES const* () const {
    return &m_winSecurityAttributes;
  }
};

DirectStorageCUDA::~DirectStorageCUDA()
{
}

struct DirectStorageCUDAFileHandleImpl : DirectStorageCUDAFileHandle
{
    ~DirectStorageCUDAFileHandleImpl() {};

    using File = winrt::com_ptr<IDStorageFile>;
    File file;

    IDStorageFile* get() { return file.get(); }
    IDStorageFile** put() { return file.put(); }
};

InteropBuffer::~InteropBuffer()
{
}

class InteropBufferImpl : public InteropBuffer
{
public:
  InteropBufferImpl(winrt::com_ptr<ID3D12Device> const& d3d_device, size_t size)
  {

    // Create the ID3D12Resource buffer which will be used as temporary scratch space for d3d
    // since it's not possible to import CUDA memory into DX.
    D3D12_HEAP_PROPERTIES bufferHeapProps = {};
    bufferHeapProps.Type = D3D12_HEAP_TYPE_DEFAULT;


    D3D12_HEAP_DESC hd = {};
    hd.SizeInBytes = size;
    hd.Properties = bufferHeapProps;
    hd.Flags = D3D12_HEAP_FLAG_SHARED;
    hd.Alignment = 0;
    d3d_device->CreateHeap(&hd, IID_PPV_ARGS(&m_d3d_heap));


    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = size;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bufferDesc.SampleDesc.Count = 1;

//#define USE_BUFFER
#if defined(USE_BUFFER) // 
    winrt::check_hresult(d3d_device->CreateCommittedResource(
      &bufferHeapProps,
      D3D12_HEAP_FLAG_NONE | D3D12_HEAP_FLAG_SHARED,
      &bufferDesc,
      D3D12_RESOURCE_STATE_COMMON,
      nullptr,
      IID_PPV_ARGS(m_d3d_buffer.put())));
#else
    winrt::check_hresult(d3d_device->CreatePlacedResource(
      m_d3d_heap.get(),
      0,
      &bufferDesc,
      D3D12_RESOURCE_STATE_COMMON,
      nullptr,
      IID_PPV_ARGS(m_d3d_buffer.put())));
#endif

#if 0
    // debug begin
    bufferHeapProps.Type = D3D12_HEAP_TYPE_READBACK;
    winrt::check_hresult(d3d_device->CreateCommittedResource(
      &bufferHeapProps,
      D3D12_HEAP_FLAG_NONE,
      &bufferDesc,
      D3D12_RESOURCE_STATE_COMMON,
      nullptr,
      IID_PPV_ARGS(m_host_buffer.put())));

    m_host_buffer->Map(0, nullptr, &m_host_ptr);
    d3d_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_cmdallocator));
    d3d_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_cmdallocator.get(), nullptr, IID_PPV_ARGS(&m_cmdlist));
#endif

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    d3d_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&m_cmd_queue));
    // debug end

    // create a shared handle to require to import the d3d buffer into CUDA
    HANDLE sharedHandle;
    WindowsSecurityAttributes windowsSecurityAttributes;
    LPCWSTR name = NULL;
#if USE_BUFFER
    d3d_device->CreateSharedHandle(m_d3d_buffer.get(), windowsSecurityAttributes, GENERIC_ALL, name, &sharedHandle);

    cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
#else
    d3d_device->CreateSharedHandle(m_d3d_heap.get(), windowsSecurityAttributes, GENERIC_ALL, name, &sharedHandle);

    cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Heap;
#endif
    externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
    externalMemoryHandleDesc.size = bufferDesc.Width;
    externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;
    auto result = cudaImportExternalMemory(&m_externalMemory, &externalMemoryHandleDesc);

    CloseHandle(sharedHandle);

    // get pointer to external memory imported form d3d
    cudaExternalMemoryBufferDesc externalMemoryBufferDesc = {};
    externalMemoryBufferDesc.offset = 0;
    externalMemoryBufferDesc.size = externalMemoryHandleDesc.size;
    externalMemoryBufferDesc.flags = 0;

    result = cudaExternalMemoryGetMappedBuffer(&m_cuda_dev_ptr, m_externalMemory, &externalMemoryBufferDesc);
    result = cudaDeviceSynchronize();

    auto err = cudaMemset(m_cuda_dev_ptr, 255, 512*1024*1024);
    result = cudaDeviceSynchronize();
    std::cout << "err: " << err << std::endl;
  }

  ~InteropBufferImpl() {
    auto result = cudaDestroyExternalMemory(m_externalMemory);
    cudaFree(m_cuda_dev_ptr);
    if (result != cudaSuccess) {
      std::cout << "cudaDestroyExternalMemory interop buffer: " << result << std::endl;
    }
  }

  void* get_device_ptr() const {
    return m_cuda_dev_ptr;
  }

  ID3D12Resource* get_d3d_buffer() const {
    return m_d3d_buffer.get();
  }

  void* get_host_ptr() const {
#if 0
    m_cmdlist->Reset(m_cmdallocator.get(), nullptr);
    m_cmdlist->CopyResource(m_host_buffer.get(), m_d3d_buffer.get());
    m_cmdlist->Close();

    ID3D12CommandList *ptr = m_cmdlist.get();
    m_cmd_queue->ExecuteCommandLists(1, &ptr);
    Sleep(2);
#endif

    return m_host_ptr;
  }

private:
  winrt::com_ptr<ID3D12CommandQueue> m_cmd_queue = {};
  winrt::com_ptr<ID3D12Resource> m_d3d_buffer = {};
  winrt::com_ptr<ID3D12Heap> m_d3d_heap = {};

  cudaExternalMemory_t m_externalMemory;
  void* m_cuda_dev_ptr;

  // debug
  winrt::com_ptr<ID3D12Resource> m_host_buffer = {};
  winrt::com_ptr<ID3D12GraphicsCommandList> m_cmdlist = {};
  winrt::com_ptr<ID3D12CommandAllocator> m_cmdallocator = {};
  void* m_host_ptr;
};

class DirectStorageCUDAImpl : public DirectStorageCUDA
{
public:
  DirectStorageCUDAImpl(int scratch_size, int number_of_scratch_spaces);

  virtual ~DirectStorageCUDAImpl() {
    flush(true);
    std::cout << "~DirectStorageCudaImpl" << std::endl;
  }

    struct FileInfo {
        const std::string& filename;
        void* cuda_device_ptr;
        size_t offset;
        size_t size;
    };

    virtual std::unique_ptr<InteropBuffer> create_interop_buffer(size_t size);
    virtual DirectStorageCUDA::File openFile(std::string const& filename);
    virtual void loadFile(DirectStorageCUDA::File const& file, size_t read_start, size_t read_len, void* cuda_dst_ptr);
    virtual void loadFile(File const& file, size_t read_start, size_t read_len, InteropBuffer* interop_buffer, size_t interop_buffer_offset);
    virtual void flush(bool last);
private:
    class StagingArea
    {
    public:
      StagingArea(winrt::com_ptr<ID3D12Device> d3d_device, winrt::com_ptr<IDStorageFactory> d3d_factory, size_t chunk_size, size_t number_of_chunks)
        : m_d3d_device(d3d_device)
        , m_d3d_factory(d3d_factory)
        , m_chunk_size(chunk_size)
        , m_number_of_chunks(number_of_chunks)
        , m_total_staging_space(chunk_size * number_of_chunks)
      {
        // Create a DirectStorage queue which will be used to load data into a
        // buffer on the GPU.
        DSTORAGE_QUEUE_DESC queueDesc{};
        queueDesc.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
        queueDesc.Priority = DSTORAGE_PRIORITY_NORMAL;
        queueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        queueDesc.Device = m_d3d_device.get();

        winrt::check_hresult(m_d3d_factory->CreateQueue(&queueDesc, IID_PPV_ARGS(m_d3d_storage_queue.put())));

        // Create the ID3D12Resource buffer which will be used as temporary scratch space for d3d
        // since it's not possible to import CUDA memory into DX.
        D3D12_HEAP_PROPERTIES bufferHeapProps = {};
        bufferHeapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

        D3D12_RESOURCE_DESC bufferDesc = {};
        bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufferDesc.Width = m_chunk_size * m_number_of_chunks;
        bufferDesc.Height = 1;
        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.MipLevels = 1;
        bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bufferDesc.SampleDesc.Count = 1;

        winrt::check_hresult(m_d3d_device->CreateCommittedResource(
          &bufferHeapProps,
          D3D12_HEAP_FLAG_NONE | D3D12_HEAP_FLAG_SHARED,
          &bufferDesc,
          D3D12_RESOURCE_STATE_COMMON,
          nullptr,
          IID_PPV_ARGS(m_d3d_scratch_space.put())));


        // create a shared handle to require to import the d3d buffer into CUDA
        HANDLE sharedHandle;
        WindowsSecurityAttributes windowsSecurityAttributes;
        LPCWSTR name = NULL;
        m_d3d_device->CreateSharedHandle(m_d3d_scratch_space.get(), windowsSecurityAttributes, GENERIC_ALL, name, &sharedHandle);

        cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
        externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
        externalMemoryHandleDesc.size = bufferDesc.Width;
        externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;
        auto result = cudaImportExternalMemory(&m_externalMemory, &externalMemoryHandleDesc);

        CloseHandle(sharedHandle);

        // get pointer to external memory imported form d3d
        cudaExternalMemoryBufferDesc externalMemoryBufferDesc = {};
        externalMemoryBufferDesc.offset = 0;
        externalMemoryBufferDesc.size = externalMemoryHandleDesc.size;
        externalMemoryBufferDesc.flags = 0;

        result = cudaExternalMemoryGetMappedBuffer(&m_cuda_scratch_space, m_externalMemory, &externalMemoryBufferDesc);

        // create d3d fence for synchronization
        auto resultDx = m_d3d_device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_d3d_fence));

        // import d3d fence as semaphore into CUDA.
        cudaExternalSemaphoreHandleDesc extSemHandleDesc = {};
        extSemHandleDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
        m_d3d_device->CreateSharedHandle(m_d3d_fence.get(), nullptr, GENERIC_ALL, nullptr, &extSemHandleDesc.handle.win32.handle);
        result = cudaImportExternalSemaphore(&m_externalSemaphore, &extSemHandleDesc);

        cudaStreamCreate(&m_cudaStream);

        // intialize fence to wait for in flush
        waitParams.params.fence.value = 1;
      }

      ~StagingArea()
      {
        std::cout << "~StagingArea" << std::endl;
        auto result = cudaDestroyExternalMemory(m_externalMemory);
        cudaFree(m_cuda_scratch_space);
        if (result != cudaSuccess) {
          std::cout << "cudaDestroyExternalMemory interop buffer: " << result << std::endl;
        }
        // TODO ensure that no resources are being leaked
      }

      // enqueue as much data as possible into the current staging area.
      // enqueue will return true if all data has been enqueued, false otherwise.
      // start, len and cuda_dev_ptr will be updated.
      bool enqueue(DirectStorageCUDA::File const& file, size_t& read_start, size_t& len, void*& cuda_dst_ptr)
      {
        if (len == 0)
          return false;

        m_enqueued = true;

        size_t memcpy_src_start = m_current_staging_offset;

        static size_t load_cnt = 0;
        size_t read_end = read_start + len;
        for (size_t src_start = read_start; src_start < read_end; src_start += m_chunk_size)
        {
          ++load_cnt;
          size_t src_end = min(read_end, src_start + m_chunk_size);
          size_t src_size = src_end - src_start;

          if (m_current_staging_offset + src_size >= m_total_staging_space)
          {
            //std::cout << load_cnt << std::endl;
            load_cnt = 0;

            size_t processed_len = m_current_staging_offset - memcpy_src_start;
            m_staging_memcpies.push_back(MemcpyOp(cuda_dst_ptr, (void*)((char*)m_cuda_scratch_space + memcpy_src_start), processed_len));

            cuda_dst_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(cuda_dst_ptr) + processed_len);
            read_start += processed_len;
            len -= processed_len;

            flush(false);

            memcpy_src_start = m_current_staging_offset;

            return true;
          }

          DSTORAGE_REQUEST request = {};
          request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
          request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
          request.Source.File.Source = static_cast<DirectStorageCUDAFileHandleImpl*>(file.get())->get();
          request.Source.File.Offset = src_start;
          request.Source.File.Size = src_size; // filesize
          request.UncompressedSize = src_size; // filesize 
          request.Destination.Buffer.Resource = m_d3d_scratch_space.get();
          request.Destination.Buffer.Offset = m_current_staging_offset;
          request.Destination.Buffer.Size = src_size;

          m_d3d_storage_queue->EnqueueRequest(&request);

          m_current_staging_offset += request.Destination.Buffer.Size;
        }

        m_staging_memcpies.push_back(MemcpyOp((void*)((char*)cuda_dst_ptr), (void*)((char*)m_cuda_scratch_space + memcpy_src_start), m_current_staging_offset - memcpy_src_start));

        size_t processed_len = m_current_staging_offset - memcpy_src_start;
        cuda_dst_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(cuda_dst_ptr) + m_current_staging_offset - memcpy_src_start);
        read_start += processed_len;
        len -= processed_len;

        return false;
      }

      void enqueue(DirectStorageCUDA::File const& file, size_t& read_start, size_t& read_len, InteropBuffer* interop_buffer, size_t interop_buffer_offset)
      {
        InteropBufferImpl* ibi = static_cast<InteropBufferImpl*>(interop_buffer);
        bool flushed;
        while (read_len) {
          size_t request_size = min(m_chunk_size, read_len);

          DSTORAGE_REQUEST request = {};
          request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
          request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
          request.Source.File.Source = static_cast<DirectStorageCUDAFileHandleImpl*>(file.get())->get();
          request.Source.File.Offset = read_start;
          request.Source.File.Size = request_size; // filesize
          request.UncompressedSize = request_size; // filesize 
          request.Destination.Buffer.Resource = ibi->get_d3d_buffer();
          request.Destination.Buffer.Offset = interop_buffer_offset;
          request.Destination.Buffer.Size = request_size;
          //std::cout << read_start / (1024*1024) << " / " << interop_buffer_offset / (1024 * 1024) << "/" << request_size / (1024 * 1024) << std::endl;

          m_d3d_storage_queue->EnqueueRequest(&request);

          read_len -= request_size;
          interop_buffer_offset += request_size;
          read_start += request_size;

          m_enqueued = true;
          //flush(true);
        };

      }


      void wait()
      {
        if (m_enqueued) {
          cudaStreamSynchronize(m_cudaStream);
          m_enqueued = false;
        }
      }

      void flush(bool last)
      {
        m_d3d_storage_queue->EnqueueSignal(m_d3d_fence.get(), waitParams.params.fence.value);
        m_d3d_storage_queue->Submit();

        nvtxRangePop();
        nvtxRangePush("wait");
        cudaWaitExternalSemaphoresAsync(&m_externalSemaphore, &waitParams, 1, m_cudaStream);
        nvtxRangePop();
        nvtxRangePush("memcpy");
#if 1
        for (auto const& op : m_staging_memcpies) {
          auto result = cudaMemcpyAsync(op.m_dst, op.m_src, op.m_size, cudaMemcpyDeviceToDevice, m_cudaStream);
        }
#endif
        nvtxRangePop();
        nvtxRangePush("sync");
        //cudaStreamSynchronize(m_cudaStream);
        nvtxRangePop();

        // increase fence value by 1 for next flush call
        waitParams.params.fence.value += 1;

        // reset staging area
        m_staging_memcpies.clear();
        m_current_staging_offset = 0;

#if 1
        if (last) {
          DSTORAGE_ERROR_RECORD errorRecord{};
          m_d3d_storage_queue->RetrieveErrorRecord(&errorRecord);
          if (FAILED(errorRecord.FirstFailure.HResult))
          {
            //
            // errorRecord.FailureCount - The number of failed requests in the queue since the last
            //                            RetrieveErrorRecord call.
            // errorRecord.FirstFailure - Detailed record about the first failed command in the enqueue order.
            //
            std::cout << "The DirectStorage request failed! HRESULT=0x" << std::hex << errorRecord.FirstFailure.HResult << std::endl;
          }
        }
#endif
      }

      winrt::com_ptr<ID3D12Device> m_d3d_device = {};
      winrt::com_ptr<IDStorageFactory> m_d3d_factory = {};
      winrt::com_ptr<IDStorageQueue> m_d3d_storage_queue = {};
      winrt::com_ptr<ID3D12Resource> m_d3d_scratch_space = {};
      winrt::com_ptr<ID3D12Fence> m_d3d_fence = {};

      // cuda external memory resources
      cudaExternalMemoryHandleType m_externalMemoryHandleType;
      cudaExternalMemory_t m_externalMemory;
      cudaExternalSemaphore_t m_externalSemaphore;
      cudaExternalSemaphoreWaitParams waitParams = {};

      size_t m_chunk_size;
      size_t m_number_of_chunks;
      size_t m_total_staging_space;

      cudaStream_t m_cudaStream;
      void* m_cuda_scratch_space;
      bool m_enqueued = false; // is any data enqueued

      // memcpy 
      size_t m_current_staging_offset = 0; // current offset in the staging buffer

      // memcpies from the staging buffer to the actual CUDA memory
      struct MemcpyOp {
        MemcpyOp(void* dst, void* src, size_t size)
          : m_dst(dst), m_src(src), m_size(size) {}
        void* m_dst;
        void* m_src;
        size_t m_size;
      };
      std::vector<MemcpyOp> m_staging_memcpies;

    };

    winrt::com_ptr<ID3D12Device> m_d3d_device = {};
    winrt::com_ptr<IDStorageFactory> m_d3d_factory = {};

    size_t m_chunk_size;
    size_t m_number_of_chunks;

    std::vector<std::unique_ptr<StagingArea>> m_staging_areas;
    size_t m_staging_index = 0;
};

std::unique_ptr<DirectStorageCUDA> DirectStorageCUDA::create(int scratch_size, int number_of_scratch_spaces)
{
    return std::make_unique<DirectStorageCUDAImpl>(scratch_size, number_of_scratch_spaces);
}

  // copy read_len bytes starting at read_start from the given file to the given cuda ptr
void DirectStorageCUDAImpl::loadFile(DirectStorageCUDA::File const& file, size_t read_start, size_t read_len, void* cuda_dst_ptr)
{
  bool flushed;
  while (read_len) {
    flushed = m_staging_areas[m_staging_index]->enqueue(file, read_start, read_len, cuda_dst_ptr);
    if (flushed) {
      m_staging_index = (m_staging_index + 1) % m_staging_areas.size();
      m_staging_areas[m_staging_index]->wait();
    }
  };
}

void DirectStorageCUDAImpl::loadFile(DirectStorageCUDA::File const& file, size_t read_start, size_t read_len, InteropBuffer *interop_buffer, size_t interop_buffer_offset)
{
  if (!interop_buffer)
    return;

  m_staging_areas[m_staging_index]->enqueue(file, read_start, read_len, interop_buffer, interop_buffer_offset);
}


void DirectStorageCUDAImpl::flush(bool last)
{
  for (auto& sa : m_staging_areas) {
    sa->flush(last);
  }
  if (last) {
    for (auto& sa : m_staging_areas) {
      sa->wait();
    }
  }
}

DirectStorageCUDAImpl::DirectStorageCUDAImpl(int scratch_size, int number_of_scratch_spaces)
    : m_chunk_size(scratch_size)
    , m_number_of_chunks(number_of_scratch_spaces)
{
  DSTORAGE_CONFIGURATION direct_storage_config = {};
  direct_storage_config.NumSubmitThreads = 1;
  DStorageSetConfiguration(&direct_storage_config);

    winrt::check_hresult(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&m_d3d_device)));
    winrt::check_hresult(DStorageGetFactory(IID_PPV_ARGS(m_d3d_factory.put())));

    size_t num_staging_areas = 2;
    for (size_t idx = 0; idx < num_staging_areas; ++idx) {
      m_staging_areas.emplace_back(std::make_unique<StagingArea>(m_d3d_device, m_d3d_factory, m_chunk_size, m_number_of_chunks));

    }
}

DirectStorageCUDAImpl::File DirectStorageCUDAImpl::openFile(std::string const& filename)
{
    File file = std::make_unique<DirectStorageCUDAFileHandleImpl>();
    std::wstring wfilename(filename.begin(), filename.end());
    nvtxRangePush("factory open file");
    HRESULT hr = m_d3d_factory->OpenFile(wfilename.c_str(), IID_PPV_ARGS(static_cast<DirectStorageCUDAFileHandleImpl*>(file.get())->put()));
    if (FAILED(hr))
    {
        std::wcout << L"The file '" << wfilename << L"' could not be opened. HRESULT=0x" << std::hex << hr << std::endl;
        return {};
    }
    nvtxRangePop();
    return file;
}

std::unique_ptr<InteropBuffer> DirectStorageCUDAImpl::create_interop_buffer(size_t size)
{
  return std::make_unique<InteropBufferImpl>(m_d3d_device, size);
}
