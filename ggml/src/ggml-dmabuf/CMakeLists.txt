include(CheckIncludeFile)

# shared buffers are only built when the kernel dma-buf uapi is available
check_include_file("linux/dma-buf.h"  GGML_DMABUF_HAVE_DMA_BUF)
check_include_file("linux/dma-heap.h" GGML_DMABUF_HAVE_DMA_HEAP)
if (NOT GGML_DMABUF_HAVE_DMA_BUF OR NOT GGML_DMABUF_HAVE_DMA_HEAP)
    message(WARNING "dma-buf buffers require <linux/dma-buf.h> and <linux/dma-heap.h>, disabling")
    return()
endif()

target_sources(ggml-base PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml-dmabuf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml-dmabuf-allocators.cpp)
target_compile_definitions(ggml-base PUBLIC GGML_USE_DMABUF)
