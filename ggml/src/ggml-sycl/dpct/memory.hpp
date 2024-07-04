//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Part of the LLVM Project, under the Apache License v2.0 with LLVM
 *  Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 **************************************************************************/

#pragma once

#include <cassert>
#include <cstring>
#include <map>
#include <type_traits>

#if defined(__linux__)
#include <sys/mman.h>
#elif defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#error "Only support Windows and Linux."
#endif

#include <sycl/ext/intel/experimental/usm_properties.hpp>
#include <sycl/ext/oneapi/group_local_memory.hpp>
#include <sycl/sycl.hpp>

#include "device.hpp"
#include "ggml.h"

namespace dpct {

typedef uint8_t byte_t;
typedef sycl::buffer<byte_t> buffer_t;

enum memcpy_direction {
    host_to_host,
    host_to_device,
    device_to_host,
    device_to_device,
    automatic
};

enum memory_region {
    global = 0, // device global memory
    constant,   // device constant memory
    local,      // device local memory
    shared,     // memory which can be accessed by host and device
};

/// Pitched 2D/3D memory data.
class pitched_data {
  public:
    pitched_data() : pitched_data(nullptr, 0, 0, 0) {}
    pitched_data(void *data, size_t pitch, size_t x, size_t y)
        : _data(data), _pitch(pitch), _x(x), _y(y) {}

    void *get_data_ptr() { return _data; }
    void set_data_ptr(void *data) { _data = data; }

    size_t get_pitch() { return _pitch; }
    void set_pitch(size_t pitch) { _pitch = pitch; }

    size_t get_x() { return _x; }
    void set_x(size_t x) { _x = x; };

    size_t get_y() { return _y; }
    void set_y(size_t y) { _y = y; }

  private:
    void *_data;
    size_t _pitch, _x, _y;
};

namespace detail {

enum class pointer_access_attribute {
    host_only = 0,
    device_only,
    host_device,
    end
};

static pointer_access_attribute get_pointer_attribute(sycl::queue &q,
                                                      const void *ptr) {
    switch (sycl::get_pointer_type(ptr, q.get_context())) {
    case sycl::usm::alloc::unknown:
        return pointer_access_attribute::host_only;
    case sycl::usm::alloc::device:
        return pointer_access_attribute::device_only;
    case sycl::usm::alloc::shared:
    case sycl::usm::alloc::host:
        return pointer_access_attribute::host_device;
    }
}

class mem_mgr {
    mem_mgr() {
        // Reserved address space, no real memory allocation happens here.
#if defined(__linux__)
        mapped_address_space =
            (byte_t *)mmap(nullptr, mapped_region_size, PROT_NONE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#elif defined(_WIN64)
        mapped_address_space = (byte_t *)VirtualAlloc(
            NULL,               // NULL specified as the base address parameter
            mapped_region_size, // Size of allocation
            MEM_RESERVE,        // Allocate reserved pages
            PAGE_NOACCESS);     // Protection = no access
#else
#error "Only support Windows and Linux."
#endif
        next_free = mapped_address_space;
    };

  public:
    using buffer_id_t = int;

    struct allocation {
        buffer_t buffer;
        byte_t *alloc_ptr;
        size_t size;
    };

    ~mem_mgr() {
#if defined(__linux__)
        munmap(mapped_address_space, mapped_region_size);
#elif defined(_WIN64)
        VirtualFree(mapped_address_space, 0, MEM_RELEASE);
#else
#error "Only support Windows and Linux."
#endif
    };

    mem_mgr(const mem_mgr &) = delete;
    mem_mgr &operator=(const mem_mgr &) = delete;
    mem_mgr(mem_mgr &&) = delete;
    mem_mgr &operator=(mem_mgr &&) = delete;

    /// Allocate
    void *mem_alloc(size_t size) {
        if (!size)
            return nullptr;
        std::lock_guard<std::mutex> lock(m_mutex);
        if (next_free + size > mapped_address_space + mapped_region_size) {
            throw std::runtime_error(
                "dpct_malloc: out of memory for virtual memory pool");
        }
        // Allocation
        sycl::range<1> r(size);
        buffer_t buf(r);
        allocation A{buf, next_free, size};
        // Map allocation to device pointer
        void *result = next_free;
        m_map.emplace(next_free + size, A);
        // Update pointer to the next free space.
        next_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

        return result;
    }

    /// Deallocate
    void mem_free(const void *ptr) {
        if (!ptr)
            return;
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = get_map_iterator(ptr);
        m_map.erase(it);
    }

    /// map: device pointer -> allocation(buffer, alloc_ptr, size)
    allocation translate_ptr(const void *ptr) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = get_map_iterator(ptr);
        return it->second;
    }

    /// Check if the pointer represents device pointer or not.
    bool is_device_ptr(const void *ptr) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return (mapped_address_space <= ptr) &&
               (ptr < mapped_address_space + mapped_region_size);
    }

    /// Returns the instance of memory manager singleton.
    static mem_mgr &instance() {
        static mem_mgr m;
        return m;
    }

  private:
    std::map<byte_t *, allocation> m_map;
    mutable std::mutex m_mutex;
    byte_t *mapped_address_space;
    byte_t *next_free;
    const size_t mapped_region_size = 128ull * 1024 * 1024 * 1024;
    const size_t alignment = 256;
    /// This padding may be defined to some positive value to debug
    /// out of bound accesses.
    const size_t extra_padding = 0;

    std::map<byte_t *, allocation>::iterator get_map_iterator(const void *ptr) {
        auto it = m_map.upper_bound((byte_t *)ptr);
        if (it == m_map.end()) {
            // Not a virtual pointer.
            throw std::runtime_error(
                "can not get buffer from non-virtual pointer");
        }
        const allocation &alloc = it->second;
        if (ptr < alloc.alloc_ptr) {
            // Out of bound.
            // This may happen if there's a gap between allocations due to
            // alignment or extra padding and pointer points to this gap.
            throw std::runtime_error("invalid virtual pointer");
        }
        return it;
    }
};

template <class T, memory_region Memory, size_t Dimension> class accessor;
template <memory_region Memory, class T = byte_t> class memory_traits {
  public:
    static constexpr sycl::access::target target = sycl::access::target::device;
    static constexpr sycl::access_mode mode =
        (Memory == constant) ? sycl::access_mode::read
                             : sycl::access_mode::read_write;
    static constexpr size_t type_size = sizeof(T);
    using element_t =
        typename std::conditional<Memory == constant, const T, T>::type;
    using value_t = typename std::remove_cv<T>::type;
    template <size_t Dimension = 1>
    using accessor_t = typename std::conditional<
        Memory == local, sycl::local_accessor<value_t, Dimension>,
        sycl::accessor<T, Dimension, mode, target>>::type;
    using pointer_t = T *;
};

static inline void *dpct_malloc(size_t size, sycl::queue &q) {
    return sycl::malloc_device(size, q.get_device(), q.get_context());
}

#define PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
static inline void *dpct_malloc(size_t &pitch, size_t x, size_t y, size_t z,
                                sycl::queue &q) {
    pitch = PITCH_DEFAULT_ALIGN(x);
    return dpct_malloc(pitch * y * z, q);
}

/**
 * @brief Sets \p value to the first \p size elements starting from \p dev_ptr
 * in \p q.
 * @tparam valueT The type of the element to be set.
 * @param [in] q The queue in which the operation is done.
 * @param [in] dev_ptr Pointer to the virtual device memory address.
 * @param [in] value The value to be set.
 * @param [in] size Number of elements to be set to the value.
 * @return An event representing the memset operation.
 */
template <typename valueT>
static inline sycl::event dpct_memset(sycl::queue &q, void *dev_ptr,
                                      valueT value, size_t size) {
    return q.fill(dev_ptr, value, size);
}

/**
 * @brief Sets \p value to the 3D memory region pointed by \p data in \p q.
 * @tparam valueT The type of the element to be set.
 * @param [in] q The queue in which the operation is done.
 * @param [in] data Pointer to the pitched device memory region.
 * @param [in] value The value to be set.
 * @param [in] size 3D memory region by number of elements.
 * @return An event list representing the memset operations.
 */
template <typename valueT>
static inline std::vector<sycl::event>
dpct_memset(sycl::queue &q, pitched_data data, valueT value,
            sycl::range<3> size) {
    std::vector<sycl::event> event_list;
    size_t slice = data.get_pitch() * data.get_y();
    unsigned char *data_surface = (unsigned char *)data.get_data_ptr();
    for (size_t z = 0; z < size.get(2); ++z) {
        unsigned char *data_ptr = data_surface;
        for (size_t y = 0; y < size.get(1); ++y) {
            event_list.push_back(dpct_memset(q, data_ptr, value, size.get(0)));
            data_ptr += data.get_pitch();
        }
        data_surface += slice;
    }
    return event_list;
}

/**
 * @brief Sets \p val to the pitched 2D memory region pointed by \p ptr in \p q.
 * @tparam valueT The type of the element to be set.
 * @param [in] q The queue in which the operation is done.
 * @param [in] ptr Pointer to the virtual device memory.
 * @param [in] pitch The pitch size by number of elements, including padding.
 * @param [in] val The value to be set.
 * @param [in] x The width of memory region by number of elements.
 * @param [in] y The height of memory region by number of elements.
 * @return An event list representing the memset operations.
 */
template <typename valueT>
static inline std::vector<sycl::event> dpct_memset(sycl::queue &q, void *ptr,
                                                   size_t pitch, valueT val,
                                                   size_t x, size_t y) {
    return dpct_memset(q, pitched_data(ptr, pitch, x, 1), val,
                       sycl::range<3>(x, y, 1));
}

static memcpy_direction deduce_memcpy_direction(sycl::queue &q, void *to_ptr,
                                                const void *from_ptr,
                                                memcpy_direction dir) {
    switch (dir) {
    case memcpy_direction::host_to_host:
    case memcpy_direction::host_to_device:
    case memcpy_direction::device_to_host:
    case memcpy_direction::device_to_device:
        return dir;
    case memcpy_direction::automatic: {
        // table[to_attribute][from_attribute]
        static const memcpy_direction direction_table
            [static_cast<unsigned>(pointer_access_attribute::end)]
            [static_cast<unsigned>(pointer_access_attribute::end)] = {
                {memcpy_direction::host_to_host,
                 memcpy_direction::device_to_host,
                 memcpy_direction::host_to_host},
                {memcpy_direction::host_to_device,
                 memcpy_direction::device_to_device,
                 memcpy_direction::device_to_device},
                {memcpy_direction::host_to_host,
                 memcpy_direction::device_to_device,
                 memcpy_direction::device_to_device}};
        return direction_table
            [static_cast<unsigned>(get_pointer_attribute(q, to_ptr))]
            [static_cast<unsigned>(get_pointer_attribute(q, from_ptr))];
    }
    default:
        throw std::runtime_error("dpct_memcpy: invalid direction value");
    }
}

static sycl::event
dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr, size_t size,
            memcpy_direction direction,
            const std::vector<sycl::event> &dep_events = {}) {
    if (!size)
        return sycl::event{};
    return q.memcpy(to_ptr, from_ptr, size, dep_events);
    GGML_UNUSED(direction);
}

// Get actual copy range and make sure it will not exceed range.
static inline size_t get_copy_range(sycl::range<3> size, size_t slice,
                                    size_t pitch) {
    return slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
}

static inline size_t get_offset(sycl::id<3> id, size_t slice, size_t pitch) {
    return slice * id.get(2) + pitch * id.get(1) + id.get(0);
}

/// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
/// and \p from_range to another specified by \p to_ptr and \p to_range.
static inline std::vector<sycl::event>
dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr,
            sycl::range<3> to_range, sycl::range<3> from_range,
            sycl::id<3> to_id, sycl::id<3> from_id, sycl::range<3> size,
            memcpy_direction direction,
            const std::vector<sycl::event> &dep_events = {}) {
    // RAII for host pointer
    class host_buffer {
        void *_buf;
        size_t _size;
        sycl::queue &_q;
        const std::vector<sycl::event> &_deps; // free operation depends

      public:
        host_buffer(size_t size, sycl::queue &q,
                    const std::vector<sycl::event> &deps)
            : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
        void *get_ptr() const { return _buf; }
        size_t get_size() const { return _size; }
        ~host_buffer() {
            if (_buf) {
                _q.submit([&](sycl::handler &cgh) {
                    cgh.depends_on(_deps);
                    cgh.host_task([buf = _buf] { std::free(buf); });
                });
            }
        }
    };
    std::vector<sycl::event> event_list;

    size_t to_slice = to_range.get(1) * to_range.get(0),
           from_slice = from_range.get(1) * from_range.get(0);
    unsigned char *to_surface =
        (unsigned char *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
    const unsigned char *from_surface =
        (const unsigned char *)from_ptr +
        get_offset(from_id, from_slice, from_range.get(0));

    if (to_slice == from_slice && to_slice == size.get(1) * size.get(0)) {
        return {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                            direction, dep_events)};
    }
    direction = deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
    size_t size_slice = size.get(1) * size.get(0);
    switch (direction) {
    case host_to_host:
        for (size_t z = 0; z < size.get(2); ++z) {
            unsigned char *to_ptr = to_surface;
            const unsigned char *from_ptr = from_surface;
            if (to_range.get(0) == from_range.get(0) &&
                to_range.get(0) == size.get(0)) {
                event_list.push_back(dpct_memcpy(
                    q, to_ptr, from_ptr, size_slice, direction, dep_events));
            } else {
                for (size_t y = 0; y < size.get(1); ++y) {
                    event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr,
                                                     size.get(0), direction,
                                                     dep_events));
                    to_ptr += to_range.get(0);
                    from_ptr += from_range.get(0);
                }
            }
            to_surface += to_slice;
            from_surface += from_slice;
        }
        break;
    case host_to_device: {
        host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                        event_list);
        std::vector<sycl::event> host_events;
        if (to_slice == size_slice) {
            // Copy host data to a temp host buffer with the shape of target.
            host_events = dpct_memcpy(q, buf.get_ptr(), from_surface, to_range,
                                      from_range, sycl::id<3>(0, 0, 0),
                                      sycl::id<3>(0, 0, 0), size, host_to_host,
                                      dep_events);
        } else {
            // Copy host data to a temp host buffer with the shape of target.
            host_events = dpct_memcpy(
                q, buf.get_ptr(), from_surface, to_range, from_range,
                sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_to_host,
                // If has padding data, not sure whether it is useless. So fill
                // temp buffer with it.
                std::vector<sycl::event>{
                    dpct_memcpy(q, buf.get_ptr(), to_surface, buf.get_size(),
                                device_to_host, dep_events)});
        }
        // Copy from temp host buffer to device with only one submit.
        event_list.push_back(dpct_memcpy(q, to_surface, buf.get_ptr(),
                                         buf.get_size(), host_to_device,
                                         host_events));
        break;
    }
    case device_to_host: {
        host_buffer buf(get_copy_range(size, from_slice, from_range.get(0)), q,
                        event_list);
        // Copy from host temp buffer to host target with reshaping.
        event_list = dpct_memcpy(
            q, to_surface, buf.get_ptr(), to_range, from_range,
            sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_to_host,
            // Copy from device to temp host buffer with only one submit.
            std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                 buf.get_size(), device_to_host,
                                                 dep_events)});
        break;
    }
    case device_to_device:
        event_list.push_back(q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dep_events);
            cgh.parallel_for<class dpct_memcpy_3d_detail>(size, [=](sycl::id<3>
                                                                        id) {
                to_surface[get_offset(id, to_slice, to_range.get(0))] =
                    from_surface[get_offset(id, from_slice, from_range.get(0))];
            });
        }));
        break;
    default:
        throw std::runtime_error("dpct_memcpy: invalid direction value");
    }
    return event_list;
}

/// memcpy 2D/3D matrix specified by pitched_data.
static inline std::vector<sycl::event>
dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
            pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
            memcpy_direction direction = automatic) {
    return dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                       sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                       sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id,
                       from_id, size, direction);
}

/// memcpy 2D matrix with pitch.
static inline std::vector<sycl::event>
dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr, size_t to_pitch,
            size_t from_pitch, size_t x, size_t y,
            memcpy_direction direction = automatic) {
    return dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                       sycl::range<3>(from_pitch, y, 1), sycl::id<3>(0, 0, 0),
                       sycl::id<3>(0, 0, 0), sycl::range<3>(x, y, 1),
                       direction);
}

namespace deprecated {

template <typename T, sycl::usm::alloc AllocKind> class usm_allocator {
  private:
    using Alloc = sycl::usm_allocator<T, AllocKind>;
    Alloc _impl;

  public:
    using value_type = typename std::allocator_traits<Alloc>::value_type;
    using pointer = typename std::allocator_traits<Alloc>::pointer;
    using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
    using void_pointer = typename std::allocator_traits<Alloc>::void_pointer;
    using const_void_pointer =
        typename std::allocator_traits<Alloc>::const_void_pointer;
    using reference = typename std::allocator_traits<Alloc>::value_type &;
    using const_reference =
        const typename std::allocator_traits<Alloc>::value_type &;
    using difference_type =
        typename std::allocator_traits<Alloc>::difference_type;
    using size_type = typename std::allocator_traits<Alloc>::size_type;
    using propagate_on_container_copy_assignment =
        typename std::allocator_traits<
            Alloc>::propagate_on_container_copy_assignment;
    using propagate_on_container_move_assignment =
        typename std::allocator_traits<
            Alloc>::propagate_on_container_move_assignment;
    using propagate_on_container_swap =
        typename std::allocator_traits<Alloc>::propagate_on_container_swap;
    using is_always_equal =
        typename std::allocator_traits<Alloc>::is_always_equal;

    template <typename U> struct rebind {
        typedef usm_allocator<U, AllocKind> other;
    };

    usm_allocator() : _impl(dpct::get_default_queue()) {}
    ~usm_allocator() {}
    usm_allocator(const usm_allocator &other) : _impl(other._impl) {}
    usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
    pointer address(reference r) { return &r; }
    const_pointer address(const_reference r) { return &r; }
    pointer allocate(size_type cnt, const_void_pointer hint = nullptr) {
        return std::allocator_traits<Alloc>::allocate(_impl, cnt, hint);
    }
    void deallocate(pointer p, size_type cnt) {
        std::allocator_traits<Alloc>::deallocate(_impl, p, cnt);
    }
    size_type max_size() const {
        return std::allocator_traits<Alloc>::max_size(_impl);
    }
    bool operator==(const usm_allocator &other) const {
        return _impl == other._impl;
    }
    bool operator!=(const usm_allocator &other) const {
        return _impl != other._impl;
    }
};

} // namespace deprecated

inline void dpct_free(void *ptr, const sycl::queue &q) {
    if (ptr) {
        sycl::free(ptr, q.get_context());
    }
}

template <typename T> inline auto get_memory(const void *x) {
    T *new_x = reinterpret_cast<T *>(const_cast<void *>(x));
    return new_x;
}

} // namespace detail

static sycl::event
dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr, size_t size,
            memcpy_direction direction,
            const std::vector<sycl::event> &dep_events = {}) {
    if (!size)
        return sycl::event{};
    return q.memcpy(to_ptr, from_ptr, size, dep_events);
    GGML_UNUSED(direction);
}

// Get actual copy range and make sure it will not exceed range.
static inline size_t get_copy_range(sycl::range<3> size, size_t slice,
                                    size_t pitch) {
    return slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
}

static inline size_t get_offset(sycl::id<3> id, size_t slice, size_t pitch) {
    return slice * id.get(2) + pitch * id.get(1) + id.get(0);
}

/// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
/// and \p from_range to another specified by \p to_ptr and \p to_range.
static inline std::vector<sycl::event>
dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr,
            sycl::range<3> to_range, sycl::range<3> from_range,
            sycl::id<3> to_id, sycl::id<3> from_id, sycl::range<3> size,
            memcpy_direction direction,
            const std::vector<sycl::event> &dep_events = {}) {
    // RAII for host pointer
    class host_buffer {
        void *_buf;
        size_t _size;
        sycl::queue &_q;
        const std::vector<sycl::event> &_deps; // free operation depends

      public:
        host_buffer(size_t size, sycl::queue &q,
                    const std::vector<sycl::event> &deps)
            : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
        void *get_ptr() const { return _buf; }
        size_t get_size() const { return _size; }
        ~host_buffer() {
            if (_buf) {
                _q.submit([&](sycl::handler &cgh) {
                    cgh.depends_on(_deps);
                    cgh.host_task([buf = _buf] { std::free(buf); });
                });
            }
        }
    };
    std::vector<sycl::event> event_list;

    size_t to_slice = to_range.get(1) * to_range.get(0),
           from_slice = from_range.get(1) * from_range.get(0);
    unsigned char *to_surface =
        (unsigned char *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
    const unsigned char *from_surface =
        (const unsigned char *)from_ptr +
        get_offset(from_id, from_slice, from_range.get(0));

    if (to_slice == from_slice && to_slice == size.get(1) * size.get(0)) {
        return {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                            direction, dep_events)};
    }
    direction = detail::deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
    size_t size_slice = size.get(1) * size.get(0);
    switch (direction) {
    case host_to_host:
        for (size_t z = 0; z < size.get(2); ++z) {
            unsigned char *to_ptr = to_surface;
            const unsigned char *from_ptr = from_surface;
            if (to_range.get(0) == from_range.get(0) &&
                to_range.get(0) == size.get(0)) {
                event_list.push_back(dpct_memcpy(
                    q, to_ptr, from_ptr, size_slice, direction, dep_events));
            } else {
                for (size_t y = 0; y < size.get(1); ++y) {
                    event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr,
                                                     size.get(0), direction,
                                                     dep_events));
                    to_ptr += to_range.get(0);
                    from_ptr += from_range.get(0);
                }
            }
            to_surface += to_slice;
            from_surface += from_slice;
        }
        break;
    case host_to_device: {
        host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                        event_list);
        std::vector<sycl::event> host_events;
        if (to_slice == size_slice) {
            // Copy host data to a temp host buffer with the shape of target.
            host_events = dpct_memcpy(q, buf.get_ptr(), from_surface, to_range,
                                      from_range, sycl::id<3>(0, 0, 0),
                                      sycl::id<3>(0, 0, 0), size, host_to_host,
                                      dep_events);
        } else {
            // Copy host data to a temp host buffer with the shape of target.
            host_events = dpct_memcpy(
                q, buf.get_ptr(), from_surface, to_range, from_range,
                sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_to_host,
                // If has padding data, not sure whether it is useless. So fill
                // temp buffer with it.
                std::vector<sycl::event>{
                    dpct_memcpy(q, buf.get_ptr(), to_surface, buf.get_size(),
                                device_to_host, dep_events)});
        }
        // Copy from temp host buffer to device with only one submit.
        event_list.push_back(dpct_memcpy(q, to_surface, buf.get_ptr(),
                                         buf.get_size(), host_to_device,
                                         host_events));
        break;
    }
    case device_to_host: {
        host_buffer buf(get_copy_range(size, from_slice, from_range.get(0)), q,
                        event_list);
        // Copy from host temp buffer to host target with reshaping.
        event_list = dpct_memcpy(
            q, to_surface, buf.get_ptr(), to_range, from_range,
            sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_to_host,
            // Copy from device to temp host buffer with only one submit.
            std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                 buf.get_size(), device_to_host,
                                                 dep_events)});
        break;
    }
    case device_to_device:
        event_list.push_back(q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dep_events);
            cgh.parallel_for<class dpct_memcpy_3d_detail>(size, [=](sycl::id<3>
                                                                        id) {
                to_surface[get_offset(id, to_slice, to_range.get(0))] =
                    from_surface[get_offset(id, from_slice, from_range.get(0))];
            });
        }));
        break;
    default:
        throw std::runtime_error("dpct_memcpy: invalid direction value");
    }
    return event_list;
}

/// memcpy 2D/3D matrix specified by pitched_data.
static inline std::vector<sycl::event>
dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
            pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
            memcpy_direction direction = automatic) {
    return dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                       sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                       sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id,
                       from_id, size, direction);
}

/// memcpy 2D matrix with pitch.
static inline std::vector<sycl::event>
dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr, size_t to_pitch,
            size_t from_pitch, size_t x, size_t y,
            memcpy_direction direction = automatic) {
    return dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                       sycl::range<3>(from_pitch, y, 1), sycl::id<3>(0, 0, 0),
                       sycl::id<3>(0, 0, 0), sycl::range<3>(x, y, 1),
                       direction);
}

static inline void async_dpct_memcpy(void *to_ptr, size_t to_pitch,
                                     const void *from_ptr, size_t from_pitch,
                                     size_t x, size_t y,
                                     memcpy_direction direction = automatic,
                                     sycl::queue &q = get_default_queue()) {
    detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch, from_pitch, x, y,
                        direction);
}

static void async_dpct_memcpy(void *to_ptr, const void *from_ptr, size_t size,
                              memcpy_direction direction = automatic,
                              sycl::queue &q = dpct::get_default_queue()) {
    detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction);
}

static inline void dpct_free(void *ptr, sycl::queue &q = get_default_queue()) {
    detail::dpct_free(ptr, q);
}

/// dpct accessor used as device function parameter.
template <class T, memory_region Memory, size_t Dimension> class accessor;
template <class T, memory_region Memory> class accessor<T, Memory, 3> {
  public:
    using memory_t = detail::memory_traits<Memory, T>;
    using element_t = typename memory_t::element_t;
    using pointer_t = typename memory_t::pointer_t;
    using accessor_t = typename memory_t::template accessor_t<3>;
    accessor(pointer_t data, const sycl::range<3> &in_range)
        : _data(data), _range(in_range) {}
    template <memory_region M = Memory>
    accessor(typename std::enable_if<M != local, const accessor_t>::type &acc)
        : accessor(acc, acc.get_range()) {}
    accessor(const accessor_t &acc, const sycl::range<3> &in_range)
        : accessor(acc.get_pointer(), in_range) {}
    accessor<T, Memory, 2> operator[](size_t index) const {
        sycl::range<2> sub(_range.get(1), _range.get(2));
        return accessor<T, Memory, 2>(_data + index * sub.size(), sub);
    }

    pointer_t get_ptr() const { return _data; }

  private:
    pointer_t _data;
    sycl::range<3> _range;
};
template <class T, memory_region Memory> class accessor<T, Memory, 2> {
  public:
    using memory_t = detail::memory_traits<Memory, T>;
    using element_t = typename memory_t::element_t;
    using pointer_t = typename memory_t::pointer_t;
    using accessor_t = typename memory_t::template accessor_t<2>;
    accessor(pointer_t data, const sycl::range<2> &in_range)
        : _data(data), _range(in_range) {}
    template <memory_region M = Memory>
    accessor(typename std::enable_if<M != local, const accessor_t>::type &acc)
        : accessor(acc, acc.get_range()) {}
    accessor(const accessor_t &acc, const sycl::range<2> &in_range)
        : accessor(acc.get_pointer(), in_range) {}

    pointer_t operator[](size_t index) const {
        return _data + _range.get(1) * index;
    }

    pointer_t get_ptr() const { return _data; }

  private:
    pointer_t _data;
    sycl::range<2> _range;
};

namespace detail {
/// Device variable with address space of shared, global or constant.
template <class T, memory_region Memory, size_t Dimension> class device_memory {
  public:
    using accessor_t =
        typename detail::memory_traits<Memory,
                                       T>::template accessor_t<Dimension>;
    using value_t = typename detail::memory_traits<Memory, T>::value_t;
    using dpct_accessor_t = dpct::accessor<T, Memory, Dimension>;

    device_memory() : device_memory(sycl::range<Dimension>(1)) {}

    /// Constructor of 1-D array with initializer list
    device_memory(const sycl::range<Dimension> &in_range,
                  std::initializer_list<value_t> &&init_list)
        : device_memory(in_range) {
        assert(init_list.size() <= in_range.size());
        _host_ptr = (value_t *)std::malloc(_size);
        std::memset(_host_ptr, 0, _size);
        std::memcpy(_host_ptr, init_list.begin(), init_list.size() * sizeof(T));
    }

    /// Constructor of 2-D array with initializer list
    template <size_t D = Dimension>
    device_memory(
        const typename std::enable_if<D == 2, sycl::range<2>>::type &in_range,
        std::initializer_list<std::initializer_list<value_t>> &&init_list)
        : device_memory(in_range) {
        assert(init_list.size() <= in_range[0]);
        _host_ptr = (value_t *)std::malloc(_size);
        std::memset(_host_ptr, 0, _size);
        auto tmp_data = _host_ptr;
        for (auto sub_list : init_list) {
            assert(sub_list.size() <= in_range[1]);
            std::memcpy(tmp_data, sub_list.begin(),
                        sub_list.size() * sizeof(T));
            tmp_data += in_range[1];
        }
    }

    /// Constructor with range
    device_memory(const sycl::range<Dimension> &range_in)
        : _size(range_in.size() * sizeof(T)), _range(range_in),
          _reference(false), _host_ptr(nullptr), _device_ptr(nullptr) {
        static_assert(
            (Memory == global) || (Memory == constant) || (Memory == shared),
            "device memory region should be global, constant or shared");
        // Make sure that singleton class mem_mgr and dev_mgr will destruct
        // later than this.
        detail::mem_mgr::instance();
        dev_mgr::instance();
    }

    /// Constructor with range
    template <class... Args>
    device_memory(Args... Arguments)
        : device_memory(sycl::range<Dimension>(Arguments...)) {}

    ~device_memory() {
        if (_device_ptr && !_reference)
            dpct::dpct_free(_device_ptr);
        if (_host_ptr)
            std::free(_host_ptr);
    }

    /// Allocate memory with default queue, and init memory if has initial
    /// value.
    void init() { init(dpct::get_default_queue()); }
    /// Allocate memory with specified queue, and init memory if has initial
    /// value.
    void init(sycl::queue &q) {
        if (_device_ptr)
            return;
        if (!_size)
            return;
        allocate_device(q);
        if (_host_ptr)
            detail::dpct_memcpy(q, _device_ptr, _host_ptr, _size,
                                host_to_device);
    }

    /// The variable is assigned to a device pointer.
    void assign(value_t *src, size_t size) {
        this->~device_memory();
        new (this) device_memory(src, size);
    }

    /// Get memory pointer of the memory object, which is virtual pointer when
    /// usm is not used, and device pointer when usm is used.
    value_t *get_ptr() { return get_ptr(get_default_queue()); }
    /// Get memory pointer of the memory object, which is virtual pointer when
    /// usm is not used, and device pointer when usm is used.
    value_t *get_ptr(sycl::queue &q) {
        init(q);
        return _device_ptr;
    }

    /// Get the device memory object size in bytes.
    size_t get_size() { return _size; }

    template <size_t D = Dimension>
    typename std::enable_if<D == 1, T>::type &operator[](size_t index) {
        init();
        return _device_ptr[index];
    }

    /// Get dpct::accessor with dimension info for the device memory object
    /// when usm is used and dimension is greater than 1.
    template <size_t D = Dimension>
    typename std::enable_if<D != 1, dpct_accessor_t>::type
    get_access([[maybe_unused]] sycl::handler &cgh) {
        return dpct_accessor_t((T *)_device_ptr, _range);
    }

  private:
    device_memory(value_t *memory_ptr, size_t size)
        : _size(size), _range(size / sizeof(T)), _reference(true),
          _device_ptr(memory_ptr) {}

    void allocate_device(sycl::queue &q) {
#ifndef DPCT_USM_LEVEL_NONE
        if (Memory == shared) {
            _device_ptr = (value_t *)sycl::malloc_shared(_size, q.get_device(),
                                                         q.get_context());
            return;
        }
#ifdef SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY
        if (Memory == constant) {
            _device_ptr = (value_t *)sycl::malloc_device(
                _size, q.get_device(), q.get_context(),
                sycl::ext::oneapi::property::usm::device_read_only());
            return;
        }
#endif
#endif
        _device_ptr = (value_t *)detail::dpct_malloc(_size, q);
    }

    size_t _size;
    sycl::range<Dimension> _range;
    bool _reference;
    value_t *_host_ptr;
    value_t *_device_ptr;
};
template <class T, memory_region Memory>
class device_memory<T, Memory, 0> : public device_memory<T, Memory, 1> {
  public:
    using base = device_memory<T, Memory, 1>;
    using value_t = typename base::value_t;
    using accessor_t =
        typename detail::memory_traits<Memory, T>::template accessor_t<0>;

    /// Constructor with initial value.
    device_memory(const value_t &val) : base(sycl::range<1>(1), {val}) {}

    /// Default constructor
    device_memory() : base(1) {}
};
} // namespace detail

template <class T, size_t Dimension>
using global_memory = detail::device_memory<T, global, Dimension>;
template <class T, size_t Dimension>
using constant_memory = detail::device_memory<T, constant, Dimension>;
template <class T, size_t Dimension>
using shared_memory = detail::device_memory<T, shared, Dimension>;
} // namespace dpct
