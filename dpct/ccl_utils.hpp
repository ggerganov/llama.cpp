//==---- ccl_utils.hpp----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_CCL_UTILS_HPP__
#define __DPCT_CCL_UTILS_HPP__

#include <sycl/sycl.hpp>
#include <oneapi/ccl.hpp>
#include <unordered_map>
#include <memory>

#include "device.hpp"

namespace dpct {
namespace ccl {
namespace detail {

/// Get stored kvs with specified kvs address.
inline std::shared_ptr<oneapi::ccl::kvs> &
get_kvs(const oneapi::ccl::kvs::address_type &addr) {
  struct hash {
    std::size_t operator()(const oneapi::ccl::kvs::address_type &in) const {
      return std::hash<std::string_view>()(std::string_view(in.data(), in.size()));
    }
  };
  static std::unordered_map<oneapi::ccl::kvs::address_type,
                            std::shared_ptr<oneapi::ccl::kvs>, hash>
      kvs_map;
  return kvs_map[addr];
}

/// Help class to init ccl environment. 
class ccl_init_helper {
public:
  ccl_init_helper() { oneapi::ccl::init(); }
};

} // namespace detail

/// Get concatenated library version as an integer.
static inline int get_version() {
  oneapi::ccl::init();
  auto ver = oneapi::ccl::get_library_version();
  return ver.major * 10000 + ver.minor * 100 + ver.update;
}

/// Create main kvs and return its address.
static inline oneapi::ccl::kvs::address_type create_kvs_address() {
  oneapi::ccl::init();
  auto ptr = oneapi::ccl::create_main_kvs();
  auto addr = ptr->get_address();
  detail::get_kvs(addr) = ptr;
  return addr;
}

/// Get stored kvs with /p addr if exist. Otherwise, create kvs with /p addr.
static inline std::shared_ptr<oneapi::ccl::kvs>
create_kvs(const oneapi::ccl::kvs::address_type &addr) {
  oneapi::ccl::init();
  auto &ptr = detail::get_kvs(addr);
  if (!ptr)
    ptr = oneapi::ccl::create_kvs(addr);
  return ptr;
}

/// dpct communicator extension
class communicator_wrapper : public dpct::ccl::detail::ccl_init_helper {
public:
  communicator_wrapper(
      int size, int rank, oneapi::ccl::kvs::address_type id,
      const oneapi::ccl::comm_attr &attr = oneapi::ccl::default_comm_attr)
      : _device_comm(oneapi::ccl::create_device(
            static_cast<sycl::device &>(dpct::get_current_device()))),
        _context_comm(oneapi::ccl::create_context(dpct::get_default_context())),
        _comm(oneapi::ccl::create_communicator(
            size, rank, _device_comm, _context_comm, dpct::ccl::create_kvs(id),
            attr)) {
    _queue_init = false;
    _ccl_stream_ptr = nullptr;
  }

  ~communicator_wrapper() {
    delete _ccl_stream_ptr;
  };

  /// Return the rank in a oneapi::ccl::communicator
  /// \returns The rank corresponding to communicator object
  int rank() const {
    return _comm.rank();
  }

  /// Retrieves the number of rank in oneapi::ccl::communicator
  /// \returns The number of the ranks
  int size() const {
    return _comm.size();
  }

  /// Return underlying native device, which was used in oneapi::ccl::communicator
  sycl::device get_device() const {
    return _comm.get_device().get_native();
  }

  /// \brief allreduce is a collective communication operation that performs the global reduction operation
  ///       on values from all ranks of communicator and distributes the result back to all ranks.
  /// \param sendbuff the buffer with @c count elements of @c dtype that stores local data to be reduced
  /// \param recvbuff [out] the buffer to store reduced result, must have the same dimension as @c sendbuff
  /// \param count the number of elements of type @c dtype in @c sendbuff and @c recvbuff
  /// \param dtype the datatype of elements in @c sendbuff and @c recvbuff
  /// \param rtype the type of the reduction operation to be applied
  /// \param queue_ptr a sycl::queue ptr associated with the operation
  /// \return @ref void
  void allreduce(const void *sendbuff, void *recvbuff, size_t count,
                 oneapi::ccl::datatype dtype, oneapi::ccl::reduction rtype,
                 sycl::queue *queue_ptr) {
    call_func_wrapper(
        [=](const oneapi::ccl::stream &stream) {
          return oneapi::ccl::allreduce(sendbuff, recvbuff, count, dtype, rtype,
                                        _comm, stream);
        },
        queue_ptr);
  }

  /// \brief reduce is a collective communication operation that performs the
  ///        global reduction operation on values from all ranks of the communicator
  ///        and returns the result to the root rank.
  /// \param sendbuff the buffer with @c count elements of @c dtype that stores
  ///        local data to be reduced 
  /// \param recvbuff [out] the buffer to store reduced result, 
  ///        must have the same dimension as @c sendbuff 
  /// \param count the number of elements of type @c dtype in @c sendbuff and @c recvbuff 
  /// \param dtype the datatype of elements in @c sendbuff and @c recvbuff 
  /// \param root the rank that gets the result of reduction 
  /// \param rtype the type of the reduction operation to be applied 
  /// \param queue_ptr a sycl::queue ptr associated with the operation 
  /// \return @ref void
  void reduce(const void *sendbuff, void *recvbuff, size_t count,
              oneapi::ccl::datatype dtype, oneapi::ccl::reduction rtype,
              int root, sycl::queue *queue_ptr) {
    call_func_wrapper(
        [=](const oneapi::ccl::stream &stream) {
          return oneapi::ccl::reduce(sendbuff, recvbuff, count, dtype, rtype,
                                     root, _comm, stream);
        },
        queue_ptr);
  }

  /// \brief broadcast is a collective communication operation that broadcasts data
  ///        from one rank of communicator (denoted as root) to all other ranks.
  ///        Only support in-place operation
  /// \param sendbuff the buffer with @c count elements of @c dtype that stores
  ///        local data to be reduced 
  /// \param recvbuff [out] the buffer to store reduced result
  /// \param count the number of elements of type @c dtype in @c buf 
  /// \param dtype thedatatype of elements in @c buf 
  /// \param root the rank that broadcasts @c buf
  /// \param queue_ptr a sycl::queue ptr associated with the operation
  /// \return @ref void
  void broadcast(void *sendbuff, void *recvbuff, size_t count,
                 oneapi::ccl::datatype dtype, int root,
                 sycl::queue *queue_ptr) {
    if (sendbuff != recvbuff) {
      throw std::runtime_error(
          "oneCCL broadcast only support in-place operation. "
          "sendbuff and recvbuff must be same.");
      return;
    }
    call_func_wrapper(
        [=](const oneapi::ccl::stream &stream) {
          return oneapi::ccl::broadcast(recvbuff, count, dtype, root, _comm,
                                        stream);
        },
        queue_ptr);
  }

  /// \brief reduce_scatter is a collective communication operation that performs the global reduction operation
  ///        on values from all ranks of the communicator and scatters the result in blocks back to all ranks.
  /// \param sendbuff the buffer with @c count elements of @c dtype that stores local data to be reduced
  /// \param recvbuff [out] the buffer to store reduced result, must have the same dimension as @c sendbuff
  /// \param recv_count the number of elements of type @c dtype in receive block
  /// \param dtype the datatype of elements in @c sendbuff and @c recvbuff
  /// \param rtype the type of the reduction operation to be applied
  /// \param queue_ptr a sycl::queue ptr associated with the operation
  /// \return @ref void
  void reduce_scatter(const void *sendbuff, void *recvbuff, size_t recv_count,
                      oneapi::ccl::datatype dtype, oneapi::ccl::reduction rtype,
                      sycl::queue *queue_ptr) {
    call_func_wrapper(
        [=](const oneapi::ccl::stream &stream) {
          return oneapi::ccl::reduce_scatter(sendbuff, recvbuff, recv_count,
                                             dtype, rtype, _comm, stream);
        },
        queue_ptr);
  }

  /// \brief send is a pt2pt communication operation that sends data from one rank of communicator.
  /// \param sendbuff the buffer with @c count elements of @c dtype serves as send buffer for root
  /// \param count the number of elements of type @c dtype in @c sendbuff
  /// \param dtype the datatype of elements in @c sendbuff
  /// \param peer the rank that receives @c sendbuff
  /// \param queue_ptr a sycl::queue ptr associated with the operation
  /// \return @ref void
  void send(void *sendbuff, size_t count, oneapi::ccl::datatype dtype, int peer,
            sycl::queue *queue_ptr) {
    call_func_wrapper(
        [=](const oneapi::ccl::stream &stream) {
          return oneapi::ccl::send(sendbuff, count, dtype, peer, _comm, stream);
        },
        queue_ptr);
  }

  /// \brief recv is a pt2pt communication operation that sends data from one rank of communicator.
  /// \param recvbuff the buffer with @c count elements of @c dtype serves as  receive buffer
  /// \param count the number of elements of type @c dtype in @c recvbuff
  /// \param dtype the datatype of elements in @c recvbuff
  /// \param peer the rank that receives @c recvbuff
  /// \param queue_ptr a sycl::queue ptr associated with the operation
  /// \return @ref void
  void recv(void *recvbuff, size_t count, oneapi::ccl::datatype dtype, int peer,
            sycl::queue *queue_ptr) {
    call_func_wrapper(
        [=](const oneapi::ccl::stream &stream) {
          return oneapi::ccl::recv(recvbuff, count, dtype, peer, _comm, stream);
        },
        queue_ptr);
  }

private:
  oneapi::ccl::device _device_comm;
  oneapi::ccl::context _context_comm;
  oneapi::ccl::communicator _comm;
  sycl::queue _queue;
  bool _queue_init;
  oneapi::ccl::stream *_ccl_stream_ptr;

  template <class Fn>
  void call_func_wrapper(Fn func, sycl::queue *qptr) {
    if (_queue_init && *qptr != _queue) {
      call_func_async(func, qptr);
    } else {
      if(!_queue_init) {
        _queue = *qptr;
        _queue_init = true;
        _ccl_stream_ptr = new oneapi::ccl::stream(oneapi::ccl::create_stream(_queue));
      }
      std::invoke(func, *_ccl_stream_ptr);
    }
  }

  class call_func_async {
    sycl::queue *_q_ptr;
    struct call_async_impl {
      oneapi::ccl::stream _ccl_stream_impl;
      oneapi::ccl::event _ccl_event_impl;
      template <class Fn>
      explicit call_async_impl(Fn func, sycl::queue *qptr)
          : _ccl_stream_impl(oneapi::ccl::create_stream(*qptr)),
            _ccl_event_impl(std::invoke(func, _ccl_stream_impl)) {}
    };
    call_async_impl *_imp;

  public:
    template <class Fn>
    explicit call_func_async(Fn func, sycl::queue *qptr)
        : _q_ptr(qptr),
          _imp(new call_async_impl(func, qptr)) {}
    ~call_func_async() {
      _q_ptr->submit([&](sycl::handler &cgh)
                     { cgh.host_task([=]
                                     {
        _imp->_ccl_event_impl.wait();
        delete _imp; }); });
    }
  };
};

typedef dpct::ccl::communicator_wrapper *comm_ptr;

} // namespace ccl
} // namespace dpct

#endif // __DPCT_CCL_UTILS_HPP__