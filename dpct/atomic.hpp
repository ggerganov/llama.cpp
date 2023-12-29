//==---- atomic.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_ATOMIC_HPP__
#define __DPCT_ATOMIC_HPP__

#include <sycl/sycl.hpp>

namespace dpct {

/// Atomically add the value operand to the value at the addr and assign the
/// result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to add to the value at \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_add(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_add(operand);
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomic_fetch_add(T1 *addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_add(operand);
}

/// Atomically add the value operand to the value at the addr and assign the
/// result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to add to the value at \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_add(T *addr, T operand,
                          sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_add<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_add<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_add<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          typename T1, typename T2>
inline T1 atomic_fetch_add(T1 *addr, T2 operand,
                           sycl::memory_order memoryOrder) {
  atomic_fetch_add<T1, addressSpace>(addr, operand, memoryOrder);
}

/// Atomically subtract the value operand from the value at the addr and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to subtract from the value at \p addr
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_sub(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_sub(operand);
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomic_fetch_sub(T1 *addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_sub(operand);
}

/// Atomically subtract the value operand from the value at the addr and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to subtract from the value at \p addr
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_sub(T *addr, T operand,
                          sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_sub<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_sub<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_sub<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          typename T1, typename T2>
inline T1 atomic_fetch_sub(T1 *addr, T2 operand,
                           sycl::memory_order memoryOrder) {
  atomic_fetch_sub<T1, addressSpace>(addr, operand, memoryOrder);
}

/// Atomically perform a bitwise AND between the value operand and the value at the addr
/// and assign the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise AND operation with the value at the \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_and(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_and(operand);
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomic_fetch_and(T1 *addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_and(operand);
}

/// Atomically perform a bitwise AND between the value operand and the value at the addr
/// and assign the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise AND operation with the value at the \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_and(T *addr, T operand,
                          sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_and<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_and<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_and<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          typename T1, typename T2>
inline T1 atomic_fetch_and(T1 *addr, T2 operand,
                           sycl::memory_order memoryOrder) {
  atomic_fetch_and<T1, addressSpace>(addr, operand, memoryOrder);
}

/// Atomically or the value at the addr with the value operand, and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise OR operation with the value at the \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_or(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_or(operand);
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomic_fetch_or(T1 *addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_or(operand);
}

/// Atomically or the value at the addr with the value operand, and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise OR operation with the value at the \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_or(T *addr, T operand,
                         sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_or<T, addressSpace, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_or<T, addressSpace, sycl::memory_order::acq_rel,
                           sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_or<T, addressSpace, sycl::memory_order::seq_cst,
                           sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          typename T1, typename T2>
inline T1 atomic_fetch_or(T1 *addr, T2 operand,
                           sycl::memory_order memoryOrder) {
  atomic_fetch_or<T1, addressSpace>(addr, operand, memoryOrder);
}

/// Atomically xor the value at the addr with the value operand, and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise XOR operation with the value at the \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_xor(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_xor(operand);
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomic_fetch_xor(T1 *addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_xor(operand);
}

/// Atomically xor the value at the addr with the value operand, and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise XOR operation with the value at the \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_xor(T *addr, T operand,
                          sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_xor<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_xor<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_xor<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          typename T1, typename T2>
inline T1 atomic_fetch_xor(T1 *addr, T2 operand,
                           sycl::memory_order memoryOrder) {
  atomic_fetch_xor<T1, addressSpace>(addr, operand, memoryOrder);
}

/// Atomically calculate the minimum of the value at addr and the value operand
/// and assign the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_min(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_min(operand);
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomic_fetch_min(T1 *addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_min(operand);
}

/// Atomically calculate the minimum of the value at addr and the value operand
/// and assign the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_min(T *addr, T operand,
                          sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_min<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_min<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_min<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          typename T1, typename T2>
inline T1 atomic_fetch_min(T1 *addr, T2 operand,
                           sycl::memory_order memoryOrder) {
  atomic_fetch_min<T1, addressSpace>(addr, operand, memoryOrder);
}

/// Atomically calculate the maximum of the value at addr and the value operand
/// and assign the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_max(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_max(operand);
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomic_fetch_max(T1 *addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_max(operand);
}

/// Atomically calculate the maximum of the value at addr and the value operand
/// and assign the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_max(T *addr, T operand,
                          sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_max<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_max<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_max<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          typename T1, typename T2>
inline T1 atomic_fetch_max(T1 *addr, T2 operand,
                           sycl::memory_order memoryOrder) {
  atomic_fetch_max<T1, addressSpace>(addr, operand, memoryOrder);
}

/// Atomically set \p operand to the value stored in \p addr, if old value stored in
/// \p addr is equal to zero or greater than \p operand, else decrease the value stored
/// in \p addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The threshold value.
/// \param memoryOrder The memory ordering used.
/// \returns The old value stored in \p addr.
template <sycl::access::address_space addressSpace = sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline unsigned int atomic_fetch_compare_dec(unsigned int *addr,
                                             unsigned int operand) {
  auto atm = sycl::atomic_ref<unsigned int, memoryOrder, memoryScope,
                                  addressSpace>(addr[0]);
  unsigned int old;

	while (true) {
	  old = atm.load();
	  if (old == 0 || old > operand) {
		  if (atm.compare_exchange_strong(old, operand))
        break;
	  } else if (atm.compare_exchange_strong(old, old - 1))
      break;
	}

  return old;
}

/// Atomically increment the value stored in \p addr if old value stored in \p
/// addr is less than \p operand, else set 0 to the value stored in \p addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The threshold value.
/// \param memoryOrder The memory ordering used.
/// \returns The old value stored in \p addr.
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline unsigned int atomic_fetch_compare_inc(unsigned int *addr,
                                             unsigned int operand) {
  auto atm = sycl::atomic_ref<unsigned int, memoryOrder, memoryScope,
                                  addressSpace>(addr[0]);
  unsigned int old;
  while (true) {
    old = atm.load();
    if (old >= operand) {
      if (atm.compare_exchange_strong(old, 0))
        break;
    } else if (atm.compare_exchange_strong(old, old + 1))
      break;
  }
  return old;
}

/// Atomically increment the value stored in \p addr if old value stored in \p
/// addr is less than \p operand, else set 0 to the value stored in \p addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The threshold value.
/// \param memoryOrder The memory ordering used.
/// \returns The old value stored in \p addr.
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space>
inline unsigned int
atomic_fetch_compare_inc(unsigned int *addr, unsigned int operand,
                         sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_compare_inc<addressSpace, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device>(addr,
                                                                   operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_compare_inc<addressSpace, sycl::memory_order::acq_rel,
                                    sycl::memory_scope::device>(addr,
                                                                   operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_compare_inc<addressSpace, sycl::memory_order::seq_cst,
                                    sycl::memory_scope::device>(addr,
                                                                   operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

/// Atomically exchange the value at the address addr with the value operand.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to be exchanged with the value pointed by \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_exchange(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.exchange(operand);
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomic_exchange(T1 *addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.exchange(operand);
}

/// Atomically exchange the value at the address addr with the value operand.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to be exchanged with the value pointed by \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_exchange(T *addr, T operand,
                         sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_exchange<T, addressSpace, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_exchange<T, addressSpace, sycl::memory_order::acq_rel,
                           sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_exchange<T, addressSpace, sycl::memory_order::seq_cst,
                           sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          typename T1, typename T2>
inline T1 atomic_exchange(T1 *addr, T2 operand,
                           sycl::memory_order memoryOrder) {
  atomic_exchange<T1, addressSpace>(addr, operand, memoryOrder);
}

/// Atomically compare the value at \p addr to the value expected and exchange
/// with the value desired if the value at \p addr is equal to the value expected.
/// Returns the value at the \p addr before the call.
/// \param [in, out] addr Multi_ptr.
/// \param expected The value to compare against the value at \p addr.
/// \param desired The value to assign to \p addr if the value at \p addr is expected.
/// \param success The memory ordering used when comparison succeeds.
/// \param fail The memory ordering used when comparison fails.
/// \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
T atomic_compare_exchange_strong(
    sycl::multi_ptr<T, addressSpace> addr, T expected, T desired,
    sycl::memory_order success = sycl::memory_order::relaxed,
    sycl::memory_order fail = sycl::memory_order::relaxed) {
  auto atm = sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(*addr);

  atm.compare_exchange_strong(expected, desired, success, fail);
  return expected;
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2, typename T3>
T1 atomic_compare_exchange_strong(
    sycl::multi_ptr<T1, addressSpace> addr, T2 expected, T3 desired,
    sycl::memory_order success = sycl::memory_order::relaxed,
    sycl::memory_order fail = sycl::memory_order::relaxed) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(*addr);
  T1 expected_value = expected;
  atm.compare_exchange_strong(expected_value, desired, success, fail);
  return expected_value;
}

/// Atomically compare the value at \p addr to the value expected and exchange
/// with the value desired if the value at \p addr is equal to the value expected.
/// Returns the value at the \p addr before the call.
/// \param [in] addr The pointer to the data.
/// \param expected The value to compare against the value at \p addr.
/// \param desired The value to assign to \p addr if the value at \p addr is expected.
/// \param success The memory ordering used when comparison succeeds.
/// \param fail The memory ordering used when comparison fails.
/// \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
T atomic_compare_exchange_strong(
    T *addr, T expected, T desired,
    sycl::memory_order success = sycl::memory_order::relaxed,
    sycl::memory_order fail = sycl::memory_order::relaxed) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  atm.compare_exchange_strong(expected, desired, success, fail);
  return expected;
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2, typename T3>
T1 atomic_compare_exchange_strong(
    T1 *addr, T2 expected, T3 desired,
    sycl::memory_order success = sycl::memory_order::relaxed,
    sycl::memory_order fail = sycl::memory_order::relaxed) {
  T1 expected_value = expected;
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  atm.compare_exchange_strong(expected_value, desired, success, fail);
  return expected_value;
}

/// Atomic extension to implement standard APIs in std::atomic
namespace detail{
template <typename T> struct IsValidAtomicType {
  static constexpr bool value =
      (std::is_same<T, int>::value || std::is_same<T, unsigned int>::value ||
       std::is_same<T, long>::value || std::is_same<T, unsigned long>::value ||
       std::is_same<T, long long>::value ||
       std::is_same<T, unsigned long long>::value ||
       std::is_same<T, float>::value || std::is_same<T, double>::value ||
       std::is_pointer<T>::value);
};
} // namespace detail

template <typename T,
          sycl::memory_scope DefaultScope = sycl::memory_scope::system,
          sycl::memory_order DefaultOrder = sycl::memory_order::seq_cst,
          sycl::access::address_space Space =
              sycl::access::address_space::generic_space>
class atomic{
  static_assert(
    detail::IsValidAtomicType<T>::value,
    "Invalid atomic type.  Valid types are int, unsigned int, long, "
      "unsigned long, long long, unsigned long long, float, double "
      "and pointer types");
  T __d;

public:
  /// default memory synchronization order
  static constexpr sycl::memory_order default_read_order =
      sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space>::default_read_order;
  static constexpr sycl::memory_order default_write_order =
      sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space>::default_write_order;
  static constexpr sycl::memory_scope default_scope = DefaultScope;
  static constexpr sycl::memory_order default_read_modify_write_order =
      DefaultOrder;
  

  /// Default constructor.
  constexpr atomic() noexcept = default;
  /// Constructor with initialize value.
  constexpr atomic(T d) noexcept : __d(d){};

  /// atomically replaces the value of the referenced object with a non-atomic argument
  /// \param operand The value to replace the pointed value.
  /// \param memoryOrder The memory ordering used.
  /// \param memoryScope The memory scope used.
  void store(T operand, sycl::memory_order memoryOrder = default_write_order,
             sycl::memory_scope memoryScope = default_scope) noexcept {
    sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    atm.store(operand, memoryOrder, memoryScope);
  }

  /// atomically obtains the value of the referenced object
  /// \param memoryOrder The memory ordering used.
  /// \param memoryScope The memory scope used.
  /// \returns The value of the referenced object
  T load(sycl::memory_order memoryOrder = default_read_order,
         sycl::memory_scope memoryScope = default_scope) const noexcept {
    sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(
      const_cast<T &>(__d));
    return atm.load(memoryOrder, memoryScope);
  }

  /// atomically replaces the value of the referenced object and obtains the value held previously
  /// \param operand The value to replace the pointed value.
  /// \param memoryOrder The memory ordering used.
  /// \param memoryScope The memory scope used.
  /// \returns The value of the referenced object before the call.
  T exchange(T operand,
             sycl::memory_order memoryOrder = default_read_modify_write_order,
             sycl::memory_scope memoryScope = default_scope) noexcept {

    sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.exchange(operand, memoryOrder, memoryScope);
  }

  /// atomically compares the value of the referenced object with non-atomic argument 
  /// and performs atomic exchange if equal or atomic load if not
  /// \param expected The value expected to be found in the object referenced by the atomic_ref object
  /// \param desired  The value to store in the referenced object if it is as expected
  /// \param success The memory models for the read-modify-write
  /// \param failure The memory models for load operations
  /// \param memoryScope The memory scope used.
  /// \returns true if the referenced object was successfully changed, false otherwise.
  bool compare_exchange_weak(
      T &expected, T desired,
      sycl::memory_order success, sycl::memory_order failure,
      sycl::memory_scope memoryScope = default_scope) noexcept {
    sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.compare_exchange_weak(expected, desired, success, failure, memoryScope);
  }
  /// \param expected The value expected to be found in the object referenced by the atomic_ref object
  /// \param desired  The value to store in the referenced object if it is as expected
  /// \param memoryOrder 	The memory synchronization ordering for operations
  /// \param memoryScope The memory scope used.
  /// \returns true if the referenced object was successfully changed, false otherwise.
  bool compare_exchange_weak(T &expected, T desired,
                  sycl::memory_order memoryOrder = default_read_modify_write_order,
                  sycl::memory_scope memoryScope = default_scope) noexcept {
    sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.compare_exchange_weak(expected, desired, memoryOrder, memoryScope);
  }

  /// atomically compares the value of the referenced object with non-atomic argument 
  /// and performs atomic exchange if equal or atomic load if not
  /// \param expected The value expected to be found in the object referenced by the atomic_ref object
  /// \param desired  The value to store in the referenced object if it is as expected
  /// \param success The memory models for the read-modify-write
  /// \param failure The memory models for load operations
  /// \param memoryScope The memory scope used.
  /// \returns true if the referenced object was successfully changed, false otherwise.
  bool compare_exchange_strong(
      T &expected, T desired,
      sycl::memory_order success, sycl::memory_order failure,
      sycl::memory_scope memoryScope = default_scope) noexcept {

    sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.compare_exchange_strong(expected, desired, success, failure, memoryScope);
  }
  /// \param expected The value expected to be found in the object referenced by the atomic_ref object
  /// \param desired  The value to store in the referenced object if it is as expected
  /// \param memoryOrder 	The memory synchronization ordering for operations
  /// \param memoryScope The memory scope used.
  /// \returns true if the referenced object was successfully changed, false otherwise.
  bool compare_exchange_strong(T &expected, T desired,
                    sycl::memory_order memoryOrder = default_read_modify_write_order,
                    sycl::memory_scope memoryScope = default_scope) noexcept {
    sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.compare_exchange_strong(expected, desired, memoryOrder, memoryScope);
  }

  /// atomically adds the argument to the value stored in the atomic object and obtains the value held previously
  /// \param operand 	The other argument of arithmetic addition
  /// \param memoryOrder The memory ordering used.
  /// \param memoryScope The memory scope used.
  /// \returns The value of the referenced object before the call.
  T fetch_add(T operand,
              sycl::memory_order memoryOrder = default_read_modify_write_order,
              sycl::memory_scope  memoryScope = default_scope) noexcept {

    sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.fetch_add(operand, memoryOrder,  memoryScope);
  }

  /// atomically subtracts the argument from the value stored in the atomic object and obtains the value held previously
  /// \param operand 	The other argument of arithmetic subtraction
  /// \param memoryOrder The memory ordering used.
  /// \param memoryScope The memory scope used.
  /// \returns The value of the referenced object before the call.
  T fetch_sub(T operand,
              sycl::memory_order memoryOrder = default_read_modify_write_order,
              sycl::memory_scope memoryScope = default_scope) noexcept {

    sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.fetch_sub(operand, memoryOrder, memoryScope);
  }
};

} // namespace dpct
#endif // __DPCT_ATOMIC_HPP__
