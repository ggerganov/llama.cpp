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

#ifndef GGML_SYCL_DPCT_HELPER_HPP
#define GGML_SYCL_DPCT_HELPER_HPP

#include <sstream>

#include <sycl/sycl.hpp>

inline std::string get_device_type_name(const sycl::device &Device) {
    auto DeviceType = Device.get_info<sycl::info::device::device_type>();
    switch (DeviceType) {
    case sycl::info::device_type::cpu:
        return "cpu";
    case sycl::info::device_type::gpu:
        return "gpu";
    case sycl::info::device_type::host:
        return "host";
    case sycl::info::device_type::accelerator:
        return "acc";
    default:
        return "unknown";
    }
}

inline std::string get_device_backend_and_type(const sycl::device &device) {
    std::stringstream device_type;
    sycl::backend backend = device.get_backend();
    device_type <<  backend << ":" << get_device_type_name(device);
    return device_type.str();
}

#endif // GGML_SYCL_DPCT_HELPER_HPP
