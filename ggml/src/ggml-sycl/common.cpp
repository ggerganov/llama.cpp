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

#include "common.hpp"

int get_current_device_id() {
  return dpct::dev_mgr::instance().current_device_id();
}

void* ggml_sycl_host_malloc(size_t size) try {
  if (getenv("GGML_SYCL_NO_PINNED") != nullptr) {
    return nullptr;
  }
//   ggml_sycl_info().device_mgr->first_queue
  void* ptr = nullptr;
  // allow to use dpct::get_in_order_queue() for host malloc
  auto q = dpct::get_in_order_queue();
//   sycl::queue q = *ggml_sycl_info().device_mgr->qptrs[0][0];

  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (void*)sycl::malloc_host(size, q));

//  printf("zjy ggml_sycl_host_malloc ptr=%p queue=%p size=%lu \n", ptr,q, size);
  if (err != 0) {
    // clear the error
    fprintf(
        stderr,
        "WARNING: failed to allocate %.2f MB of pinned memory: %s\n",
        size / 1024.0 / 1024.0,
        "syclGetErrorString is not supported");
    return nullptr;
  }

  return ptr;
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_sycl_host_free(void* ptr) try {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, dpct::get_in_order_queue())));
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static inline int get_sycl_env(const char *env_name, int default_val) {
    char *user_device_string = getenv(env_name);
    int user_number = default_val;

    unsigned n;
    if (user_device_string != NULL &&
        sscanf(user_device_string, " %u", &n) == 1) {
        user_number = (int)n;
    } else {
        user_number = default_val;
    }
    return user_number;
}

void print_device_detail_part1(int id, sycl::device &device, std::string device_type) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    std::string version;
    version += std::to_string(prop.get_major_version());
    version += ".";
    version += std::to_string(prop.get_minor_version());

    device_type = std::regex_replace(device_type, std::regex("ext_oneapi_"), "");
    std::string name = std::string(prop.get_name());
    name = std::regex_replace(name, std::regex("\\(R\\)"), "");
    name = std::regex_replace(name, std::regex("\\(TM\\)"), "");

    auto global_mem_size = prop.get_global_mem_size()/1000000;

    fprintf(stderr, "|%2d|%19s|%4s|%39s|%14luM|\n", id, device_type.c_str(), version.c_str(),
        name.c_str(), global_mem_size);
}

void print_device_detail_part2(int id, sycl::device &device, std::string device_type) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    fprintf(stderr, "|%2d|%17d|%14d|%12d|%34s|\n", id,
        prop.get_max_compute_units(),
        prop.get_max_work_group_size(), prop.get_max_sub_group_size(),
        device.get_info<sycl::info::device::driver_version>().c_str());
}

void ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    int device_count = dpct::dev_mgr::instance().device_count();
    std::map<std::string, size_t> DeviceNums;
    fprintf(stderr, "found %d SYCL devices:\n", device_count);
    fprintf(stderr, "Part1:\n");
    fprintf(stderr, "|ID|        Device Type| Ver|                                   Name|Global mem size|\n");
    fprintf(stderr, "|--|-------------------|----|---------------------------------------|---------------|\n");
    for (int id = 0; id < device_count; ++id) {
        sycl::device device = dpct::dev_mgr::instance().get_device(id);
        sycl::backend backend = device.get_backend();
        std::string backend_type = get_device_backend_and_type(device);
        int type_id=DeviceNums[backend_type]++;
        std::stringstream device_type;
        device_type << "[" <<  backend_type << ":" << std::to_string(type_id) << "]";
        print_device_detail_part1(id, device, device_type.str());
    }

    std::map<std::string, size_t> DeviceNums2;
    fprintf(stderr, "\nPart2:\n");
    fprintf(stderr, "|ID|Max compute units|Max work group|Max subgroup|                    Driver version|\n");
    fprintf(stderr, "|--|-----------------|--------------|------------|----------------------------------|\n");
    for (int id = 0; id < device_count; ++id) {
        sycl::device device = dpct::dev_mgr::instance().get_device(id);
        sycl::backend backend = device.get_backend();
        std::string backend_type = get_device_backend_and_type(device);
        int type_id=DeviceNums2[backend_type]++;
        std::stringstream device_type;
        device_type << "[" <<  backend_type << ":" << std::to_string(type_id) << "]";
        print_device_detail_part2(id, device, device_type.str());
    }
}

static ggml_sycl_device_info ggml_sycl_init() try {
    static bool initialized = false;

    if (!initialized) {
        fprintf(stderr, "[SYCL] call ggml_init_sycl\n");

        g_ggml_sycl_debug = get_sycl_env("GGML_SYCL_DEBUG", 0);
        fprintf(stderr, "%s: GGML_SYCL_DEBUG: %d\n", __func__,
                g_ggml_sycl_debug);

#if defined(GGML_SYCL_F16)
        fprintf(stderr, "%s: GGML_SYCL_F16: yes\n", __func__);
#else
        fprintf(stderr, "%s: GGML_SYCL_F16: no\n", __func__);
#endif

#if defined(GGML_SYCL_FORCE_MMQ)
        fprintf(stderr, "%s: GGML_SYCL_FORCE_MMQ:   yes\n", __func__);
#else
        fprintf(stderr, "%s: GGML_SYCL_FORCE_MMQ:   no\n", __func__);
#endif

#if defined(SYCL_USE_XMX)
        fprintf(stderr, "%s: SYCL_USE_XMX: yes\n", __func__);
#else
        fprintf(stderr, "%s: SYCL_USE_XMX: no\n", __func__);
#endif

        if (CHECK_TRY_ERROR(g_all_sycl_device_count =
                                dpct::dev_mgr::instance().device_count()) !=
            0) {
            initialized = true;
            return;
        }
        GGML_ASSERT(g_all_sycl_device_count <= GGML_SYCL_MAX_DEVICES);
        ggml_backend_sycl_print_sycl_devices();
        initialized = true;
    }

    static ggml_sycl_device_info info;

    if (info.device_count == 0) {
        fprintf(stderr, "%s: failed to initialize " GGML_SYCL_NAME ": no available device found\n",
                __func__);
        return info;
    }
    GGML_ASSERT(info.device_count <= GGML_SYCL_MAX_DEVICES);

    return info;
} catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

ggml_sycl_device_info &ggml_sycl_info() {
    static ggml_sycl_device_info info = ggml_sycl_init();
    return info;
}

//--ggml_sycl_device_info--
