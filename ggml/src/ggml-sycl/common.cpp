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

  void* ptr = nullptr;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (void*)sycl::malloc_host(size, dpct::get_in_order_queue()));

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

static inline bool env_existed(const char *env_name) {
     char *user_device_string = getenv(env_name);
     return user_device_string!=NULL;
}

static std::vector<int> get_sycl_visible_devices() {
    static std::vector<int> device_ids;
    char *devices_env = getenv("GGML_SYCL_VISIBLE_DEVICES");
    if (devices_env != nullptr) {
        std::string devices(devices_env);
        std::replace(devices.begin(), devices.end(), ',', ' ');

        std::stringstream ss(devices);
        int tmp;
        while (ss >> tmp) {
            device_ids.push_back(tmp);
        }
    }
    return device_ids;
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

    static ggml_sycl_device_info info = {};
    info.refresh_device();

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

//--sycl_device_mgr--

sycl_device_mgr::sycl_device_mgr(
    ggml_sycl_backend_device_filter device_filter) {
    switch (device_filter) {
    case SYCL_DEVICES_TOP_LEVEL_ZERO:
        detect_sycl_gpu_list_with_max_cu();
        create_context_for_group_gpus();
        break;
    case SYCL_ALL_DEVICES:
        detect_all_sycl_device_list();
        create_context_for_devices();
        break;
    case SYCL_VISIBLE_DEVICES:
        detect_sycl_visible_device_list();
        create_context_for_devices();
        break;
    default:
        std::cerr << "sycl_device_mgr: Invalid device_filter " << device_filter
                  << std::endl;
    }
    init_allow_devices();
}

/*
Bind all gpus in same host with same context, for better performance in
device-to-device copy in the future.
*/
void sycl_device_mgr::create_context_for_group_gpus() {
    sycl::context ctx = sycl::context(devices);
    assert(device_ids.size() > 0);
    first_queue = _create_queue_ptr(devices[0]);
    sycl::context ctx0 = first_queue->get_context();
    for (int i = 0; i < device_ids.size(); i++) {
        ctxs.push_back(ctx0);
    }
}

sycl::queue *sycl_device_mgr::_create_queue_ptr(sycl::device device) {
    auto q = dpct::get_current_device().create_queue(device);
    return q;
    // _queues.push_back(q);
    // return & _queues.back();
}

sycl::queue *sycl_device_mgr::create_queue_for_device(sycl::device &device) {
    dpct::select_device(dpct::dev_mgr::instance().get_device_id(device));
    auto qptr = _create_queue_ptr(device);
    return qptr;
}

sycl::queue *sycl_device_mgr::create_queue_for_device_id(int device_id) {
    int i = get_device_index(device_id);
    sycl::device device = dpct::dev_mgr::instance().get_device(device_id);
    return create_queue_for_device(device);
}

int sycl_device_mgr::get_device_index(int device_id) {
    for (int i = 0; i < device_ids.size(); i++) {
        if (device_ids[i] == device_id)
            return i;
    }
    return -1;
}

void sycl_device_mgr::create_context_for_devices() {
    for (int i = 0; i < device_ids.size(); i++) {
        sycl::context ctx = sycl::context(devices[i]);
        ctxs.push_back(ctx);
    }
}

void sycl_device_mgr::init_allow_devices() {
    device_list = "";
    for (size_t i = 0; i < device_ids.size(); ++i) {
        device_list += std::to_string(device_ids[i]);
        device_list += ",";
    }
    if (device_list.length() > 1) {
        device_list.pop_back();
    }
}

bool sycl_device_mgr::is_allowed_device(int device_id) {
    return std::find(device_ids.begin(), device_ids.end(), device_id) !=
           device_ids.end();
}

void sycl_device_mgr::detect_all_sycl_device_list() try {
    int device_count = dpct::dev_mgr::instance().device_count();

    for (int id = 0; id < device_count; id++) {
        sycl::device device = dpct::dev_mgr::instance().get_device(id);
        device_ids.push_back(id);
        devices.push_back(device);
        dpct::device_info prop;
        dpct::get_device_info(prop, device);
        work_group_sizes.push_back(prop.get_max_work_group_size());
        max_compute_units.push_back(prop.get_max_compute_units());
    }
    return;
} catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void sycl_device_mgr::detect_sycl_visible_device_list() try {
    std::vector<int> sycl_devices = get_sycl_visible_devices();
    int device_count = dpct::dev_mgr::instance().device_count();

    for (int i = 0; i < sycl_devices.size(); i++) {
        int id = sycl_devices[i];
        if (id >= device_count) {
            std::cerr << __func__ << ": invalid device_id:" << id
                      << " from GGML_SYCL_VISIBLE_DEVICES="
                      << getenv("GGML_SYCL_VISIBLE_DEVICES")
                      << ", available IDs: ";
            if (device_count > 1) {
                std::cerr << "[0, " << device_count - 1 << "]";
            } else if (device_count == 1) {
                std::cerr << "[0]";
            } else {
                std::cerr << "[]";
            }
            std::cerr << std::endl;
        }
        sycl::device device = dpct::dev_mgr::instance().get_device(id);
        device_ids.push_back(id);
        devices.push_back(device);
        dpct::device_info prop;
        dpct::get_device_info(prop, device);
        work_group_sizes.push_back(prop.get_max_work_group_size());
        max_compute_units.push_back(prop.get_max_compute_units());
    }
    return;
} catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

/*
Use all GPUs with same top max compute units
*/
void sycl_device_mgr::detect_sycl_gpu_list_with_max_cu() try {
    int device_count = dpct::dev_mgr::instance().device_count();
    int local_max_compute_units = 0;
    for (int id = 0; id < device_count; id++) {
        sycl::device device = dpct::dev_mgr::instance().get_device(id);
        if (!device.is_gpu())
            continue;
        dpct::device_info prop;
        dpct::get_device_info(prop, device);
        if (local_max_compute_units < prop.get_max_compute_units())
            local_max_compute_units = prop.get_max_compute_units();
    }

    for (int id = 0; id < device_count; id++) {
        sycl::device device = dpct::dev_mgr::instance().get_device(id);
        if (!device.is_gpu())
            continue;
        dpct::device_info prop;
        dpct::get_device_info(prop, device);
        if (local_max_compute_units == prop.get_max_compute_units() &&
            is_ext_oneapi_device(device)) {
            device_ids.push_back(id);
            devices.push_back(device);
            work_group_sizes.push_back(prop.get_max_work_group_size());
            max_compute_units.push_back(prop.get_max_compute_units());
        }
    }
    return;
} catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

int sycl_device_mgr::get_device_count() { return (int)device_ids.size(); }

bool sycl_device_mgr::is_ext_oneapi_device(const sycl::device &dev) {
    sycl::backend dev_backend = dev.get_backend();
    if (dev_backend == sycl::backend::ext_oneapi_level_zero ||
        dev_backend == sycl::backend::ext_oneapi_cuda ||
        dev_backend == sycl::backend::ext_oneapi_hip)
        return true;
    return false;
}
//--sycl_device_mgr--

//--ggml_sycl_device_info--
void ggml_sycl_device_info::print_gpu_device_list() {
    GGML_ASSERT(device_mgr);

    char *hint = NULL;
    if (oneapi_device_selector_existed && sycl_visible_devices_existed) {
        hint = "detect %d SYCL devices:[%s] by ONEAPI_DEVICE_SELECTOR=%s and "
               "GGML_SYCL_VISIBLE_DEVICES=%s\n";
        fprintf(stderr, hint, device_mgr->get_device_count(), devices_list(),
                getenv("ONEAPI_DEVICE_SELECTOR"),
                getenv("GGML_SYCL_VISIBLE_DEVICES"));
    } else if (oneapi_device_selector_existed) {
        hint = "detect %d SYCL devices:[%s] by ONEAPI_DEVICE_SELECTOR=%s\n";
        fprintf(stderr, hint, device_mgr->get_device_count(), devices_list(),
                getenv("ONEAPI_DEVICE_SELECTOR"));
    } else if (sycl_visible_devices_existed) {
        hint = "detect %d SYCL devices:[%s] by GGML_SYCL_VISIBLE_DEVICES=%s\n";
        fprintf(stderr, hint, device_mgr->get_device_count(), devices_list(),
                getenv("GGML_SYCL_VISIBLE_DEVICES"));
    } else {
        hint = "detect %d SYCL level-zero GPUs:[%s] with top Max compute "
               "units:%d, to use any SYCL devices, set/export "
               "GGML_SYCL_VISIBLE_DEVICES or ONEAPI_DEVICE_SELECTOR\n";
        fprintf(stderr, hint, device_mgr->get_device_count(), devices_list(),
                device_mgr->max_compute_units[0]);
    }
}

int ggml_sycl_device_info::work_group_size(int device_id) {
    GGML_ASSERT(device_mgr);
    return device_mgr->work_group_sizes[device_id];
}

void ggml_sycl_device_info::refresh_device() {
    oneapi_device_selector_existed = env_existed("ONEAPI_DEVICE_SELECTOR");
    sycl_visible_devices_existed = env_existed("GGML_SYCL_VISIBLE_DEVICES");
    if (!device_mgr)
        delete device_mgr;

    if (sycl_visible_devices_existed) {
        device_mgr = new sycl_device_mgr(SYCL_VISIBLE_DEVICES);
    } else if (oneapi_device_selector_existed) {
        device_mgr = new sycl_device_mgr(SYCL_ALL_DEVICES);
    } else {
        device_mgr = new sycl_device_mgr(SYCL_DEVICES_TOP_LEVEL_ZERO);
    }

    device_count = device_mgr->get_device_count();

    int64_t total_vram = 0;

    for (int i = 0; i < device_count; ++i) {
        int id = get_device_id(i);
        devices[id].vmm = 0;
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(id))));

        default_tensor_split[i] =
            total_vram; // continue data, so use device index
        total_vram += prop.get_global_mem_size();

        devices[id].cc =
            100 * prop.get_major_version() + 10 * prop.get_minor_version();
    }

    for (int i = 0; i < device_count; ++i) {
        default_tensor_split[i] /=
            total_vram; // continue data, so use device index
    }

    print_gpu_device_list();
}

bool ggml_sycl_device_info::is_allowed_device(int device_id) {
    return device_mgr->is_allowed_device(device_id);
}

const char *ggml_sycl_device_info::devices_list() {
    return device_mgr->device_list.c_str();
}

int ggml_sycl_device_info::get_device_id(int device_index) {
    if (device_index < device_mgr->device_ids.size()) {
        return device_mgr->device_ids.at(device_index);
    } else {
        std::cerr << __func__ << ":SYCL device:" << device_index
                  << " is out of range:[" << devices_list() << "]" << std::endl;
        std::exit(1);
    }
}

//--ggml_sycl_device_info--
