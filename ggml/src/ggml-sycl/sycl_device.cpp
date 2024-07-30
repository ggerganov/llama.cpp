#include "sycl_device.hpp"
#include "sycl_hw.hpp"


void ggml_sycl_device_info::init(
    ggml_sycl_backend_device_filter device_filter) {
    switch (device_filter) {
    case SYCL_DEVICES_TOP_LEVEL_ZERO:
        detect_sycl_gpu_list_with_max_cu();
        create_context_for_devices();
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
        std::cerr << "ggml_sycl_device_info: Invalid device_filter " << device_filter
                  << std::endl;
    }
    init_allow_devices();
    device_count = ids.size();
}

/*
Bind all devices in same host with same context, for better performance in
device-to-device copy in the future.
*/
void ggml_sycl_device_info::create_context_for_devices() {
    assert(devices.size() > 0);
    sycl::context ctx = sycl::context(devices);
    first_queue = dpct::get_current_device().create_queue(ctx, devices[0]);
    co_ctx = first_queue->get_context();
}

sycl::queue *ggml_sycl_device_info::_create_queue_ptr(sycl::device device) {
    auto q = dpct::get_current_device().create_queue(co_ctx, device);
    return q;
}

sycl::queue *ggml_sycl_device_info::create_queue_for_device(sycl::device &device) {
    dpct::select_device(dpct::dev_mgr::instance().get_device_id(device));
    auto qptr = _create_queue_ptr(device);
    return qptr;
}

sycl::queue *ggml_sycl_device_info::create_queue_for_device_id(int id) {
    sycl::device device = dpct::dev_mgr::instance().get_device(id);
    return create_queue_for_device(device);
}

int ggml_sycl_device_info::get_device_index(int id) {
    for (int i = 0; i < ids.size(); i++) {
        if (ids[i] == id)
            return i;
    }
    return -1;
}

void ggml_sycl_device_info::init_allow_devices() {
    device_list = "";
    for (auto & id: ids) {
        device_list += std::to_string(id);
        device_list += ",";
    }
    if (device_list.length() > 1) {
        device_list.pop_back();
    }
}

bool ggml_sycl_device_info::is_allowed_device(int id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

void ggml_sycl_device_info::detect_all_sycl_device_list() try {
    int all_device_count = dpct::dev_mgr::instance().device_count();

    for (int id = 0; id < all_device_count; id++) {
        add_device_info(id);
    }
    return;
} catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

std::vector<int> ggml_sycl_device_info::get_sycl_visible_devices() {
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

void ggml_sycl_device_info::detect_sycl_visible_device_list() try {
    std::vector<int> sycl_devices = get_sycl_visible_devices();
    int all_device_count = dpct::dev_mgr::instance().device_count();

    for (auto & id: sycl_devices) {
        if (id >= all_device_count) {
            std::cerr << __func__ << ": invalid device_id:" << id
                      << " from GGML_SYCL_VISIBLE_DEVICES="
                      << getenv("GGML_SYCL_VISIBLE_DEVICES")
                      << ", available IDs: ";
            if (all_device_count > 1) {
                std::cerr << "[0, " << all_device_count - 1 << "]";
            } else if (all_device_count == 1) {
                std::cerr << "[0]";
            } else {
                std::cerr << "[]";
            }
            std::cerr << std::endl;
        }
        add_device_info(id);
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
void ggml_sycl_device_info::detect_sycl_gpu_list_with_max_cu() try {
    int all_device_count = dpct::dev_mgr::instance().device_count();
    int local_max_compute_units = 0;
    for (int id = 0; id < all_device_count; id++) {
        sycl::device device = dpct::dev_mgr::instance().get_device(id);
        if (!device.is_gpu())
            continue;
        dpct::device_info prop;
        dpct::get_device_info(prop, device);
        if (local_max_compute_units < prop.get_max_compute_units())
            local_max_compute_units = prop.get_max_compute_units();
    }

    for (int id = 0; id < all_device_count; id++) {
        sycl::device device = dpct::dev_mgr::instance().get_device(id);
        if (!device.is_gpu())
            continue;
        dpct::device_info prop;
        dpct::get_device_info(prop, device);
        if (local_max_compute_units == prop.get_max_compute_units() &&
            is_ext_oneapi_device(device)) {
            add_device_info(id);
        }
    }
    return;
} catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

int ggml_sycl_device_info::get_device_count() { return device_count; }

bool ggml_sycl_device_info::is_ext_oneapi_device(const sycl::device &dev) {
    sycl::backend dev_backend = dev.get_backend();
    if (dev_backend == sycl::backend::ext_oneapi_level_zero ||
        dev_backend == sycl::backend::ext_oneapi_cuda ||
        dev_backend == sycl::backend::ext_oneapi_hip)
        return true;
    return false;
}

void ggml_sycl_device_info::add_device_info(int id) {
    sycl::device device = dpct::dev_mgr::instance().get_device(id);
    dpct::device_info prop;
    dpct::get_device_info(prop, device);

    ids.push_back(id);
    devices.push_back(device);

    device_infos[id].id = id;
    device_infos[id].device = device;
    device_infos[id].max_work_group_sizes = prop.get_max_work_group_size();
    device_infos[id].max_compute_units = prop.get_max_compute_units();
    device_infos[id].hw_family = get_device_family(&device);
    for (int i=0; i<GGML_SYCL_MAX_STREAMS;i++) {
        device_infos[id].qptrs[i] = create_queue_for_device_id(id);
    }
}

void ggml_sycl_device_info::print_gpu_device_list() {
   char *hint = NULL;
    if (oneapi_device_selector_existed && sycl_visible_devices_existed) {
        hint = "detect %d SYCL devices:[%s] by ONEAPI_DEVICE_SELECTOR=%s and "
               "GGML_SYCL_VISIBLE_DEVICES=%s\n";
        fprintf(stderr, hint, get_device_count(), devices_list(),
                getenv("ONEAPI_DEVICE_SELECTOR"),
                getenv("GGML_SYCL_VISIBLE_DEVICES"));
    } else if (oneapi_device_selector_existed) {
        hint = "detect %d SYCL devices:[%s] by ONEAPI_DEVICE_SELECTOR=%s\n";
        fprintf(stderr, hint, get_device_count(), devices_list(),
                getenv("ONEAPI_DEVICE_SELECTOR"));
    } else if (sycl_visible_devices_existed) {
        hint = "detect %d SYCL devices:[%s] by GGML_SYCL_VISIBLE_DEVICES=%s\n";
        fprintf(stderr, hint, get_device_count(), devices_list(),
                getenv("GGML_SYCL_VISIBLE_DEVICES"));
    } else {
        hint = "detect %d SYCL level-zero GPUs:[%s] with top Max compute "
               "units:%d, to use any SYCL devices, set/export "
               "GGML_SYCL_VISIBLE_DEVICES or ONEAPI_DEVICE_SELECTOR\n";
        fprintf(stderr, hint, get_device_count(), devices_list(),
                device_infos[0].max_compute_units);
    }
}

int ggml_sycl_device_info::work_group_size(int id) {
    GGML_ASSERT(is_allowed_device(id));
    return device_infos[id].max_work_group_sizes;
}

ggml_sycl_device_info::ggml_sycl_device_info() {
    oneapi_device_selector_existed = env_existed("ONEAPI_DEVICE_SELECTOR");
    sycl_visible_devices_existed = env_existed("GGML_SYCL_VISIBLE_DEVICES");

    if (sycl_visible_devices_existed) {
        init(SYCL_VISIBLE_DEVICES);
    } else if (oneapi_device_selector_existed) {
        init(SYCL_ALL_DEVICES);
    } else {
        init(SYCL_DEVICES_TOP_LEVEL_ZERO);
    }

    int64_t total_vram = 0;

    for (int i = 0; i < device_count; ++i) {
        int id = get_device_id(i);
        device_infos[id].vmm = 0;
        dpct::device_info prop;
        dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(id));

        // continue data, so use device index
        default_tensor_split[i] = total_vram;
        total_vram += prop.get_global_mem_size();

        device_infos[id].cc =
            100 * prop.get_major_version() + 10 * prop.get_minor_version();
    }

    // continue data, so use device index
    for (int i = 0; i < device_count; ++i) {
        default_tensor_split[i] /= total_vram;
    }

    print_gpu_device_list();
}

const char *ggml_sycl_device_info::devices_list() {
    return device_list.c_str();
}

int ggml_sycl_device_info::get_device_id(int device_index) {
    if (device_index < device_count) {
        return ids.at(device_index);
    } else {
        std::cerr << __func__ << ":SYCL device:" << device_index
                  << " is out of range:[" << devices_list() << "]" << std::endl;
        std::exit(1);
    }
}

int ggml_sycl_device_info::hw_family(int id) {
    return device_infos[id].hw_family;
}

static inline bool env_existed(const char *env_name) {
     char *user_device_string = getenv(env_name);
     return user_device_string!=NULL;
}