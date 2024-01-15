/*
 * #include "common.h"

#include "console.h"
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
*/

#include "ggml-sycl.h"

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif



void print_sycl_devices(){
    int device_count = dpct::dev_mgr::instance().device_count();
    fprintf(stderr, "found %d SYCL devices:\n", device_count);
    for (int id = 0; id < device_count; ++id) {
        dpct::device_info prop;
        dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(id));
        sycl::device cur_device = dpct::dev_mgr::instance().get_device(id);
        fprintf(stderr, "  Device %d: %s,\tcompute capability %d.%d,\n\tmax compute_units %d,\tmax work group size %d,\tmax sub group size %d,\tglobal mem size %lu\n", id,
                prop.get_name(), prop.get_major_version(),
                prop.get_minor_version(),
                prop.get_max_compute_units(),
                prop.get_max_work_group_size(),
                prop.get_max_sub_group_size(),
                prop.get_global_mem_size()
                );
    }
}

int main(int argc, char ** argv) {
    print_sycl_devices();
    return 0;
}
