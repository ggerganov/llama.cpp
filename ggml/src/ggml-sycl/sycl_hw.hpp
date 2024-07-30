#ifndef SYCL_HW_HPP
#define SYCL_HW_HPP

#include <algorithm>
#include <stdio.h>
#include <vector>


#include <sycl/sycl.hpp>

enum SYCL_HW_FAMILY {
  SYCL_HW_FAMILY_UNKNOWN = -1,
  SYCL_HW_FAMILY_INTEL_IGPU = 0,
  SYCL_HW_FAMILY_INTEL_ARC = 1
};

bool is_in_vector(std::vector<int> &vec, int item);

SYCL_HW_FAMILY get_device_family(sycl::device *device_ptr);

#endif // SYCL_HW_HPP
