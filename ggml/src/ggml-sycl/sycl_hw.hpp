#ifndef SYCL_HW_HPP
#define SYCL_HW_HPP

#include <algorithm>
#include <stdio.h>
#include <vector>


#include <sycl/sycl.hpp>

// const int Xe_ARC[] = {0x5600, 0x4f};
const std::vector<int> Xe_Iris_IDs = {0x4900, 0xa700};
const std::vector<int> UHD_IDs = {0x4600};

enum SYCL_HW_FAMILY {
  SYCL_HW_FAMILY_UNKNOWN = -1,
  SYCL_HW_FAMILY_INTEL_IGPU = 0
};

bool is_in_vector(std::vector<int> &vec, int item);

SYCL_HW_FAMILY get_device_family(sycl::device *device_ptr);

#endif // SYCL_HW_HPP