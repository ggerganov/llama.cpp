#include "sycl_hw.hpp"

bool is_in_vector(const std::vector<int> &vec, int item) {
  return std::find(vec.begin(), vec.end(), item) != vec.end();
}

SYCL_HW_FAMILY get_device_family(sycl::device *device_ptr) {
  auto id = device_ptr->get_info<sycl::ext::intel::info::device::device_id>();
  auto id_prefix = id & 0xff00;

  const std::vector<int> Xe_ARC = {0x5600, 0x4f00};
  const std::vector<int> Xe_Iris_IDs = {0x4900, 0xa700};
  const std::vector<int> UHD_IDs = {0x4600};

  if (is_in_vector(Xe_Iris_IDs, id_prefix) or is_in_vector(UHD_IDs, id_prefix)) {
    return SYCL_HW_FAMILY_INTEL_IGPU;
  } else if (is_in_vector(Xe_ARC, id_prefix)) {
    return SYCL_HW_FAMILY_INTEL_ARC;
  } else {
    std::cerr << "No support PCI_ID: " << std::hex << id << std::endl;
    return SYCL_HW_FAMILY_UNKNOWN;
  }
}
