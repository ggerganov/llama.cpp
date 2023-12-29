//==---- kernel.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_KERNEL_HPP__
#define __DPCT_KERNEL_HPP__

#include <sycl/sycl.hpp>
#ifdef _WIN32
#include <unordered_set>
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(__has_include) && __has_include(<filesystem>)
#include <filesystem>
#elif defined(__has_include) && __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
#else
#error "SYCLomatic runtime requires C++ filesystem support"
#endif

#include <fstream>
#include <image.hpp>
#include <random>

namespace dpct {

typedef void (*kernel_functor)(sycl::queue &, const sycl::nd_range<3> &,
                               unsigned int, void **, void **);

struct kernel_function_info {
  int max_work_group_size = 0;
};

static inline void get_kernel_function_info(kernel_function_info *kernel_info,
                                            const void *function) {
  kernel_info->max_work_group_size =
      dpct::dev_mgr::instance()
          .current_device()
          .get_info<sycl::info::device::max_work_group_size>();
}
static inline kernel_function_info
get_kernel_function_info(const void *function) {
  kernel_function_info kernel_info;
  kernel_info.max_work_group_size =
      dpct::dev_mgr::instance()
          .current_device()
          .get_info<sycl::info::device::max_work_group_size>();
  return kernel_info;
}


namespace detail {

#if defined(__has_include) && __has_include(<filesystem>)
namespace fs = std::filesystem;
#else
namespace fs = std::experimental::filesystem;
#endif

/// Write data to temporary file and return absolute path to temporary file.
/// Temporary file is created in a temporary directory both of which have random
/// names with only the user having access permissions.  Only one temporary file
/// will be created in the temporary directory.
static inline fs::path write_data_to_file(char const *const data, size_t size) {
  std::error_code ec;

  if (sizeof(size_t) >= sizeof(std::streamsize) &&
      size > (std::numeric_limits<std::streamsize>::max)())
    throw std::runtime_error("data file too large");

  // random number generator
  std::random_device dev;
  std::mt19937 prng(dev());
  std::uniform_int_distribution<uint64_t> rand(0);

  // find temporary directory
  auto tmp_dir = fs::temp_directory_path(ec);
  if (ec)
    throw std::runtime_error("could not find temporary directory");

  // create private directory
  std::stringstream directory;
  fs::path directory_path;
  constexpr int max_attempts = 5;
  int i;

  for (i = 0; i < max_attempts; i++) {
    directory << std::hex << rand(prng);
    directory_path = tmp_dir / directory.str();
    if (fs::create_directory(directory_path)) {
      break;
    }
  }
  if (i == max_attempts)
    throw std::runtime_error("could not create directory");

  // only allow owner permissions to private directory
  fs::permissions(directory_path, fs::perms::owner_all, ec);
  if (ec)
    throw std::runtime_error("could not set directory permissions");

  // random filename in private directory
  std::stringstream filename;
  filename << std::hex << rand(prng);
#ifdef _WIN32
  auto filepath = directory_path / (filename.str() + ".dll");
#else
  auto filepath = directory_path / filename.str();
#endif

  // write data to temporary file
  auto outfile = std::ofstream(filepath, std::ios::out | std::ios::binary);
  if (outfile) {
    // only allow program to write file
    fs::permissions(filepath, fs::perms::owner_write, ec);
    if (ec)
      throw std::runtime_error("could not set permissions");

    outfile.write(data, size);
    if (!outfile.good())
      throw std::runtime_error("could not write data");
    outfile.close();

    // only allow program to read/execute file
    fs::permissions(filepath, fs::perms::owner_read | fs::perms::owner_exec,
                    ec);
    if (ec)
      throw std::runtime_error("could not set permissions");
  } else
    throw std::runtime_error("could not write data");

  // check temporary file contents
  auto infile = std::ifstream(filepath, std::ios::in | std::ios::binary);
  if (infile) {
    bool mismatch = false;
    size_t cnt = 0;

    while (1) {
      char c;
      infile.get(c);
      if (infile.eof())
        break;
      if (c != data[cnt++])
        mismatch = true;
    }
    if (cnt != size || mismatch)
      throw std::runtime_error("file contents not written correctly");
  } else
    throw std::runtime_error("could not validate file");

  if (!filepath.is_absolute())
    throw std::runtime_error("temporary filepath is not absolute");

  return filepath;
}

static inline uint16_t extract16(unsigned char const *const ptr) {
  uint16_t ret = 0;

  ret |= static_cast<uint16_t>(ptr[0]) << 0;
  ret |= static_cast<uint16_t>(ptr[1]) << 8;

  return (ret);
}

static inline uint32_t extract32(unsigned char const *const ptr) {
  uint32_t ret = 0;

  ret |= static_cast<uint32_t>(ptr[0]) << 0;
  ret |= static_cast<uint32_t>(ptr[1]) << 8;
  ret |= static_cast<uint32_t>(ptr[2]) << 16;
  ret |= static_cast<uint32_t>(ptr[3]) << 24;

  return (ret);
}

static inline uint64_t extract64(unsigned char const *const ptr) {
  uint64_t ret = 0;

  ret |= static_cast<uint64_t>(ptr[0]) << 0;
  ret |= static_cast<uint64_t>(ptr[1]) << 8;
  ret |= static_cast<uint64_t>(ptr[2]) << 16;
  ret |= static_cast<uint64_t>(ptr[3]) << 24;
  ret |= static_cast<uint64_t>(ptr[4]) << 32;
  ret |= static_cast<uint64_t>(ptr[5]) << 40;
  ret |= static_cast<uint64_t>(ptr[6]) << 48;
  ret |= static_cast<uint64_t>(ptr[7]) << 56;

  return (ret);
}

static inline uint64_t get_lib_size(char const *const blob) {
#ifdef _WIN32
  ///////////////////////////////////////////////////////////////////////
  // Analyze DOS stub
  unsigned char const *const ublob =
      reinterpret_cast<unsigned char const *const>(blob);
  if (ublob[0] != 0x4d || ublob[1] != 0x5a) {
    throw std::runtime_error("Blob is not a Windows DLL.");
  }
  uint32_t pe_header_offset = extract32(ublob + 0x3c);

  ///////////////////////////////////////////////////////////////////////
  // Ananlyze PE-header
  unsigned char const *const pe_header = ublob + pe_header_offset;

  // signature
  uint32_t pe_signature = extract32(pe_header + 0);
  if (pe_signature != 0x00004550) {
    throw std::runtime_error("PE-header signature is not 0x00004550");
  }

  // machine
  uint16_t machine = extract16(pe_header + 4);
  if (machine != 0x8664) {
    throw std::runtime_error("Only DLLs for x64 supported");
  }

  // number of sections
  uint16_t number_of_sections = extract16(pe_header + 6);

  // sizeof optional header
  uint16_t sizeof_optional_header = extract16(pe_header + 20);

  // magic
  uint16_t magic = extract16(pe_header + 24);
  if (magic != 0x10b && magic != 0x20b) {
    throw std::runtime_error("MAGIC is not 0x010b or 0x020b");
  }

  ///////////////////////////////////////////////////////////////////////
  // Analyze tail of optional header
  constexpr int coff_header_size = 24;

  unsigned char const *const tail_of_optional_header =
      pe_header + coff_header_size + sizeof_optional_header;
  if (extract64(tail_of_optional_header - 8) != 0) {
    throw std::runtime_error("Optional header not zero-padded");
  }

  ///////////////////////////////////////////////////////////////////////
  // Analyze last section header
  constexpr int section_header_size = 40;
  unsigned char const *const last_section_header =
      tail_of_optional_header + section_header_size * (number_of_sections - 1);

  uint32_t sizeof_raw_data = extract32(last_section_header + 16);
  uint32_t pointer_to_raw_data = extract32(last_section_header + 20);

  return sizeof_raw_data + pointer_to_raw_data;
#else
  if (blob[0] != 0x7F || blob[1] != 'E' || blob[2] != 'L' || blob[3] != 'F')
    throw std::runtime_error("Blob is not in ELF format");

  if (blob[4] != 0x02)
    throw std::runtime_error("Only 64-bit headers are supported");

  if (blob[5] != 0x01)
    throw std::runtime_error("Only little-endian headers are supported");

  unsigned char const *const ublob =
      reinterpret_cast<unsigned char const *const>(blob);
  uint64_t e_shoff = extract64(ublob + 0x28);
  uint16_t e_shentsize = extract16(ublob + 0x3A);
  uint16_t e_shnum = extract16(ublob + 0x3C);

  return e_shoff + (e_shentsize * e_shnum);
#endif
}

#ifdef _WIN32
class path_lib_record {
public:
  void operator=(const path_lib_record &) = delete;
  ~path_lib_record() {
    for (auto entry : lib_to_path) {
      FreeLibrary(static_cast<HMODULE>(entry.first));
      fs::permissions(entry.second, fs::perms::owner_all);
      fs::remove_all(entry.second.remove_filename());
    }
  }
  static void record_lib_path(fs::path path, void *library) {
    lib_to_path[library] = path;
  }
  static void remove_lib(void *library) {
    auto path = lib_to_path[library];
    std::error_code ec;

    FreeLibrary(static_cast<HMODULE>(library));
    fs::permissions(path, fs::perms::owner_all);
    if (fs::remove_all(path.remove_filename(), ec) != 2 || ec)
      // one directory and one temporary file should have been deleted
      throw std::runtime_error("Directory delete failed");

    lib_to_path.erase(library);
  }

private:
  static inline std::unordered_map<void *, fs::path> lib_to_path;
};
#endif

} // namespace detail

class kernel_library {
public:
  kernel_library() : ptr{nullptr} {}
  kernel_library(void *ptr) : ptr{ptr} {}

  operator void *() const { return ptr; }

private:
  void *ptr;
#ifdef _WIN32
  static inline detail::path_lib_record single_instance_to_trigger_destructor;
#endif
};

namespace detail {

static inline kernel_library load_dl_from_data(char const *const data,
                                               size_t size) {
  fs::path filename = write_data_to_file(data, size);
#ifdef _WIN32
  void *so = LoadLibraryW(filename.wstring().c_str());
#else
  void *so = dlopen(filename.c_str(), RTLD_LAZY);
#endif
  if (so == nullptr)
    throw std::runtime_error("Failed to load kernel library");

#ifdef _WIN32
  detail::path_lib_record::record_lib_path(filename, so);
#else
  std::error_code ec;

  // Windows DLL cannot be deleted while in use
  if (fs::remove_all(filename.remove_filename(), ec) != 2 || ec)
    // one directory and one temporary file should have been deleted
    throw std::runtime_error("Directory delete failed");
#endif

  return so;
}

} // namespace detail

/// Load kernel library and return a handle to use the library.
/// \param [in] name The name of the library.
static inline kernel_library load_kernel_library(const std::string &name) {
  std::ifstream ifs;
  ifs.open(name, std::ios::in | std::ios::binary);

  std::stringstream buffer;
  buffer << ifs.rdbuf();

  const std::string buffer_string = buffer.str();
  return detail::load_dl_from_data(buffer_string.c_str(), buffer_string.size());
}

/// Load kernel library whose image is alreay in memory and return a handle to
/// use the library.
/// \param [in] image A pointer to the image in memory.
static inline kernel_library load_kernel_library_mem(char const *const image) {
  const size_t size = detail::get_lib_size(image);

  return detail::load_dl_from_data(image, size);
}

/// Unload kernel library.
/// \param [in,out] library Handle to the library to be closed.
static inline void unload_kernel_library(const kernel_library &library) {
#ifdef _WIN32
  detail::path_lib_record::remove_lib(library);
#else
  dlclose(library);
#endif
}

class kernel_function {
public:
  kernel_function() : ptr{nullptr} {}
  kernel_function(dpct::kernel_functor ptr) : ptr{ptr} {}

  operator void *() const { return ((void *)ptr); }

  void operator()(sycl::queue &q, const sycl::nd_range<3> &range,
                  unsigned int a, void **args, void **extra) {
    ptr(q, range, a, args, extra);
  }

private:
  dpct::kernel_functor ptr;
};

/// Find kernel function in a kernel library and return its address.
/// \param [in] library Handle to the kernel library.
/// \param [in] name Name of the kernel function.
static inline dpct::kernel_function
get_kernel_function(kernel_library &library, const std::string &name) {
#ifdef _WIN32
  dpct::kernel_functor fn = reinterpret_cast<dpct::kernel_functor>(
      GetProcAddress(static_cast<HMODULE>(static_cast<void *>(library)),
                     (name + std::string("_wrapper")).c_str()));
#else
  dpct::kernel_functor fn = reinterpret_cast<dpct::kernel_functor>(
      dlsym(library, (name + std::string("_wrapper")).c_str()));
#endif
  if (fn == nullptr)
    throw std::runtime_error("Failed to get function");
  return fn;
}

/// Invoke a kernel function.
/// \param [in] function kernel function.
/// \param [in] queue SYCL queue used to execute kernel
/// \param [in] groupRange SYCL group range
/// \param [in] localRange SYCL local range
/// \param [in] localMemSize The size of local memory required by the kernel
///             function.
/// \param [in] kernelParams Array of pointers to kernel arguments.
/// \param [in] extra Extra arguments.
static inline void invoke_kernel_function(dpct::kernel_function &function,
                                          sycl::queue &queue,
                                          sycl::range<3> groupRange,
                                          sycl::range<3> localRange,
                                          unsigned int localMemSize,
                                          void **kernelParams, void **extra) {
  function(queue, sycl::nd_range<3>(groupRange * localRange, localRange),
           localMemSize, kernelParams, extra);
}

/// Find image wrapper in a kernel library and return its address.
/// \param [in] library Handle to the kernel library.
/// \param [in] name Name of the target image wrapper.
static inline dpct::image_wrapper_base_p
get_image_wrapper(dpct::kernel_library &library, const std::string &name) {
#ifdef _WIN32
  dpct::image_wrapper_base_p fn =
      reinterpret_cast<dpct::image_wrapper_base_p>(GetProcAddress(
          static_cast<HMODULE>(static_cast<void *>(library)), name.c_str()));
#else
  dpct::image_wrapper_base_p fn = reinterpret_cast<dpct::image_wrapper_base_p>(
      dlsym(library, name.c_str()));
#endif
  if (fn == nullptr)
    throw std::runtime_error("Failed to get image");
  return fn;
}

} // namespace dpct
#endif // __DPCT_KERNEL_HPP__
