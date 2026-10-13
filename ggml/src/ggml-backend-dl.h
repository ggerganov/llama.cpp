#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#else
#    include <dlfcn.h>
#    include <unistd.h>
#endif
#include <memory>
#include <type_traits>
#ifndef GGML_USE_KNC
#include <filesystem>

namespace fs = std::filesystem;
#endif

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

struct dl_handle_deleter {
    void operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

#else

using dl_handle = void;

struct dl_handle_deleter {
    void operator()(void * handle) {
        dlclose(handle);
    }
};

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

#ifndef GGML_USE_KNC
dl_handle * dl_load_library(const fs::path & path);
#endif
void * dl_get_sym(dl_handle * handle, const char * name);
const char * dl_error();

