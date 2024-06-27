#include "nvapi.h"

#ifdef _WIN32
#  include <libloaderapi.h>
#elif __linux__
#  include <dlfcn.h>
#endif

/////

static void * lib;

static bool  load_success;

typedef void * (*nvapi_QueryInterface_t)(int);
typedef int    (*NvAPI_EnumPhysicalGPUs_t)(void *, void *);
typedef int    (*NvAPI_GPU_SetForcePstate_t)(void *, int, int);
typedef int    (*NvAPI_Initialize_t)();
typedef int    (*NvAPI_Unload_t)();

static nvapi_QueryInterface_t     nvapi_QueryInterface;
static NvAPI_EnumPhysicalGPUs_t   NvAPI_EnumPhysicalGPUs;
static NvAPI_GPU_SetForcePstate_t NvAPI_GPU_SetForcePstate;
static NvAPI_Initialize_t         NvAPI_Initialize;
static NvAPI_Unload_t             NvAPI_Unload;

/////

void nvapi_init() {
    // load library
#ifdef _WIN32
    if (!lib) {
        lib = LoadLibrary("nvapi64.dll");
    }

    if (!lib) {
        lib = LoadLibrary("nvapi.dll");
    }
#elif __linux__
    if (!lib) {
        lib = dlopen("libnvidia-api.so.1", RTLD_LAZY);
    }

    if (!lib) {
        lib = dlopen("libnvidia-api.so", RTLD_LAZY);
    }
#endif

    // lookup QueryInterface
    if (lib) {
#ifdef _WIN32
        nvapi_QueryInterface = (nvapi_QueryInterface_t) GetProcAddress(lib, "nvapi_QueryInterface");
#elif __linux__
        nvapi_QueryInterface = (nvapi_QueryInterface_t) dlsym(lib, "nvapi_QueryInterface");
#endif
    }

    // resolve functions
    if (nvapi_QueryInterface) {
        NvAPI_EnumPhysicalGPUs = (NvAPI_EnumPhysicalGPUs_t) nvapi_QueryInterface(0xe5ac921f);
        NvAPI_GPU_SetForcePstate = (NvAPI_GPU_SetForcePstate_t) nvapi_QueryInterface(0x025bfb10);
        NvAPI_Initialize = (NvAPI_Initialize_t) nvapi_QueryInterface(0x0150e828);
        NvAPI_Unload = (NvAPI_Unload_t) nvapi_QueryInterface(0xd22bdd7e);
    }

    // initialize library
    if (NvAPI_Initialize()) {
        load_success = true;
    }
}

void nvapi_free() {
    // deinitialize library
    if (load_success) {
        NvAPI_Unload();
    }

    // free library
    if (lib) {
#ifdef _WIN32
        FreeLibrary(lib);
#elif __linux__
        dlclose(lib);
#endif
    }

    // invalidate pointers
    lib = nullptr;
    load_success = false;
}

void nvapi_set_pstate(int ids[], int ids_size, int pstate) {
    if (!load_success) {
        return;
    }

    // TODO
    (void) ids;
    (void) ids_size;
    (void) pstate;
    printf("nvapi_set_pstate: %d", pstate);
}

void nvapi_set_pstate_high() {
    nvapi_set_pstate({}, 0, 16);
}

void nvapi_set_pstate_low() {
    nvapi_set_pstate({}, 0, 8);
}
