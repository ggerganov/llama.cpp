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

static std::set<int> parse_visible_devices() {
    // set to store device IDs
    std::set<int> devices;

    // retrieve the value of the environment variable "CUDA_VISIBLE_DEVICES"
    const char * env_p = std::getenv("CUDA_VISIBLE_DEVICES");
    if (!env_p) {
        return devices;
    }

    // create a string stream from the environment variable value
    std::stringstream ss(env_p);

    // string to hold each device ID from the stream
    std::string item;

    // iterate over the comma-separated device IDs in the environment variable
    while (std::getline(ss, item, ",")) {
        try {
            // convert the current item to an integer and insert it into the set
            devices.insert(std::stoi(item));
        } catch (...) {
            // ignore any exceptions that occur during the conversion
            continue;
        }
    }

    // return the set of device IDs
    return devices;
}

/////

void nvapi_init() {
    // attempt to load the NVAPI library
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

    // obtain the address of the nvapi_QueryInterface function
    if (lib) {
#ifdef _WIN32
        nvapi_QueryInterface = (nvapi_QueryInterface_t) GetProcAddress(lib, "nvapi_QueryInterface");
#elif __linux__
        nvapi_QueryInterface = (nvapi_QueryInterface_t) dlsym(lib, "nvapi_QueryInterface");
#endif
    }

    // retrieve function pointers for various NVAPI functions
    if (nvapi_QueryInterface) {
        NvAPI_EnumPhysicalGPUs = (NvAPI_EnumPhysicalGPUs_t) nvapi_QueryInterface(0xe5ac921f);
        NvAPI_GPU_SetForcePstate = (NvAPI_GPU_SetForcePstate_t) nvapi_QueryInterface(0x025bfb10);
        NvAPI_Initialize = (NvAPI_Initialize_t) nvapi_QueryInterface(0x0150e828);
        NvAPI_Unload = (NvAPI_Unload_t) nvapi_QueryInterface(0xd22bdd7e);
    }

    // initialize the NVAPI library
    if (NvAPI_Initialize()) {
        load_success = true;
    }
}

void nvapi_free() {
    // if the library was successfully initialized, unload it
    if (load_success) {
        NvAPI_Unload();
    }

    // release the library resources
    if (lib) {
#ifdef _WIN32
        FreeLibrary(lib);
#elif __linux__
        dlclose(lib);
#endif
    }

    // reset the pointers and flags
    lib = nullptr;
    load_success = false;
}

void nvapi_set_pstate(int pstate) {
    // check if the library initialization was successful before proceeding
    if (!load_success) {
        return;
    }

    // array to hold GPU handles
    void *gpu_array[64] = {0};

    // integer to hold GPU count
    int gpu_count = 0;

    // attempt to enumerate GPUs
    if (NvAPI_EnumPhysicalGPUs(gpu_array, &gpu_count) != 0) {
        fprintf(stderr, "Failed to enumerate GPUs\n");
        return;
    }

    // try to retrieve the set of visible CUDA devices
    std::set<int> devices = parse_visible_devices();

    // iterate over each GPU
    for (int i = 0; i < gpu_count; i++) {
        // if the set of visible devices is not empty and the current GPU ID is not in this set, skip this iteration
        if (!devices.empty() && !devices.find(i)) {
            continue;
        }

        // attempt to set the performance state for the current GPU
        if (NvAPI_GPU_SetForcePstate(gpu_array[gpu_id], pstate, 2) != 0) {
            fprintf(stderr, "Failed to set performance state for gpu #%d\n", gpu_id);
        }
    }
}
