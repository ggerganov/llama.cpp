/*
 * Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/*
NVML API Reference

The NVIDIA Management Library (NVML) is a C-based programmatic interface for monitoring and
managing various states within NVIDIA Tesla &tm; GPUs. It is intended to be a platform for building
3rd party applications, and is also the underlying library for the NVIDIA-supported nvidia-smi
tool. NVML is thread-safe so it is safe to make simultaneous NVML calls from multiple threads.

API Documentation

Supported platforms:
- Windows:     Windows Server 2008 R2 64bit, Windows Server 2012 R2 64bit, Windows 7 64bit, Windows 8 64bit, Windows 10 64bit
- Linux:       32-bit and 64-bit
- Hypervisors: Windows Server 2008R2/2012 Hyper-V 64bit, Citrix XenServer 6.2 SP1+, VMware ESX 5.1/5.5

Supported products:
- Full Support
    - All Tesla products, starting with the Fermi architecture
    - All Quadro products, starting with the Fermi architecture
    - All vGPU Software products, starting with the Kepler architecture
    - Selected GeForce Titan products
- Limited Support
    - All Geforce products, starting with the Fermi architecture

The NVML library can be found at \%ProgramW6432\%\\"NVIDIA Corporation"\\NVSMI\\ on Windows. It is
not be added to the system path by default. To dynamically link to NVML, add this path to the PATH
environmental variable. To dynamically load NVML, call LoadLibrary with this path.

On Linux the NVML library will be found on the standard library path. For 64 bit Linux, both the 32 bit
and 64 bit NVML libraries will be installed.

Online documentation for this library is available at http://docs.nvidia.com/deploy/nvml-api/index.html
*/

#ifndef __nvml_nvml_h__
#define __nvml_nvml_h__

#ifdef __cplusplus
extern "C" {
#endif

/*
 * On Windows, set up methods for DLL export
 * define NVML_STATIC_IMPORT when using nvml_loader library
 */
#if defined _WINDOWS
    #if !defined NVML_STATIC_IMPORT
        #if defined NVML_LIB_EXPORT
            #define DECLDIR __declspec(dllexport)
        #else
            #define DECLDIR __declspec(dllimport)
        #endif
    #else
        #define DECLDIR
    #endif
#else
    #define DECLDIR
#endif

/**
 * NVML API versioning support
 */
#define NVML_API_VERSION            11
#define NVML_API_VERSION_STR        "11"
/**
 * Defining NVML_NO_UNVERSIONED_FUNC_DEFS will disable "auto upgrading" of APIs.
 * e.g. the user will have to call nvmlInit_v2 instead of nvmlInit. Enable this
 * guard if you need to support older versions of the API
 */
#ifndef NVML_NO_UNVERSIONED_FUNC_DEFS
    #define nvmlInit                                    nvmlInit_v2
    #define nvmlDeviceGetPciInfo                        nvmlDeviceGetPciInfo_v3
    #define nvmlDeviceGetCount                          nvmlDeviceGetCount_v2
    #define nvmlDeviceGetHandleByIndex                  nvmlDeviceGetHandleByIndex_v2
    #define nvmlDeviceGetHandleByPciBusId               nvmlDeviceGetHandleByPciBusId_v2
    #define nvmlDeviceGetNvLinkRemotePciInfo            nvmlDeviceGetNvLinkRemotePciInfo_v2
    #define nvmlDeviceRemoveGpu                         nvmlDeviceRemoveGpu_v2
    #define nvmlDeviceGetGridLicensableFeatures         nvmlDeviceGetGridLicensableFeatures_v4
    #define nvmlEventSetWait                            nvmlEventSetWait_v2
    #define nvmlDeviceGetAttributes                     nvmlDeviceGetAttributes_v2
    #define nvmlComputeInstanceGetInfo                  nvmlComputeInstanceGetInfo_v2
    #define nvmlDeviceGetComputeRunningProcesses        nvmlDeviceGetComputeRunningProcesses_v3
    #define nvmlDeviceGetGraphicsRunningProcesses       nvmlDeviceGetGraphicsRunningProcesses_v3
    #define nvmlDeviceGetMPSComputeRunningProcesses     nvmlDeviceGetMPSComputeRunningProcesses_v3
    #define nvmlBlacklistDeviceInfo_t                   nvmlExcludedDeviceInfo_t
    #define nvmlGetBlacklistDeviceCount                 nvmlGetExcludedDeviceCount
    #define nvmlGetBlacklistDeviceInfoByIndex           nvmlGetExcludedDeviceInfoByIndex
    #define nvmlDeviceGetGpuInstancePossiblePlacements  nvmlDeviceGetGpuInstancePossiblePlacements_v2
    #define nvmlVgpuInstanceGetLicenseInfo              nvmlVgpuInstanceGetLicenseInfo_v2
#endif // #ifndef NVML_NO_UNVERSIONED_FUNC_DEFS

#define NVML_STRUCT_VERSION(data, ver) (unsigned int)(sizeof(nvml ## data ## _v ## ver ## _t) | \
                                                      (ver << 24U))

/***************************************************************************************************/
/** @defgroup nvmlDeviceStructs Device Structs
 *  @{
 */
/***************************************************************************************************/

/**
 * Special constant that some fields take when they are not available.
 * Used when only part of the struct is not available.
 *
 * Each structure explicitly states when to check for this value.
 */
#define NVML_VALUE_NOT_AVAILABLE (-1)

typedef struct nvmlDevice_st* nvmlDevice_t;

/**
 * Buffer size guaranteed to be large enough for pci bus id
 */
#define NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE      32

/**
 * Buffer size guaranteed to be large enough for pci bus id for ::busIdLegacy
 */
#define NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE   16

/**
 * PCI information about a GPU device.
 */
typedef struct nvmlPciInfo_st
{
    char busIdLegacy[NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE]; //!< The legacy tuple domain:bus:device.function PCI identifier (&amp; NULL terminator)
    unsigned int domain;             //!< The PCI domain on which the device's bus resides, 0 to 0xffffffff
    unsigned int bus;                //!< The bus on which the device resides, 0 to 0xff
    unsigned int device;             //!< The device's id on the bus, 0 to 31
    unsigned int pciDeviceId;        //!< The combined 16-bit device id and 16-bit vendor id

    // Added in NVML 2.285 API
    unsigned int pciSubSystemId;     //!< The 32-bit Sub System Device ID

    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE]; //!< The tuple domain:bus:device.function PCI identifier (&amp; NULL terminator)
} nvmlPciInfo_t;

/**
 * PCI format string for ::busIdLegacy
 */
#define NVML_DEVICE_PCI_BUS_ID_LEGACY_FMT           "%04X:%02X:%02X.0"

/**
 * PCI format string for ::busId
 */
#define NVML_DEVICE_PCI_BUS_ID_FMT                  "%08X:%02X:%02X.0"

/**
 * Utility macro for filling the pci bus id format from a nvmlPciInfo_t
 */
#define NVML_DEVICE_PCI_BUS_ID_FMT_ARGS(pciInfo)    (pciInfo)->domain, \
                                                    (pciInfo)->bus,    \
                                                    (pciInfo)->device

/**
 * Detailed ECC error counts for a device.
 *
 * @deprecated  Different GPU families can have different memory error counters
 *              See \ref nvmlDeviceGetMemoryErrorCounter
 */
typedef struct nvmlEccErrorCounts_st
{
    unsigned long long l1Cache;      //!< L1 cache errors
    unsigned long long l2Cache;      //!< L2 cache errors
    unsigned long long deviceMemory; //!< Device memory errors
    unsigned long long registerFile; //!< Register file errors
} nvmlEccErrorCounts_t;

/**
 * Utilization information for a device.
 * Each sample period may be between 1 second and 1/6 second, depending on the product being queried.
 */
typedef struct nvmlUtilization_st
{
    unsigned int gpu;                //!< Percent of time over the past sample period during which one or more kernels was executing on the GPU
    unsigned int memory;             //!< Percent of time over the past sample period during which global (device) memory was being read or written
} nvmlUtilization_t;

/**
 * Memory allocation information for a device (v1).
 * The total amount is equal to the sum of the amounts of free and used memory.
 */
typedef struct nvmlMemory_st
{
    unsigned long long total;        //!< Total physical device memory (in bytes)
    unsigned long long free;         //!< Unallocated device memory (in bytes)
    unsigned long long used;         //!< Sum of Reserved and Allocated device memory (in bytes).
                                     //!< Note that the driver/GPU always sets aside a small amount of memory for bookkeeping
} nvmlMemory_t;

/**
 * Memory allocation information for a device (v2).
 * 
 * Version 2 adds versioning for the struct and the amount of system-reserved memory as an output.
 * @note The \ref nvmlMemory_v2_t.used amount also includes the \ref nvmlMemory_v2_t.reserved amount.
 */
typedef struct nvmlMemory_v2_st
{
    unsigned int version;            //!< Structure format version (must be 2)
    unsigned long long total;        //!< Total physical device memory (in bytes)
    unsigned long long reserved;     //!< Device memory (in bytes) reserved for system use (driver or firmware)
    unsigned long long free;         //!< Unallocated device memory (in bytes)
    unsigned long long used;         //!< Allocated device memory (in bytes). Note that the driver/GPU always sets aside a small amount of memory for bookkeeping
} nvmlMemory_v2_t;

#define nvmlMemory_v2 NVML_STRUCT_VERSION(Memory, 2)

/**
 * BAR1 Memory allocation Information for a device
 */
typedef struct nvmlBAR1Memory_st
{
    unsigned long long bar1Total;    //!< Total BAR1 Memory (in bytes)
    unsigned long long bar1Free;     //!< Unallocated BAR1 Memory (in bytes)
    unsigned long long bar1Used;     //!< Allocated Used Memory (in bytes)
}nvmlBAR1Memory_t;

/**
 * Information about running compute processes on the GPU, legacy version
 * for older versions of the API.
 */
typedef struct nvmlProcessInfo_v1_st
{
    unsigned int        pid;                //!< Process ID
    unsigned long long  usedGpuMemory;      //!< Amount of used GPU memory in bytes.
                                            //! Under WDDM, \ref NVML_VALUE_NOT_AVAILABLE is always reported
                                            //! because Windows KMD manages all the memory and not the NVIDIA driver
} nvmlProcessInfo_v1_t;

/**
 * Information about running compute processes on the GPU
 */
typedef struct nvmlProcessInfo_v2_st
{
    unsigned int        pid;                //!< Process ID
    unsigned long long  usedGpuMemory;      //!< Amount of used GPU memory in bytes.
                                            //! Under WDDM, \ref NVML_VALUE_NOT_AVAILABLE is always reported
                                            //! because Windows KMD manages all the memory and not the NVIDIA driver
    unsigned int        gpuInstanceId;      //!< If MIG is enabled, stores a valid GPU instance ID. gpuInstanceId is set to
                                            //  0xFFFFFFFF otherwise.
    unsigned int        computeInstanceId;  //!< If MIG is enabled, stores a valid compute instance ID. computeInstanceId is set to
                                            //  0xFFFFFFFF otherwise.
} nvmlProcessInfo_v2_t;

/**
 * Information about running compute processes on the GPU
 * Version 2 adds versioning for the struct and the conf compute protected memory in output.
 */
typedef struct nvmlProcessInfo_st
{
    unsigned int        pid;                //!< Process ID
    unsigned long long  usedGpuMemory;      //!< Amount of used GPU memory in bytes.
                                            //! Under WDDM, \ref NVML_VALUE_NOT_AVAILABLE is always reported
                                            //! because Windows KMD manages all the memory and not the NVIDIA driver
    unsigned int        gpuInstanceId;      //!< If MIG is enabled, stores a valid GPU instance ID. gpuInstanceId is set to
                                            //  0xFFFFFFFF otherwise.
    unsigned int        computeInstanceId;  //!< If MIG is enabled, stores a valid compute instance ID. computeInstanceId is set to
                                            //  0xFFFFFFFF otherwise.
} nvmlProcessInfo_t;

typedef struct nvmlDeviceAttributes_st
{
    unsigned int multiprocessorCount;       //!< Streaming Multiprocessor count
    unsigned int sharedCopyEngineCount;     //!< Shared Copy Engine count
    unsigned int sharedDecoderCount;        //!< Shared Decoder Engine count
    unsigned int sharedEncoderCount;        //!< Shared Encoder Engine count
    unsigned int sharedJpegCount;           //!< Shared JPEG Engine count
    unsigned int sharedOfaCount;            //!< Shared OFA Engine count
    unsigned int gpuInstanceSliceCount;     //!< GPU instance slice count
    unsigned int computeInstanceSliceCount; //!< Compute instance slice count
    unsigned long long memorySizeMB;        //!< Device memory size (in MiB)
} nvmlDeviceAttributes_t;

/**
 * Possible values that classify the remap availability for each bank. The max
 * field will contain the number of banks that have maximum remap availability
 * (all reserved rows are available). None means that there are no reserved
 * rows available.
 */
typedef struct nvmlRowRemapperHistogramValues_st
{
    unsigned int max;
    unsigned int high;
    unsigned int partial;
    unsigned int low;
    unsigned int none;
} nvmlRowRemapperHistogramValues_t;

/**
 * Enum to represent type of bridge chip
 */
typedef enum nvmlBridgeChipType_enum
{
    NVML_BRIDGE_CHIP_PLX = 0,
    NVML_BRIDGE_CHIP_BRO4 = 1
}nvmlBridgeChipType_t;

/**
 * Maximum number of NvLink links supported
 */
#define NVML_NVLINK_MAX_LINKS 12

/**
 * Enum to represent the NvLink utilization counter packet units
 */
typedef enum nvmlNvLinkUtilizationCountUnits_enum
{
    NVML_NVLINK_COUNTER_UNIT_CYCLES =  0,     // count by cycles
    NVML_NVLINK_COUNTER_UNIT_PACKETS = 1,     // count by packets
    NVML_NVLINK_COUNTER_UNIT_BYTES   = 2,     // count by bytes
    NVML_NVLINK_COUNTER_UNIT_RESERVED = 3,    // count reserved for internal use
    // this must be last
    NVML_NVLINK_COUNTER_UNIT_COUNT
} nvmlNvLinkUtilizationCountUnits_t;

/**
 * Enum to represent the NvLink utilization counter packet types to count
 *  ** this is ONLY applicable with the units as packets or bytes
 *  ** as specified in \a nvmlNvLinkUtilizationCountUnits_t
 *  ** all packet filter descriptions are target GPU centric
 *  ** these can be "OR'd" together
 */
typedef enum nvmlNvLinkUtilizationCountPktTypes_enum
{
    NVML_NVLINK_COUNTER_PKTFILTER_NOP        = 0x1,     // no operation packets
    NVML_NVLINK_COUNTER_PKTFILTER_READ       = 0x2,     // read packets
    NVML_NVLINK_COUNTER_PKTFILTER_WRITE      = 0x4,     // write packets
    NVML_NVLINK_COUNTER_PKTFILTER_RATOM      = 0x8,     // reduction atomic requests
    NVML_NVLINK_COUNTER_PKTFILTER_NRATOM     = 0x10,    // non-reduction atomic requests
    NVML_NVLINK_COUNTER_PKTFILTER_FLUSH      = 0x20,    // flush requests
    NVML_NVLINK_COUNTER_PKTFILTER_RESPDATA   = 0x40,    // responses with data
    NVML_NVLINK_COUNTER_PKTFILTER_RESPNODATA = 0x80,    // responses without data
    NVML_NVLINK_COUNTER_PKTFILTER_ALL        = 0xFF     // all packets
} nvmlNvLinkUtilizationCountPktTypes_t;

/**
 * Struct to define the NVLINK counter controls
 */
typedef struct nvmlNvLinkUtilizationControl_st
{
    nvmlNvLinkUtilizationCountUnits_t units;
    nvmlNvLinkUtilizationCountPktTypes_t pktfilter;
} nvmlNvLinkUtilizationControl_t;

/**
 * Enum to represent NvLink queryable capabilities
 */
typedef enum nvmlNvLinkCapability_enum
{
    NVML_NVLINK_CAP_P2P_SUPPORTED = 0,     // P2P over NVLink is supported
    NVML_NVLINK_CAP_SYSMEM_ACCESS = 1,     // Access to system memory is supported
    NVML_NVLINK_CAP_P2P_ATOMICS   = 2,     // P2P atomics are supported
    NVML_NVLINK_CAP_SYSMEM_ATOMICS= 3,     // System memory atomics are supported
    NVML_NVLINK_CAP_SLI_BRIDGE    = 4,     // SLI is supported over this link
    NVML_NVLINK_CAP_VALID         = 5,     // Link is supported on this device
    // should be last
    NVML_NVLINK_CAP_COUNT
} nvmlNvLinkCapability_t;

/**
 * Enum to represent NvLink queryable error counters
 */
typedef enum nvmlNvLinkErrorCounter_enum
{
    NVML_NVLINK_ERROR_DL_REPLAY   = 0,     // Data link transmit replay error counter
    NVML_NVLINK_ERROR_DL_RECOVERY = 1,     // Data link transmit recovery error counter
    NVML_NVLINK_ERROR_DL_CRC_FLIT = 2,     // Data link receive flow control digit CRC error counter
    NVML_NVLINK_ERROR_DL_CRC_DATA = 3,     // Data link receive data CRC error counter
    NVML_NVLINK_ERROR_DL_ECC_DATA = 4,     // Data link receive data ECC error counter

    // this must be last
    NVML_NVLINK_ERROR_COUNT
} nvmlNvLinkErrorCounter_t;

/**
 * Enum to represent NvLink's remote device type
 */
typedef enum nvmlIntNvLinkDeviceType_enum
{
    NVML_NVLINK_DEVICE_TYPE_GPU     = 0x00,
    NVML_NVLINK_DEVICE_TYPE_IBMNPU  = 0x01,
    NVML_NVLINK_DEVICE_TYPE_SWITCH  = 0x02,
    NVML_NVLINK_DEVICE_TYPE_UNKNOWN = 0xFF
} nvmlIntNvLinkDeviceType_t;

/**
 * Represents level relationships within a system between two GPUs
 * The enums are spaced to allow for future relationships
 */
typedef enum nvmlGpuLevel_enum
{
    NVML_TOPOLOGY_INTERNAL           = 0, // e.g. Tesla K80
    NVML_TOPOLOGY_SINGLE             = 10, // all devices that only need traverse a single PCIe switch
    NVML_TOPOLOGY_MULTIPLE           = 20, // all devices that need not traverse a host bridge
    NVML_TOPOLOGY_HOSTBRIDGE         = 30, // all devices that are connected to the same host bridge
    NVML_TOPOLOGY_NODE               = 40, // all devices that are connected to the same NUMA node but possibly multiple host bridges
    NVML_TOPOLOGY_SYSTEM             = 50  // all devices in the system

    // there is purposefully no COUNT here because of the need for spacing above
} nvmlGpuTopologyLevel_t;

/* Compatibility for CPU->NODE renaming */
#define NVML_TOPOLOGY_CPU NVML_TOPOLOGY_NODE

/* P2P Capability Index Status*/
typedef enum nvmlGpuP2PStatus_enum
{
    NVML_P2P_STATUS_OK     = 0,
    NVML_P2P_STATUS_CHIPSET_NOT_SUPPORED,
    NVML_P2P_STATUS_GPU_NOT_SUPPORTED,
    NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED,
    NVML_P2P_STATUS_DISABLED_BY_REGKEY,
    NVML_P2P_STATUS_NOT_SUPPORTED,
    NVML_P2P_STATUS_UNKNOWN

} nvmlGpuP2PStatus_t;

/* P2P Capability Index*/
typedef enum nvmlGpuP2PCapsIndex_enum
{
    NVML_P2P_CAPS_INDEX_READ = 0,
    NVML_P2P_CAPS_INDEX_WRITE,
    NVML_P2P_CAPS_INDEX_NVLINK,
    NVML_P2P_CAPS_INDEX_ATOMICS,
    NVML_P2P_CAPS_INDEX_PROP,
    NVML_P2P_CAPS_INDEX_UNKNOWN
}nvmlGpuP2PCapsIndex_t;

/**
 * Maximum limit on Physical Bridges per Board
 */
#define NVML_MAX_PHYSICAL_BRIDGE                         (128)

/**
 * Information about the Bridge Chip Firmware
 */
typedef struct nvmlBridgeChipInfo_st
{
    nvmlBridgeChipType_t type;                  //!< Type of Bridge Chip
    unsigned int fwVersion;                     //!< Firmware Version. 0=Version is unavailable
}nvmlBridgeChipInfo_t;

/**
 * This structure stores the complete Hierarchy of the Bridge Chip within the board. The immediate
 * bridge is stored at index 0 of bridgeInfoList, parent to immediate bridge is at index 1 and so forth.
 */
typedef struct nvmlBridgeChipHierarchy_st
{
    unsigned char  bridgeCount;                 //!< Number of Bridge Chips on the Board
    nvmlBridgeChipInfo_t bridgeChipInfo[NVML_MAX_PHYSICAL_BRIDGE]; //!< Hierarchy of Bridge Chips on the board
}nvmlBridgeChipHierarchy_t;

/**
 *  Represents Type of Sampling Event
 */
typedef enum nvmlSamplingType_enum
{
    NVML_TOTAL_POWER_SAMPLES        = 0, //!< To represent total power drawn by GPU
    NVML_GPU_UTILIZATION_SAMPLES    = 1, //!< To represent percent of time during which one or more kernels was executing on the GPU
    NVML_MEMORY_UTILIZATION_SAMPLES = 2, //!< To represent percent of time during which global (device) memory was being read or written
    NVML_ENC_UTILIZATION_SAMPLES    = 3, //!< To represent percent of time during which NVENC remains busy
    NVML_DEC_UTILIZATION_SAMPLES    = 4, //!< To represent percent of time during which NVDEC remains busy
    NVML_PROCESSOR_CLK_SAMPLES      = 5, //!< To represent processor clock samples
    NVML_MEMORY_CLK_SAMPLES         = 6, //!< To represent memory clock samples

    // Keep this last
    NVML_SAMPLINGTYPE_COUNT
}nvmlSamplingType_t;

/**
 * Represents the queryable PCIe utilization counters
 */
typedef enum nvmlPcieUtilCounter_enum
{
    NVML_PCIE_UTIL_TX_BYTES             = 0, // 1KB granularity
    NVML_PCIE_UTIL_RX_BYTES             = 1, // 1KB granularity

    // Keep this last
    NVML_PCIE_UTIL_COUNT
} nvmlPcieUtilCounter_t;

/**
 * Represents the type for sample value returned
 */
typedef enum nvmlValueType_enum
{
    NVML_VALUE_TYPE_DOUBLE = 0,
    NVML_VALUE_TYPE_UNSIGNED_INT = 1,
    NVML_VALUE_TYPE_UNSIGNED_LONG = 2,
    NVML_VALUE_TYPE_UNSIGNED_LONG_LONG = 3,
    NVML_VALUE_TYPE_SIGNED_LONG_LONG = 4,

    // Keep this last
    NVML_VALUE_TYPE_COUNT
}nvmlValueType_t;


/**
 * Union to represent different types of Value
 */
typedef union nvmlValue_st
{
    double dVal;                    //!< If the value is double
    unsigned int uiVal;             //!< If the value is unsigned int
    unsigned long ulVal;            //!< If the value is unsigned long
    unsigned long long ullVal;      //!< If the value is unsigned long long
    signed long long sllVal;        //!< If the value is signed long long
}nvmlValue_t;

/**
 * Information for Sample
 */
typedef struct nvmlSample_st
{
    unsigned long long timeStamp;       //!< CPU Timestamp in microseconds
    nvmlValue_t sampleValue;        //!< Sample Value
}nvmlSample_t;

/**
 * Represents type of perf policy for which violation times can be queried
 */
typedef enum nvmlPerfPolicyType_enum
{
    NVML_PERF_POLICY_POWER = 0,              //!< How long did power violations cause the GPU to be below application clocks
    NVML_PERF_POLICY_THERMAL = 1,            //!< How long did thermal violations cause the GPU to be below application clocks
    NVML_PERF_POLICY_SYNC_BOOST = 2,         //!< How long did sync boost cause the GPU to be below application clocks
    NVML_PERF_POLICY_BOARD_LIMIT = 3,        //!< How long did the board limit cause the GPU to be below application clocks
    NVML_PERF_POLICY_LOW_UTILIZATION = 4,    //!< How long did low utilization cause the GPU to be below application clocks
    NVML_PERF_POLICY_RELIABILITY = 5,        //!< How long did the board reliability limit cause the GPU to be below application clocks

    NVML_PERF_POLICY_TOTAL_APP_CLOCKS = 10,  //!< Total time the GPU was held below application clocks by any limiter (0 - 5 above)
    NVML_PERF_POLICY_TOTAL_BASE_CLOCKS = 11, //!< Total time the GPU was held below base clocks

    // Keep this last
    NVML_PERF_POLICY_COUNT
}nvmlPerfPolicyType_t;

/**
 * Struct to hold perf policy violation status data
 */
typedef struct nvmlViolationTime_st
{
    unsigned long long referenceTime;  //!< referenceTime represents CPU timestamp in microseconds
    unsigned long long violationTime;  //!< violationTime in Nanoseconds
}nvmlViolationTime_t;

/** @} */

/***************************************************************************************************/
/** @defgroup nvmlDeviceEnumvs Device Enums
 *  @{
 */
/***************************************************************************************************/

/**
 * Generic enable/disable enum.
 */
typedef enum nvmlEnableState_enum
{
    NVML_FEATURE_DISABLED    = 0,     //!< Feature disabled
    NVML_FEATURE_ENABLED     = 1      //!< Feature enabled
} nvmlEnableState_t;

//! Generic flag used to specify the default behavior of some functions. See description of particular functions for details.
#define nvmlFlagDefault     0x00
//! Generic flag used to force some behavior. See description of particular functions for details.
#define nvmlFlagForce       0x01

/**
 *  * The Brand of the GPU
 *   */
typedef enum nvmlBrandType_enum
{
    NVML_BRAND_UNKNOWN              = 0,
    NVML_BRAND_QUADRO               = 1,
    NVML_BRAND_TESLA                = 2,
    NVML_BRAND_NVS                  = 3,
    NVML_BRAND_GRID                 = 4,   // Deprecated from API reporting. Keeping definition for backward compatibility.
    NVML_BRAND_GEFORCE              = 5,
    NVML_BRAND_TITAN                = 6,
    NVML_BRAND_NVIDIA_VAPPS         = 7,   // NVIDIA Virtual Applications
    NVML_BRAND_NVIDIA_VPC           = 8,   // NVIDIA Virtual PC
    NVML_BRAND_NVIDIA_VCS           = 9,   // NVIDIA Virtual Compute Server
    NVML_BRAND_NVIDIA_VWS           = 10,  // NVIDIA RTX Virtual Workstation
    NVML_BRAND_NVIDIA_CLOUD_GAMING  = 11,  // NVIDIA Cloud Gaming
    NVML_BRAND_NVIDIA_VGAMING       = NVML_BRAND_NVIDIA_CLOUD_GAMING,  // Deprecated from API reporting. Keeping definition for backward compatibility.
    NVML_BRAND_QUADRO_RTX           = 12,
    NVML_BRAND_NVIDIA_RTX           = 13,
    NVML_BRAND_NVIDIA               = 14,
    NVML_BRAND_GEFORCE_RTX          = 15,  // Unused
    NVML_BRAND_TITAN_RTX            = 16,  // Unused

    // Keep this last
    NVML_BRAND_COUNT
} nvmlBrandType_t;

/**
 * Temperature thresholds.
 */
typedef enum nvmlTemperatureThresholds_enum
{
    NVML_TEMPERATURE_THRESHOLD_SHUTDOWN      = 0, // Temperature at which the GPU will
                                                  // shut down for HW protection
    NVML_TEMPERATURE_THRESHOLD_SLOWDOWN      = 1, // Temperature at which the GPU will
                                                  // begin HW slowdown
    NVML_TEMPERATURE_THRESHOLD_MEM_MAX       = 2, // Memory Temperature at which the GPU will
                                                  // begin SW slowdown
    NVML_TEMPERATURE_THRESHOLD_GPU_MAX       = 3, // GPU Temperature at which the GPU
                                                  // can be throttled below base clock
    NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MIN  = 4, // Minimum GPU Temperature that can be
                                                  // set as acoustic threshold
    NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_CURR = 5, // Current temperature that is set as
                                                  // acoustic threshold.
    NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MAX  = 6, // Maximum GPU temperature that can be
                                                  // set as acoustic threshold.
    // Keep this last
    NVML_TEMPERATURE_THRESHOLD_COUNT
} nvmlTemperatureThresholds_t;

/**
 * Temperature sensors.
 */
typedef enum nvmlTemperatureSensors_enum
{
    NVML_TEMPERATURE_GPU      = 0,    //!< Temperature sensor for the GPU die

    // Keep this last
    NVML_TEMPERATURE_COUNT
} nvmlTemperatureSensors_t;

/**
 * Compute mode.
 *
 * NVML_COMPUTEMODE_EXCLUSIVE_PROCESS was added in CUDA 4.0.
 * Earlier CUDA versions supported a single exclusive mode,
 * which is equivalent to NVML_COMPUTEMODE_EXCLUSIVE_THREAD in CUDA 4.0 and beyond.
 */
typedef enum nvmlComputeMode_enum
{
    NVML_COMPUTEMODE_DEFAULT           = 0,  //!< Default compute mode -- multiple contexts per device
    NVML_COMPUTEMODE_EXCLUSIVE_THREAD  = 1,  //!< Support Removed
    NVML_COMPUTEMODE_PROHIBITED        = 2,  //!< Compute-prohibited mode -- no contexts per device
    NVML_COMPUTEMODE_EXCLUSIVE_PROCESS = 3,  //!< Compute-exclusive-process mode -- only one context per device, usable from multiple threads at a time

    // Keep this last
    NVML_COMPUTEMODE_COUNT
} nvmlComputeMode_t;

/**
 * Max Clock Monitors available
 */
#define MAX_CLK_DOMAINS			32

/**
 * Clock Monitor error types
 */
typedef struct nvmlClkMonFaultInfo_struct {
    /**
     * The Domain which faulted
     */
    unsigned int   clkApiDomain;

    /**
     * Faults Information
     */
    unsigned int   clkDomainFaultMask;
} nvmlClkMonFaultInfo_t;

/**
 * Clock Monitor Status
 */
typedef struct nvmlClkMonStatus_status {
    /**
     * Fault status Indicator
     */
    unsigned int  bGlobalStatus;

    /**
     * Total faulted domain numbers
     */
    unsigned int   clkMonListSize;

    /**
     * The fault Information structure
     */
    nvmlClkMonFaultInfo_t clkMonList[MAX_CLK_DOMAINS];
} nvmlClkMonStatus_t;

/** 
 * ECC bit types.
 *
 * @deprecated See \ref nvmlMemoryErrorType_t for a more flexible type
 */
#define nvmlEccBitType_t nvmlMemoryErrorType_t

/**
 * Single bit ECC errors
 *
 * @deprecated Mapped to \ref NVML_MEMORY_ERROR_TYPE_CORRECTED
 */
#define NVML_SINGLE_BIT_ECC NVML_MEMORY_ERROR_TYPE_CORRECTED

/**
 * Double bit ECC errors
 *
 * @deprecated Mapped to \ref NVML_MEMORY_ERROR_TYPE_UNCORRECTED
 */
#define NVML_DOUBLE_BIT_ECC NVML_MEMORY_ERROR_TYPE_UNCORRECTED

/**
 * Memory error types
 */
typedef enum nvmlMemoryErrorType_enum
{
    /**
     * A memory error that was corrected
     *
     * For ECC errors, these are single bit errors
     * For Texture memory, these are errors fixed by resend
     */
    NVML_MEMORY_ERROR_TYPE_CORRECTED = 0,
    /**
     * A memory error that was not corrected
     *
     * For ECC errors, these are double bit errors
     * For Texture memory, these are errors where the resend fails
     */
    NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1,


    // Keep this last
    NVML_MEMORY_ERROR_TYPE_COUNT //!< Count of memory error types

} nvmlMemoryErrorType_t;

/**
 * ECC counter types.
 *
 * Note: Volatile counts are reset each time the driver loads. On Windows this is once per boot. On Linux this can be more frequent.
 *       On Linux the driver unloads when no active clients exist. If persistence mode is enabled or there is always a driver
 *       client active (e.g. X11), then Linux also sees per-boot behavior. If not, volatile counts are reset each time a compute app
 *       is run.
 */
typedef enum nvmlEccCounterType_enum
{
    NVML_VOLATILE_ECC      = 0,      //!< Volatile counts are reset each time the driver loads.
    NVML_AGGREGATE_ECC     = 1,      //!< Aggregate counts persist across reboots (i.e. for the lifetime of the device)

    // Keep this last
    NVML_ECC_COUNTER_TYPE_COUNT      //!< Count of memory counter types
} nvmlEccCounterType_t;

/**
 * Clock types.
 *
 * All speeds are in Mhz.
 */
typedef enum nvmlClockType_enum
{
    NVML_CLOCK_GRAPHICS  = 0,        //!< Graphics clock domain
    NVML_CLOCK_SM        = 1,        //!< SM clock domain
    NVML_CLOCK_MEM       = 2,        //!< Memory clock domain
    NVML_CLOCK_VIDEO     = 3,        //!< Video encoder/decoder clock domain

    // Keep this last
    NVML_CLOCK_COUNT //!< Count of clock types
} nvmlClockType_t;

/**
 * Clock Ids.  These are used in combination with nvmlClockType_t
 * to specify a single clock value.
 */
typedef enum nvmlClockId_enum
{
    NVML_CLOCK_ID_CURRENT            = 0,   //!< Current actual clock value
    NVML_CLOCK_ID_APP_CLOCK_TARGET   = 1,   //!< Target application clock
    NVML_CLOCK_ID_APP_CLOCK_DEFAULT  = 2,   //!< Default application clock target
    NVML_CLOCK_ID_CUSTOMER_BOOST_MAX = 3,   //!< OEM-defined maximum clock rate

    //Keep this last
    NVML_CLOCK_ID_COUNT //!< Count of Clock Ids.
} nvmlClockId_t;

/**
 * Driver models.
 *
 * Windows only.
 */
typedef enum nvmlDriverModel_enum
{
    NVML_DRIVER_WDDM      = 0,       //!< WDDM driver model -- GPU treated as a display device
    NVML_DRIVER_WDM       = 1        //!< WDM (TCC) model (recommended) -- GPU treated as a generic device
} nvmlDriverModel_t;

/**
 * Allowed PStates.
 */
typedef enum nvmlPStates_enum
{
    NVML_PSTATE_0               = 0,       //!< Performance state 0 -- Maximum Performance
    NVML_PSTATE_1               = 1,       //!< Performance state 1
    NVML_PSTATE_2               = 2,       //!< Performance state 2
    NVML_PSTATE_3               = 3,       //!< Performance state 3
    NVML_PSTATE_4               = 4,       //!< Performance state 4
    NVML_PSTATE_5               = 5,       //!< Performance state 5
    NVML_PSTATE_6               = 6,       //!< Performance state 6
    NVML_PSTATE_7               = 7,       //!< Performance state 7
    NVML_PSTATE_8               = 8,       //!< Performance state 8
    NVML_PSTATE_9               = 9,       //!< Performance state 9
    NVML_PSTATE_10              = 10,      //!< Performance state 10
    NVML_PSTATE_11              = 11,      //!< Performance state 11
    NVML_PSTATE_12              = 12,      //!< Performance state 12
    NVML_PSTATE_13              = 13,      //!< Performance state 13
    NVML_PSTATE_14              = 14,      //!< Performance state 14
    NVML_PSTATE_15              = 15,      //!< Performance state 15 -- Minimum Performance
    NVML_PSTATE_UNKNOWN         = 32       //!< Unknown performance state
} nvmlPstates_t;

/**
 * GPU Operation Mode
 *
 * GOM allows to reduce power usage and optimize GPU throughput by disabling GPU features.
 *
 * Each GOM is designed to meet specific user needs.
 */
typedef enum nvmlGom_enum
{
    NVML_GOM_ALL_ON                    = 0, //!< Everything is enabled and running at full speed

    NVML_GOM_COMPUTE                   = 1, //!< Designed for running only compute tasks. Graphics operations
                                            //!< are not allowed

    NVML_GOM_LOW_DP                    = 2  //!< Designed for running graphics applications that don't require
                                            //!< high bandwidth double precision
} nvmlGpuOperationMode_t;

/**
 * Available infoROM objects.
 */
typedef enum nvmlInforomObject_enum
{
    NVML_INFOROM_OEM            = 0,       //!< An object defined by OEM
    NVML_INFOROM_ECC            = 1,       //!< The ECC object determining the level of ECC support
    NVML_INFOROM_POWER          = 2,       //!< The power management object

    // Keep this last
    NVML_INFOROM_COUNT                     //!< This counts the number of infoROM objects the driver knows about
} nvmlInforomObject_t;

/**
 * Return values for NVML API calls.
 */
typedef enum nvmlReturn_enum
{
    // cppcheck-suppress *
    NVML_SUCCESS = 0,                        //!< The operation was successful
    NVML_ERROR_UNINITIALIZED = 1,            //!< NVML was not first initialized with nvmlInit()
    NVML_ERROR_INVALID_ARGUMENT = 2,         //!< A supplied argument is invalid
    NVML_ERROR_NOT_SUPPORTED = 3,            //!< The requested operation is not available on target device
    NVML_ERROR_NO_PERMISSION = 4,            //!< The current user does not have permission for operation
    NVML_ERROR_ALREADY_INITIALIZED = 5,      //!< Deprecated: Multiple initializations are now allowed through ref counting
    NVML_ERROR_NOT_FOUND = 6,                //!< A query to find an object was unsuccessful
    NVML_ERROR_INSUFFICIENT_SIZE = 7,        //!< An input argument is not large enough
    NVML_ERROR_INSUFFICIENT_POWER = 8,       //!< A device's external power cables are not properly attached
    NVML_ERROR_DRIVER_NOT_LOADED = 9,        //!< NVIDIA driver is not loaded
    NVML_ERROR_TIMEOUT = 10,                 //!< User provided timeout passed
    NVML_ERROR_IRQ_ISSUE = 11,               //!< NVIDIA Kernel detected an interrupt issue with a GPU
    NVML_ERROR_LIBRARY_NOT_FOUND = 12,       //!< NVML Shared Library couldn't be found or loaded
    NVML_ERROR_FUNCTION_NOT_FOUND = 13,      //!< Local version of NVML doesn't implement this function
    NVML_ERROR_CORRUPTED_INFOROM = 14,       //!< infoROM is corrupted
    NVML_ERROR_GPU_IS_LOST = 15,             //!< The GPU has fallen off the bus or has otherwise become inaccessible
    NVML_ERROR_RESET_REQUIRED = 16,          //!< The GPU requires a reset before it can be used again
    NVML_ERROR_OPERATING_SYSTEM = 17,        //!< The GPU control device has been blocked by the operating system/cgroups
    NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18, //!< RM detects a driver/library version mismatch
    NVML_ERROR_IN_USE = 19,                  //!< An operation cannot be performed because the GPU is currently in use
    NVML_ERROR_MEMORY = 20,                  //!< Insufficient memory
    NVML_ERROR_NO_DATA = 21,                 //!< No data
    NVML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22,  //!< The requested vgpu operation is not available on target device, becasue ECC is enabled
    NVML_ERROR_INSUFFICIENT_RESOURCES = 23,  //!< Ran out of critical resources, other than memory
    NVML_ERROR_FREQ_NOT_SUPPORTED = 24,  //!< Ran out of critical resources, other than memory
    NVML_ERROR_UNKNOWN = 999                 //!< An internal driver error occurred
} nvmlReturn_t;

/**
 * See \ref nvmlDeviceGetMemoryErrorCounter
 */
typedef enum nvmlMemoryLocation_enum
{
    NVML_MEMORY_LOCATION_L1_CACHE        = 0,    //!< GPU L1 Cache
    NVML_MEMORY_LOCATION_L2_CACHE        = 1,    //!< GPU L2 Cache
    NVML_MEMORY_LOCATION_DRAM            = 2,    //!< Turing+ DRAM
    NVML_MEMORY_LOCATION_DEVICE_MEMORY   = 2,    //!< GPU Device Memory
    NVML_MEMORY_LOCATION_REGISTER_FILE   = 3,    //!< GPU Register File
    NVML_MEMORY_LOCATION_TEXTURE_MEMORY  = 4,    //!< GPU Texture Memory
    NVML_MEMORY_LOCATION_TEXTURE_SHM     = 5,    //!< Shared memory
    NVML_MEMORY_LOCATION_CBU             = 6,    //!< CBU
    NVML_MEMORY_LOCATION_SRAM            = 7,    //!< Turing+ SRAM
    // Keep this last
    NVML_MEMORY_LOCATION_COUNT              //!< This counts the number of memory locations the driver knows about
} nvmlMemoryLocation_t;

/**
 * Causes for page retirement
 */
typedef enum nvmlPageRetirementCause_enum
{
    NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS = 0, //!< Page was retired due to multiple single bit ECC error
    NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR = 1,           //!< Page was retired due to double bit ECC error

    // Keep this last
    NVML_PAGE_RETIREMENT_CAUSE_COUNT
} nvmlPageRetirementCause_t;

/**
 * API types that allow changes to default permission restrictions
 */
typedef enum nvmlRestrictedAPI_enum
{
    NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS = 0,   //!< APIs that change application clocks, see nvmlDeviceSetApplicationsClocks
                                                      //!< and see nvmlDeviceResetApplicationsClocks
    NVML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS = 1,  //!< APIs that enable/disable Auto Boosted clocks
                                                      //!< see nvmlDeviceSetAutoBoostedClocksEnabled
    // Keep this last
    NVML_RESTRICTED_API_COUNT
} nvmlRestrictedAPI_t;

/**
 * Enum to represent NvLink ECC per-lane error counts
 */
typedef enum nvmlNvLinkEccLaneErrorCounter_enum
{
    NVML_NVLINK_ERROR_DL_ECC_LANE0 = 0, // Data link receive ECC error counter lane 0
    NVML_NVLINK_ERROR_DL_ECC_LANE1 = 1, // Data link receive ECC error counter lane 1
    NVML_NVLINK_ERROR_DL_ECC_LANE2 = 2, // Data link receive ECC error counter lane 2
    NVML_NVLINK_ERROR_DL_ECC_LANE3 = 3, // Data link receive ECC error counter lane 3

    // this must be last
    NVML_NVLINK_ERROR_DL_ECC_COUNT
} nvmlNvLinkEccLaneErrorCounter_t;

/** @} */

/***************************************************************************************************/
/** @addtogroup virtualGPU
 *  @{
 */
/***************************************************************************************************/
/** @defgroup nvmlVirtualGpuEnums vGPU Enums
 *  @{
 */
/***************************************************************************************************/

/*!
 * GPU virtualization mode types.
 */
typedef enum nvmlGpuVirtualizationMode {
    NVML_GPU_VIRTUALIZATION_MODE_NONE = 0,  //!< Represents Bare Metal GPU
    NVML_GPU_VIRTUALIZATION_MODE_PASSTHROUGH = 1,  //!< Device is associated with GPU-Passthorugh
    NVML_GPU_VIRTUALIZATION_MODE_VGPU = 2,  //!< Device is associated with vGPU inside virtual machine.
    NVML_GPU_VIRTUALIZATION_MODE_HOST_VGPU = 3,  //!< Device is associated with VGX hypervisor in vGPU mode
    NVML_GPU_VIRTUALIZATION_MODE_HOST_VSGA = 4   //!< Device is associated with VGX hypervisor in vSGA mode
} nvmlGpuVirtualizationMode_t;

/**
 * Host vGPU modes
 */
typedef enum nvmlHostVgpuMode_enum
{
    NVML_HOST_VGPU_MODE_NON_SRIOV    = 0,     //!< Non SR-IOV mode
    NVML_HOST_VGPU_MODE_SRIOV        = 1      //!< SR-IOV mode
} nvmlHostVgpuMode_t;

/*!
 * Types of VM identifiers
 */
typedef enum nvmlVgpuVmIdType {
    NVML_VGPU_VM_ID_DOMAIN_ID = 0, //!< VM ID represents DOMAIN ID
    NVML_VGPU_VM_ID_UUID = 1       //!< VM ID represents UUID
} nvmlVgpuVmIdType_t;

/**
 * vGPU GUEST info state.
 */
typedef enum nvmlVgpuGuestInfoState_enum
{
    NVML_VGPU_INSTANCE_GUEST_INFO_STATE_UNINITIALIZED = 0,  //!< Guest-dependent fields uninitialized
    NVML_VGPU_INSTANCE_GUEST_INFO_STATE_INITIALIZED   = 1   //!< Guest-dependent fields initialized
} nvmlVgpuGuestInfoState_t;

/**
 * vGPU software licensable features
 */
typedef enum {
    NVML_GRID_LICENSE_FEATURE_CODE_UNKNOWN      = 0,                                         //!< Unknown
    NVML_GRID_LICENSE_FEATURE_CODE_VGPU         = 1,                                         //!< Virtual GPU
    NVML_GRID_LICENSE_FEATURE_CODE_NVIDIA_RTX   = 2,                                         //!< Nvidia RTX
    NVML_GRID_LICENSE_FEATURE_CODE_VWORKSTATION = NVML_GRID_LICENSE_FEATURE_CODE_NVIDIA_RTX, //!< Deprecated, do not use.
    NVML_GRID_LICENSE_FEATURE_CODE_GAMING       = 3,                                         //!< Gaming
    NVML_GRID_LICENSE_FEATURE_CODE_COMPUTE      = 4                                          //!< Compute
} nvmlGridLicenseFeatureCode_t;

/**
 * Status codes for license expiry
 */
#define NVML_GRID_LICENSE_EXPIRY_NOT_AVAILABLE   0   //!< Expiry information not available
#define NVML_GRID_LICENSE_EXPIRY_INVALID         1   //!< Invalid expiry or error fetching expiry
#define NVML_GRID_LICENSE_EXPIRY_VALID           2   //!< Valid expiry
#define NVML_GRID_LICENSE_EXPIRY_NOT_APPLICABLE  3   //!< Expiry not applicable
#define NVML_GRID_LICENSE_EXPIRY_PERMANENT       4   //!< Permanent expiry

/** @} */

/***************************************************************************************************/

/** @defgroup nvmlVgpuConstants vGPU Constants
 *  @{
 */
/***************************************************************************************************/

/**
 * Buffer size guaranteed to be large enough for \ref nvmlVgpuTypeGetLicense
 */
#define NVML_GRID_LICENSE_BUFFER_SIZE       128

#define NVML_VGPU_NAME_BUFFER_SIZE          64

#define NVML_GRID_LICENSE_FEATURE_MAX_COUNT 3

#define INVALID_GPU_INSTANCE_PROFILE_ID     0xFFFFFFFF

#define INVALID_GPU_INSTANCE_ID             0xFFFFFFFF

/*!
 * Macros for vGPU instance's virtualization capabilities bitfield.
 */
#define NVML_VGPU_VIRTUALIZATION_CAP_MIGRATION         0:0
#define NVML_VGPU_VIRTUALIZATION_CAP_MIGRATION_NO      0x0
#define NVML_VGPU_VIRTUALIZATION_CAP_MIGRATION_YES     0x1

/*!
 * Macros for pGPU's virtualization capabilities bitfield.
 */
#define NVML_VGPU_PGPU_VIRTUALIZATION_CAP_MIGRATION         0:0
#define NVML_VGPU_PGPU_VIRTUALIZATION_CAP_MIGRATION_NO      0x0
#define NVML_VGPU_PGPU_VIRTUALIZATION_CAP_MIGRATION_YES     0x1

/** @} */

/***************************************************************************************************/
/** @defgroup nvmlVgpuStructs vGPU Structs
 *  @{
 */
/***************************************************************************************************/

typedef unsigned int nvmlVgpuTypeId_t;

typedef unsigned int nvmlVgpuInstance_t;

/**
 * Structure to store Utilization Value and vgpuInstance
 */
typedef struct nvmlVgpuInstanceUtilizationSample_st
{
    nvmlVgpuInstance_t  vgpuInstance;       //!< vGPU Instance
    unsigned long long  timeStamp;          //!< CPU Timestamp in microseconds
    nvmlValue_t         smUtil;             //!< SM (3D/Compute) Util Value
    nvmlValue_t         memUtil;            //!< Frame Buffer Memory Util Value
    nvmlValue_t         encUtil;            //!< Encoder Util Value
    nvmlValue_t         decUtil;            //!< Decoder Util Value
} nvmlVgpuInstanceUtilizationSample_t;

/**
 * Structure to store Utilization Value, vgpuInstance and subprocess information
 */
typedef struct nvmlVgpuProcessUtilizationSample_st
{
    nvmlVgpuInstance_t  vgpuInstance;                               //!< vGPU Instance
    unsigned int        pid;                                        //!< PID of process running within the vGPU VM
    char                processName[NVML_VGPU_NAME_BUFFER_SIZE];    //!< Name of process running within the vGPU VM
    unsigned long long  timeStamp;                                  //!< CPU Timestamp in microseconds
    unsigned int        smUtil;                                     //!< SM (3D/Compute) Util Value
    unsigned int        memUtil;                                    //!< Frame Buffer Memory Util Value
    unsigned int        encUtil;                                    //!< Encoder Util Value
    unsigned int        decUtil;                                    //!< Decoder Util Value
} nvmlVgpuProcessUtilizationSample_t;

/**
 * Structure to store the vGPU license expiry details
 */
typedef struct nvmlVgpuLicenseExpiry_st
{
    unsigned int    year;        //!< Year of license expiry
    unsigned short  month;       //!< Month of license expiry
    unsigned short  day;         //!< Day of license expiry
    unsigned short  hour;        //!< Hour of license expiry
    unsigned short  min;         //!< Minutes of license expiry
    unsigned short  sec;         //!< Seconds of license expiry
    unsigned char   status;      //!< License expiry status
} nvmlVgpuLicenseExpiry_t;

/**
 * vGPU license state
 */
#define NVML_GRID_LICENSE_STATE_UNKNOWN                 0   //!< Unknown state
#define NVML_GRID_LICENSE_STATE_UNINITIALIZED           1   //!< Uninitialized state
#define NVML_GRID_LICENSE_STATE_UNLICENSED_UNRESTRICTED 2   //!< Unlicensed unrestricted state
#define NVML_GRID_LICENSE_STATE_UNLICENSED_RESTRICTED   3   //!< Unlicensed restricted state
#define NVML_GRID_LICENSE_STATE_UNLICENSED              4   //!< Unlicensed state
#define NVML_GRID_LICENSE_STATE_LICENSED                5   //!< Licensed state

typedef struct nvmlVgpuLicenseInfo_st
{
    unsigned char               isLicensed;     //!< License status
    nvmlVgpuLicenseExpiry_t     licenseExpiry;  //!< License expiry information
    unsigned int                currentState;   //!< Current license state
} nvmlVgpuLicenseInfo_t;

/**
 * Structure to store utilization value and process Id
 */
typedef struct nvmlProcessUtilizationSample_st
{
    unsigned int        pid;            //!< PID of process
    unsigned long long  timeStamp;      //!< CPU Timestamp in microseconds
    unsigned int        smUtil;         //!< SM (3D/Compute) Util Value
    unsigned int        memUtil;        //!< Frame Buffer Memory Util Value
    unsigned int        encUtil;        //!< Encoder Util Value
    unsigned int        decUtil;        //!< Decoder Util Value
} nvmlProcessUtilizationSample_t;

/**
 * Structure to store license expiry date and time values
 */
typedef struct nvmlGridLicenseExpiry_st
{
    unsigned int   year;        //!< Year value of license expiry
    unsigned short month;       //!< Month value of license expiry
    unsigned short day;         //!< Day value of license expiry
    unsigned short hour;        //!< Hour value of license expiry
    unsigned short min;         //!< Minutes value of license expiry
    unsigned short sec;         //!< Seconds value of license expiry
    unsigned char  status;      //!< License expiry status
} nvmlGridLicenseExpiry_t;

/**
 * Structure containing vGPU software licensable feature information
 */
typedef struct nvmlGridLicensableFeature_st
{
    nvmlGridLicenseFeatureCode_t    featureCode;                                 //!< Licensed feature code
    unsigned int                    featureState;                                //!< Non-zero if feature is currently licensed, otherwise zero
    char                            licenseInfo[NVML_GRID_LICENSE_BUFFER_SIZE];  //!< Deprecated.
    char                            productName[NVML_GRID_LICENSE_BUFFER_SIZE];  //!< Product name of feature
    unsigned int                    featureEnabled;                              //!< Non-zero if feature is enabled, otherwise zero
    nvmlGridLicenseExpiry_t         licenseExpiry;                               //!< License expiry structure containing date and time
} nvmlGridLicensableFeature_t;

/**
 * Structure to store vGPU software licensable features
 */
typedef struct nvmlGridLicensableFeatures_st
{
    int                         isGridLicenseSupported;                                       //!< Non-zero if vGPU Software Licensing is supported on the system, otherwise zero
    unsigned int                licensableFeaturesCount;                                      //!< Entries returned in \a gridLicensableFeatures array
    nvmlGridLicensableFeature_t gridLicensableFeatures[NVML_GRID_LICENSE_FEATURE_MAX_COUNT];  //!< Array of vGPU software licensable features.
} nvmlGridLicensableFeatures_t;

/**
 * Simplified chip architecture
 */
#define NVML_DEVICE_ARCH_KEPLER    2 // Devices based on the NVIDIA Kepler architecture
#define NVML_DEVICE_ARCH_MAXWELL   3 // Devices based on the NVIDIA Maxwell architecture
#define NVML_DEVICE_ARCH_PASCAL    4 // Devices based on the NVIDIA Pascal architecture
#define NVML_DEVICE_ARCH_VOLTA     5 // Devices based on the NVIDIA Volta architecture
#define NVML_DEVICE_ARCH_TURING    6 // Devices based on the NVIDIA Turing architecture

#define NVML_DEVICE_ARCH_AMPERE    7 // Devices based on the NVIDIA Ampere architecture

#define NVML_DEVICE_ARCH_UNKNOWN   0xffffffff // Anything else, presumably something newer

typedef unsigned int nvmlDeviceArchitecture_t;

/**
 * PCI bus types
 */
#define NVML_BUS_TYPE_UNKNOWN  0
#define NVML_BUS_TYPE_PCI      1
#define NVML_BUS_TYPE_PCIE     2
#define NVML_BUS_TYPE_FPCI     3
#define NVML_BUS_TYPE_AGP      4

typedef unsigned int nvmlBusType_t;

/**
 * Device Power Source
 */
#define NVML_POWER_SOURCE_AC      0x00000000
#define NVML_POWER_SOURCE_BATTERY 0x00000001

typedef unsigned int nvmlPowerSource_t;

/*
 * Device PCIE link Max Speed
 */
#define NVML_PCIE_LINK_MAX_SPEED_INVALID   0x00000000
#define NVML_PCIE_LINK_MAX_SPEED_2500MBPS  0x00000001
#define NVML_PCIE_LINK_MAX_SPEED_5000MBPS  0x00000002
#define NVML_PCIE_LINK_MAX_SPEED_8000MBPS  0x00000003
#define NVML_PCIE_LINK_MAX_SPEED_16000MBPS 0x00000004
#define NVML_PCIE_LINK_MAX_SPEED_32000MBPS 0x00000005

/*
 * Adaptive clocking status
 */
#define NVML_ADAPTIVE_CLOCKING_INFO_STATUS_DISABLED 0x00000000
#define NVML_ADAPTIVE_CLOCKING_INFO_STATUS_ENABLED  0x00000001

/** @} */
/** @} */

/***************************************************************************************************/
/** @defgroup nvmlFieldValueEnums Field Value Enums
 *  @{
 */
/***************************************************************************************************/

/**
 * Field Identifiers.
 *
 * All Identifiers pertain to a device. Each ID is only used once and is guaranteed never to change.
 */
#define NVML_FI_DEV_ECC_CURRENT           1   //!< Current ECC mode. 1=Active. 0=Inactive
#define NVML_FI_DEV_ECC_PENDING           2   //!< Pending ECC mode. 1=Active. 0=Inactive
/* ECC Count Totals */
#define NVML_FI_DEV_ECC_SBE_VOL_TOTAL     3   //!< Total single bit volatile ECC errors
#define NVML_FI_DEV_ECC_DBE_VOL_TOTAL     4   //!< Total double bit volatile ECC errors
#define NVML_FI_DEV_ECC_SBE_AGG_TOTAL     5   //!< Total single bit aggregate (persistent) ECC errors
#define NVML_FI_DEV_ECC_DBE_AGG_TOTAL     6   //!< Total double bit aggregate (persistent) ECC errors
/* Individual ECC locations */
#define NVML_FI_DEV_ECC_SBE_VOL_L1        7   //!< L1 cache single bit volatile ECC errors
#define NVML_FI_DEV_ECC_DBE_VOL_L1        8   //!< L1 cache double bit volatile ECC errors
#define NVML_FI_DEV_ECC_SBE_VOL_L2        9   //!< L2 cache single bit volatile ECC errors
#define NVML_FI_DEV_ECC_DBE_VOL_L2        10  //!< L2 cache double bit volatile ECC errors
#define NVML_FI_DEV_ECC_SBE_VOL_DEV       11  //!< Device memory single bit volatile ECC errors
#define NVML_FI_DEV_ECC_DBE_VOL_DEV       12  //!< Device memory double bit volatile ECC errors
#define NVML_FI_DEV_ECC_SBE_VOL_REG       13  //!< Register file single bit volatile ECC errors
#define NVML_FI_DEV_ECC_DBE_VOL_REG       14  //!< Register file double bit volatile ECC errors
#define NVML_FI_DEV_ECC_SBE_VOL_TEX       15  //!< Texture memory single bit volatile ECC errors
#define NVML_FI_DEV_ECC_DBE_VOL_TEX       16  //!< Texture memory double bit volatile ECC errors
#define NVML_FI_DEV_ECC_DBE_VOL_CBU       17  //!< CBU double bit volatile ECC errors
#define NVML_FI_DEV_ECC_SBE_AGG_L1        18  //!< L1 cache single bit aggregate (persistent) ECC errors
#define NVML_FI_DEV_ECC_DBE_AGG_L1        19  //!< L1 cache double bit aggregate (persistent) ECC errors
#define NVML_FI_DEV_ECC_SBE_AGG_L2        20  //!< L2 cache single bit aggregate (persistent) ECC errors
#define NVML_FI_DEV_ECC_DBE_AGG_L2        21  //!< L2 cache double bit aggregate (persistent) ECC errors
#define NVML_FI_DEV_ECC_SBE_AGG_DEV       22  //!< Device memory single bit aggregate (persistent) ECC errors
#define NVML_FI_DEV_ECC_DBE_AGG_DEV       23  //!< Device memory double bit aggregate (persistent) ECC errors
#define NVML_FI_DEV_ECC_SBE_AGG_REG       24  //!< Register File single bit aggregate (persistent) ECC errors
#define NVML_FI_DEV_ECC_DBE_AGG_REG       25  //!< Register File double bit aggregate (persistent) ECC errors
#define NVML_FI_DEV_ECC_SBE_AGG_TEX       26  //!< Texture memory single bit aggregate (persistent) ECC errors
#define NVML_FI_DEV_ECC_DBE_AGG_TEX       27  //!< Texture memory double bit aggregate (persistent) ECC errors
#define NVML_FI_DEV_ECC_DBE_AGG_CBU       28  //!< CBU double bit aggregate ECC errors

/* Page Retirement */
#define NVML_FI_DEV_RETIRED_SBE           29  //!< Number of retired pages because of single bit errors
#define NVML_FI_DEV_RETIRED_DBE           30  //!< Number of retired pages because of double bit errors
#define NVML_FI_DEV_RETIRED_PENDING       31  //!< If any pages are pending retirement. 1=yes. 0=no.

/* NvLink Flit Error Counters */
#define NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L0    32 //!< NVLink flow control CRC  Error Counter for Lane 0
#define NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L1    33 //!< NVLink flow control CRC  Error Counter for Lane 1
#define NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L2    34 //!< NVLink flow control CRC  Error Counter for Lane 2
#define NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L3    35 //!< NVLink flow control CRC  Error Counter for Lane 3
#define NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L4    36 //!< NVLink flow control CRC  Error Counter for Lane 4
#define NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L5    37 //!< NVLink flow control CRC  Error Counter for Lane 5
#define NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL 38 //!< NVLink flow control CRC  Error Counter total for all Lanes

/* NvLink CRC Data Error Counters */
#define NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L0    39 //!< NVLink data CRC Error Counter for Lane 0
#define NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L1    40 //!< NVLink data CRC Error Counter for Lane 1
#define NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L2    41 //!< NVLink data CRC Error Counter for Lane 2
#define NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L3    42 //!< NVLink data CRC Error Counter for Lane 3
#define NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L4    43 //!< NVLink data CRC Error Counter for Lane 4
#define NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L5    44 //!< NVLink data CRC Error Counter for Lane 5
#define NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL 45 //!< NvLink data CRC Error Counter total for all Lanes

/* NvLink Replay Error Counters */
#define NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0      46 //!< NVLink Replay Error Counter for Lane 0
#define NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L1      47 //!< NVLink Replay Error Counter for Lane 1
#define NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L2      48 //!< NVLink Replay Error Counter for Lane 2
#define NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L3      49 //!< NVLink Replay Error Counter for Lane 3
#define NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L4      50 //!< NVLink Replay Error Counter for Lane 4
#define NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L5      51 //!< NVLink Replay Error Counter for Lane 5
#define NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL   52 //!< NVLink Replay Error Counter total for all Lanes

/* NvLink Recovery Error Counters */
#define NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L0    53 //!< NVLink Recovery Error Counter for Lane 0
#define NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L1    54 //!< NVLink Recovery Error Counter for Lane 1
#define NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L2    55 //!< NVLink Recovery Error Counter for Lane 2
#define NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L3    56 //!< NVLink Recovery Error Counter for Lane 3
#define NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L4    57 //!< NVLink Recovery Error Counter for Lane 4
#define NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L5    58 //!< NVLink Recovery Error Counter for Lane 5
#define NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL 59 //!< NVLink Recovery Error Counter total for all Lanes

/* NvLink Bandwidth Counters */
/*
 * NVML_FI_DEV_NVLINK_BANDWIDTH_* field values are now deprecated.
 * Please use the following field values instead:
 * NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX
 * NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX
 * NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_TX
 * NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_RX
 */
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L0     60 //!< NVLink Bandwidth Counter for Counter Set 0, Lane 0
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L1     61 //!< NVLink Bandwidth Counter for Counter Set 0, Lane 1
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L2     62 //!< NVLink Bandwidth Counter for Counter Set 0, Lane 2
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L3     63 //!< NVLink Bandwidth Counter for Counter Set 0, Lane 3
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L4     64 //!< NVLink Bandwidth Counter for Counter Set 0, Lane 4
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L5     65 //!< NVLink Bandwidth Counter for Counter Set 0, Lane 5
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C0_TOTAL  66 //!< NVLink Bandwidth Counter Total for Counter Set 0, All Lanes

/* NvLink Bandwidth Counters */
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L0     67 //!< NVLink Bandwidth Counter for Counter Set 1, Lane 0
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L1     68 //!< NVLink Bandwidth Counter for Counter Set 1, Lane 1
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L2     69 //!< NVLink Bandwidth Counter for Counter Set 1, Lane 2
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L3     70 //!< NVLink Bandwidth Counter for Counter Set 1, Lane 3
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L4     71 //!< NVLink Bandwidth Counter for Counter Set 1, Lane 4
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L5     72 //!< NVLink Bandwidth Counter for Counter Set 1, Lane 5
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C1_TOTAL  73 //!< NVLink Bandwidth Counter Total for Counter Set 1, All Lanes

/* NVML Perf Policy Counters */
#define NVML_FI_DEV_PERF_POLICY_POWER              74   //!< Perf Policy Counter for Power Policy
#define NVML_FI_DEV_PERF_POLICY_THERMAL            75   //!< Perf Policy Counter for Thermal Policy
#define NVML_FI_DEV_PERF_POLICY_SYNC_BOOST         76   //!< Perf Policy Counter for Sync boost Policy
#define NVML_FI_DEV_PERF_POLICY_BOARD_LIMIT        77   //!< Perf Policy Counter for Board Limit
#define NVML_FI_DEV_PERF_POLICY_LOW_UTILIZATION    78   //!< Perf Policy Counter for Low GPU Utilization Policy
#define NVML_FI_DEV_PERF_POLICY_RELIABILITY        79   //!< Perf Policy Counter for Reliability Policy
#define NVML_FI_DEV_PERF_POLICY_TOTAL_APP_CLOCKS   80   //!< Perf Policy Counter for Total App Clock Policy
#define NVML_FI_DEV_PERF_POLICY_TOTAL_BASE_CLOCKS  81   //!< Perf Policy Counter for Total Base Clocks Policy

/* Memory temperatures */
#define NVML_FI_DEV_MEMORY_TEMP  82 //!< Memory temperature for the device

/* Energy Counter */
#define NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION 83 //!< Total energy consumption for the GPU in mJ since the driver was last reloaded

/* NVLink Speed */
#define NVML_FI_DEV_NVLINK_SPEED_MBPS_L0     84  //!< NVLink Speed in MBps for Link 0
#define NVML_FI_DEV_NVLINK_SPEED_MBPS_L1     85  //!< NVLink Speed in MBps for Link 1
#define NVML_FI_DEV_NVLINK_SPEED_MBPS_L2     86  //!< NVLink Speed in MBps for Link 2
#define NVML_FI_DEV_NVLINK_SPEED_MBPS_L3     87  //!< NVLink Speed in MBps for Link 3
#define NVML_FI_DEV_NVLINK_SPEED_MBPS_L4     88  //!< NVLink Speed in MBps for Link 4
#define NVML_FI_DEV_NVLINK_SPEED_MBPS_L5     89  //!< NVLink Speed in MBps for Link 5
#define NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON 90  //!< Common NVLink Speed in MBps for active links

#define NVML_FI_DEV_NVLINK_LINK_COUNT        91  //!< Number of NVLinks present on the device

#define NVML_FI_DEV_RETIRED_PENDING_SBE      92  //!< If any pages are pending retirement due to SBE. 1=yes. 0=no.
#define NVML_FI_DEV_RETIRED_PENDING_DBE      93  //!< If any pages are pending retirement due to DBE. 1=yes. 0=no.

#define NVML_FI_DEV_PCIE_REPLAY_COUNTER             94  //!< PCIe replay counter
#define NVML_FI_DEV_PCIE_REPLAY_ROLLOVER_COUNTER    95  //!< PCIe replay rollover counter

/* NvLink Flit Error Counters */
#define NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L6     96 //!< NVLink flow control CRC  Error Counter for Lane 6
#define NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L7     97 //!< NVLink flow control CRC  Error Counter for Lane 7
#define NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L8     98 //!< NVLink flow control CRC  Error Counter for Lane 8
#define NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L9     99 //!< NVLink flow control CRC  Error Counter for Lane 9
#define NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L10   100 //!< NVLink flow control CRC  Error Counter for Lane 10
#define NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L11   101 //!< NVLink flow control CRC  Error Counter for Lane 11

/* NvLink CRC Data Error Counters */
#define NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L6    102 //!< NVLink data CRC Error Counter for Lane 6
#define NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L7    103 //!< NVLink data CRC Error Counter for Lane 7
#define NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L8    104 //!< NVLink data CRC Error Counter for Lane 8
#define NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L9    105 //!< NVLink data CRC Error Counter for Lane 9
#define NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L10   106 //!< NVLink data CRC Error Counter for Lane 10
#define NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L11   107 //!< NVLink data CRC Error Counter for Lane 11

/* NvLink Replay Error Counters */
#define NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L6      108 //!< NVLink Replay Error Counter for Lane 6
#define NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L7      109 //!< NVLink Replay Error Counter for Lane 7
#define NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L8      110 //!< NVLink Replay Error Counter for Lane 8
#define NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L9      111 //!< NVLink Replay Error Counter for Lane 9
#define NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L10     112 //!< NVLink Replay Error Counter for Lane 10
#define NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L11     113 //!< NVLink Replay Error Counter for Lane 11

/* NvLink Recovery Error Counters */
#define NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L6    114 //!< NVLink Recovery Error Counter for Lane 6
#define NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L7    115 //!< NVLink Recovery Error Counter for Lane 7
#define NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L8    116 //!< NVLink Recovery Error Counter for Lane 8
#define NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L9    117 //!< NVLink Recovery Error Counter for Lane 9
#define NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L10   118 //!< NVLink Recovery Error Counter for Lane 10
#define NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L11   119 //!< NVLink Recovery Error Counter for Lane 11

/* NvLink Bandwidth Counters */
/*
 * NVML_FI_DEV_NVLINK_BANDWIDTH_* field values are now deprecated.
 * Please use the following field values instead:
 * NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX
 * NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX
 * NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_TX
 * NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_RX
 */
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L6     120 //!< NVLink Bandwidth Counter for Counter Set 0, Lane 6
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L7     121 //!< NVLink Bandwidth Counter for Counter Set 0, Lane 7
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L8     122 //!< NVLink Bandwidth Counter for Counter Set 0, Lane 8
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L9     123 //!< NVLink Bandwidth Counter for Counter Set 0, Lane 9
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L10    124 //!< NVLink Bandwidth Counter for Counter Set 0, Lane 10
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L11    125 //!< NVLink Bandwidth Counter for Counter Set 0, Lane 11

/* NvLink Bandwidth Counters */
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L6     126 //!< NVLink Bandwidth Counter for Counter Set 1, Lane 6
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L7     127 //!< NVLink Bandwidth Counter for Counter Set 1, Lane 7
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L8     128 //!< NVLink Bandwidth Counter for Counter Set 1, Lane 8
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L9     129 //!< NVLink Bandwidth Counter for Counter Set 1, Lane 9
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L10    130 //!< NVLink Bandwidth Counter for Counter Set 1, Lane 10
#define NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L11    131 //!< NVLink Bandwidth Counter for Counter Set 1, Lane 11

/* NVLink Speed */
#define NVML_FI_DEV_NVLINK_SPEED_MBPS_L6     132  //!< NVLink Speed in MBps for Link 6
#define NVML_FI_DEV_NVLINK_SPEED_MBPS_L7     133  //!< NVLink Speed in MBps for Link 7
#define NVML_FI_DEV_NVLINK_SPEED_MBPS_L8     134  //!< NVLink Speed in MBps for Link 8
#define NVML_FI_DEV_NVLINK_SPEED_MBPS_L9     135  //!< NVLink Speed in MBps for Link 9
#define NVML_FI_DEV_NVLINK_SPEED_MBPS_L10    136  //!< NVLink Speed in MBps for Link 10
#define NVML_FI_DEV_NVLINK_SPEED_MBPS_L11    137  //!< NVLink Speed in MBps for Link 11

/**
 * NVLink throughput counters field values
 *
 * Link ID needs to be specified in the scopeId field in nvmlFieldValue_t.
 * A scopeId of UINT_MAX returns aggregate value summed up across all links
 * for the specified counter type in fieldId.
 */
#define NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX      138 //!< NVLink TX Data throughput in KiB
#define NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX      139 //!< NVLink RX Data throughput in KiB
#define NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_TX       140 //!< NVLink TX Data + protocol overhead in KiB
#define NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_RX       141 //!< NVLink RX Data + protocol overhead in KiB

/* Row Remapper */
#define NVML_FI_DEV_REMAPPED_COR        142 //!< Number of remapped rows due to correctable errors
#define NVML_FI_DEV_REMAPPED_UNC        143 //!< Number of remapped rows due to uncorrectable errors
#define NVML_FI_DEV_REMAPPED_PENDING    144 //!< If any rows are pending remapping. 1=yes 0=no
#define NVML_FI_DEV_REMAPPED_FAILURE    145 //!< If any rows failed to be remapped 1=yes 0=no

/**
 * Remote device NVLink ID
 *
 * Link ID needs to be specified in the scopeId field in nvmlFieldValue_t.
 */
#define NVML_FI_DEV_NVLINK_REMOTE_NVLINK_ID     146 //!< Remote device NVLink ID

/**
 * NVSwitch: connected NVLink count
 */
#define NVML_FI_DEV_NVSWITCH_CONNECTED_LINK_COUNT   147  //!< Number of NVLinks connected to NVSwitch

/* NvLink ECC Data Error Counters
 *
 * Lane ID needs to be specified in the scopeId field in nvmlFieldValue_t.
 *
 */
#define NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L0    148 //!< NVLink data ECC Error Counter for Link 0
#define NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L1    149 //!< NVLink data ECC Error Counter for Link 1
#define NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L2    150 //!< NVLink data ECC Error Counter for Link 2
#define NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L3    151 //!< NVLink data ECC Error Counter for Link 3
#define NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L4    152 //!< NVLink data ECC Error Counter for Link 4
#define NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L5    153 //!< NVLink data ECC Error Counter for Link 5
#define NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L6    154 //!< NVLink data ECC Error Counter for Link 6
#define NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L7    155 //!< NVLink data ECC Error Counter for Link 7
#define NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L8    156 //!< NVLink data ECC Error Counter for Link 8
#define NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L9    157 //!< NVLink data ECC Error Counter for Link 9
#define NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L10   158 //!< NVLink data ECC Error Counter for Link 10
#define NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L11   159 //!< NVLink data ECC Error Counter for Link 11
#define NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_TOTAL 160 //!< NvLink data ECC Error Counter total for all Links

#define NVML_FI_MAX 161 //!< One greater than the largest field ID defined above

/**
 * Information for a Field Value Sample
 */
typedef struct nvmlFieldValue_st
{
    unsigned int fieldId;       //!< ID of the NVML field to retrieve. This must be set before any call that uses this struct. See the constants starting with NVML_FI_ above.
    unsigned int scopeId;       //!< Scope ID can represent data used by NVML depending on fieldId's context. For example, for NVLink throughput counter data, scopeId can represent linkId.
    long long timestamp;        //!< CPU Timestamp of this value in microseconds since 1970
    long long latencyUsec;      //!< How long this field value took to update (in usec) within NVML. This may be averaged across several fields that are serviced by the same driver call.
    nvmlValueType_t valueType;  //!< Type of the value stored in value
    nvmlReturn_t nvmlReturn;    //!< Return code for retrieving this value. This must be checked before looking at value, as value is undefined if nvmlReturn != NVML_SUCCESS
    nvmlValue_t value;          //!< Value for this field. This is only valid if nvmlReturn == NVML_SUCCESS
} nvmlFieldValue_t;


/** @} */

/***************************************************************************************************/
/** @defgroup nvmlUnitStructs Unit Structs
 *  @{
 */
/***************************************************************************************************/

typedef struct nvmlUnit_st* nvmlUnit_t;

/**
 * Description of HWBC entry
 */
typedef struct nvmlHwbcEntry_st
{
    unsigned int hwbcId;
    char firmwareVersion[32];
} nvmlHwbcEntry_t;

/**
 * Fan state enum.
 */
typedef enum nvmlFanState_enum
{
    NVML_FAN_NORMAL       = 0,     //!< Fan is working properly
    NVML_FAN_FAILED       = 1      //!< Fan has failed
} nvmlFanState_t;

/**
 * Led color enum.
 */
typedef enum nvmlLedColor_enum
{
    NVML_LED_COLOR_GREEN       = 0,     //!< GREEN, indicates good health
    NVML_LED_COLOR_AMBER       = 1      //!< AMBER, indicates problem
} nvmlLedColor_t;


/**
 * LED states for an S-class unit.
 */
typedef struct nvmlLedState_st
{
    char cause[256];               //!< If amber, a text description of the cause
    nvmlLedColor_t color;          //!< GREEN or AMBER
} nvmlLedState_t;

/**
 * Static S-class unit info.
 */
typedef struct nvmlUnitInfo_st
{
    char name[96];                      //!< Product name
    char id[96];                        //!< Product identifier
    char serial[96];                    //!< Product serial number
    char firmwareVersion[96];           //!< Firmware version
} nvmlUnitInfo_t;

/**
 * Power usage information for an S-class unit.
 * The power supply state is a human readable string that equals "Normal" or contains
 * a combination of "Abnormal" plus one or more of the following:
 *
 *    - High voltage
 *    - Fan failure
 *    - Heatsink temperature
 *    - Current limit
 *    - Voltage below UV alarm threshold
 *    - Low-voltage
 *    - SI2C remote off command
 *    - MOD_DISABLE input
 *    - Short pin transition
*/
typedef struct nvmlPSUInfo_st
{
    char state[256];                 //!< The power supply state
    unsigned int current;            //!< PSU current (A)
    unsigned int voltage;            //!< PSU voltage (V)
    unsigned int power;              //!< PSU power draw (W)
} nvmlPSUInfo_t;

/**
 * Fan speed reading for a single fan in an S-class unit.
 */
typedef struct nvmlUnitFanInfo_st
{
    unsigned int speed;              //!< Fan speed (RPM)
    nvmlFanState_t state;            //!< Flag that indicates whether fan is working properly
} nvmlUnitFanInfo_t;

/**
 * Fan speed readings for an entire S-class unit.
 */
typedef struct nvmlUnitFanSpeeds_st
{
    nvmlUnitFanInfo_t fans[24];      //!< Fan speed data for each fan
    unsigned int count;              //!< Number of fans in unit
} nvmlUnitFanSpeeds_t;

/** @} */

/***************************************************************************************************/
/** @addtogroup nvmlEvents
 *  @{
 */
/***************************************************************************************************/

/**
 * Handle to an event set
 */
typedef struct nvmlEventSet_st* nvmlEventSet_t;

/** @defgroup nvmlEventType Event Types
 * @{
 * Event Types which user can be notified about.
 * See description of particular functions for details.
 *
 * See \ref nvmlDeviceRegisterEvents and \ref nvmlDeviceGetSupportedEventTypes to check which devices
 * support each event.
 *
 * Types can be combined with bitwise or operator '|' when passed to \ref nvmlDeviceRegisterEvents
 */
//! Event about single bit ECC errors
/**
 * \note A corrected texture memory error is not an ECC error, so it does not generate a single bit event
 */
#define nvmlEventTypeSingleBitEccError     0x0000000000000001LL

//! Event about double bit ECC errors
/**
 * \note An uncorrected texture memory error is not an ECC error, so it does not generate a double bit event
 */
#define nvmlEventTypeDoubleBitEccError     0x0000000000000002LL

//! Event about PState changes
/**
 *  \note On Fermi architecture PState changes are also an indicator that GPU is throttling down due to
 *  no work being executed on the GPU, power capping or thermal capping. In a typical situation,
 *  Fermi-based GPU should stay in P0 for the duration of the execution of the compute process.
 */
#define nvmlEventTypePState                0x0000000000000004LL

//! Event that Xid critical error occurred
#define nvmlEventTypeXidCriticalError      0x0000000000000008LL

//! Event about clock changes
/**
 * Kepler only
 */
#define nvmlEventTypeClock                 0x0000000000000010LL

//! Event about AC/Battery power source changes
#define nvmlEventTypePowerSourceChange     0x0000000000000080LL

//! Event about MIG configuration changes
#define nvmlEventMigConfigChange           0x0000000000000100LL

//! Mask with no events
#define nvmlEventTypeNone                  0x0000000000000000LL

//! Mask of all events
#define nvmlEventTypeAll (nvmlEventTypeNone    \
        | nvmlEventTypeSingleBitEccError       \
        | nvmlEventTypeDoubleBitEccError       \
        | nvmlEventTypePState                  \
        | nvmlEventTypeClock                   \
        | nvmlEventTypeXidCriticalError        \
        | nvmlEventTypePowerSourceChange       \
        | nvmlEventMigConfigChange             \
        )
/** @} */

/**
 * Information about occurred event
 */
typedef struct nvmlEventData_st
{
    nvmlDevice_t        device;             //!< Specific device where the event occurred
    unsigned long long  eventType;          //!< Information about what specific event occurred
    unsigned long long  eventData;          //!< Stores XID error for the device in the event of nvmlEventTypeXidCriticalError,
                                            //   eventData is 0 for any other event. eventData is set as 999 for unknown xid error.
    unsigned int        gpuInstanceId;      //!< If MIG is enabled and nvmlEventTypeXidCriticalError event is attributable to a GPU
                                            //   instance, stores a valid GPU instance ID. gpuInstanceId is set to 0xFFFFFFFF
                                            //   otherwise.
    unsigned int        computeInstanceId;  //!< If MIG is enabled and nvmlEventTypeXidCriticalError event is attributable to a
                                            //   compute instance, stores a valid compute instance ID. computeInstanceId is set to
                                            //   0xFFFFFFFF otherwise.
} nvmlEventData_t;

/** @} */

/***************************************************************************************************/
/** @addtogroup nvmlClocksThrottleReasons
 *  @{
 */
/***************************************************************************************************/

/** Nothing is running on the GPU and the clocks are dropping to Idle state
 * \note This limiter may be removed in a later release
 */
#define nvmlClocksThrottleReasonGpuIdle                   0x0000000000000001LL

/** GPU clocks are limited by current setting of applications clocks
 *
 * @see nvmlDeviceSetApplicationsClocks
 * @see nvmlDeviceGetApplicationsClock
 */
#define nvmlClocksThrottleReasonApplicationsClocksSetting 0x0000000000000002LL

/**
 * @deprecated Renamed to \ref nvmlClocksThrottleReasonApplicationsClocksSetting
 *             as the name describes the situation more accurately.
 */
#define nvmlClocksThrottleReasonUserDefinedClocks         nvmlClocksThrottleReasonApplicationsClocksSetting

/** SW Power Scaling algorithm is reducing the clocks below requested clocks
 *
 * @see nvmlDeviceGetPowerUsage
 * @see nvmlDeviceSetPowerManagementLimit
 * @see nvmlDeviceGetPowerManagementLimit
 */
#define nvmlClocksThrottleReasonSwPowerCap                0x0000000000000004LL

/** HW Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
 *
 * This is an indicator of:
 *   - temperature being too high
 *   - External Power Brake Assertion is triggered (e.g. by the system power supply)
 *   - Power draw is too high and Fast Trigger protection is reducing the clocks
 *   - May be also reported during PState or clock change
 *      - This behavior may be removed in a later release.
 *
 * @see nvmlDeviceGetTemperature
 * @see nvmlDeviceGetTemperatureThreshold
 * @see nvmlDeviceGetPowerUsage
 */
#define nvmlClocksThrottleReasonHwSlowdown                0x0000000000000008LL

/** Sync Boost
 *
 * This GPU has been added to a Sync boost group with nvidia-smi or DCGM in
 * order to maximize performance per watt. All GPUs in the sync boost group
 * will boost to the minimum possible clocks across the entire group. Look at
 * the throttle reasons for other GPUs in the system to see why those GPUs are
 * holding this one at lower clocks.
 *
 */
#define nvmlClocksThrottleReasonSyncBoost                 0x0000000000000010LL

/** SW Thermal Slowdown
 *
 * This is an indicator of one or more of the following:
 *  - Current GPU temperature above the GPU Max Operating Temperature
 *  - Current memory temperature above the Memory Max Operating Temperature
 *
 */
#define nvmlClocksThrottleReasonSwThermalSlowdown         0x0000000000000020LL

/** HW Thermal Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
 *
 * This is an indicator of:
 *   - temperature being too high
 *
 * @see nvmlDeviceGetTemperature
 * @see nvmlDeviceGetTemperatureThreshold
 * @see nvmlDeviceGetPowerUsage
 */
#define nvmlClocksThrottleReasonHwThermalSlowdown         0x0000000000000040LL

/** HW Power Brake Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
 *
 * This is an indicator of:
 *   - External Power Brake Assertion being triggered (e.g. by the system power supply)
 *
 * @see nvmlDeviceGetTemperature
 * @see nvmlDeviceGetTemperatureThreshold
 * @see nvmlDeviceGetPowerUsage
 */
#define nvmlClocksThrottleReasonHwPowerBrakeSlowdown      0x0000000000000080LL

/** GPU clocks are limited by current setting of Display clocks
 *
 * @see bug 1997531
 */
#define nvmlClocksThrottleReasonDisplayClockSetting       0x0000000000000100LL

/** Bit mask representing no clocks throttling
 *
 * Clocks are as high as possible.
 * */
#define nvmlClocksThrottleReasonNone                      0x0000000000000000LL

/** Bit mask representing all supported clocks throttling reasons
 * New reasons might be added to this list in the future
 */
#define nvmlClocksThrottleReasonAll (nvmlClocksThrottleReasonNone \
      | nvmlClocksThrottleReasonGpuIdle                           \
      | nvmlClocksThrottleReasonApplicationsClocksSetting         \
      | nvmlClocksThrottleReasonSwPowerCap                        \
      | nvmlClocksThrottleReasonHwSlowdown                        \
      | nvmlClocksThrottleReasonSyncBoost                         \
      | nvmlClocksThrottleReasonSwThermalSlowdown                 \
      | nvmlClocksThrottleReasonHwThermalSlowdown                 \
      | nvmlClocksThrottleReasonHwPowerBrakeSlowdown              \
      | nvmlClocksThrottleReasonDisplayClockSetting               \
)
/** @} */

/***************************************************************************************************/
/** @defgroup nvmlAccountingStats Accounting Statistics
 *  @{
 *
 *  Set of APIs designed to provide per process information about usage of GPU.
 *
 *  @note All accounting statistics and accounting mode live in nvidia driver and reset
 *        to default (Disabled) when driver unloads.
 *        It is advised to run with persistence mode enabled.
 *
 *  @note Enabling accounting mode has no negative impact on the GPU performance.
 */
/***************************************************************************************************/

/**
 * Describes accounting statistics of a process.
 */
typedef struct nvmlAccountingStats_st {
    unsigned int gpuUtilization;                //!< Percent of time over the process's lifetime during which one or more kernels was executing on the GPU.
                                                //! Utilization stats just like returned by \ref nvmlDeviceGetUtilizationRates but for the life time of a
                                                //! process (not just the last sample period).
                                                //! Set to NVML_VALUE_NOT_AVAILABLE if nvmlDeviceGetUtilizationRates is not supported

    unsigned int memoryUtilization;             //!< Percent of time over the process's lifetime during which global (device) memory was being read or written.
                                                //! Set to NVML_VALUE_NOT_AVAILABLE if nvmlDeviceGetUtilizationRates is not supported

    unsigned long long maxMemoryUsage;          //!< Maximum total memory in bytes that was ever allocated by the process.
                                                //! Set to NVML_VALUE_NOT_AVAILABLE if nvmlProcessInfo_t->usedGpuMemory is not supported


    unsigned long long time;                    //!< Amount of time in ms during which the compute context was active. The time is reported as 0 if
                                                //!< the process is not terminated

    unsigned long long startTime;               //!< CPU Timestamp in usec representing start time for the process

    unsigned int isRunning;                     //!< Flag to represent if the process is running (1 for running, 0 for terminated)

    unsigned int reserved[5];                   //!< Reserved for future use
} nvmlAccountingStats_t;

/** @} */

/***************************************************************************************************/
/** @defgroup nvmlEncoderStructs Encoder Structs
 *  @{
 */
/***************************************************************************************************/

/**
 * Represents type of encoder for capacity can be queried
 */
typedef enum nvmlEncoderQueryType_enum
{
    NVML_ENCODER_QUERY_H264 = 0,        //!< H264 encoder
    NVML_ENCODER_QUERY_HEVC = 1         //!< HEVC encoder
}nvmlEncoderType_t;

/**
 * Structure to hold encoder session data
 */
typedef struct nvmlEncoderSessionInfo_st
{
    unsigned int       sessionId;       //!< Unique session ID
    unsigned int       pid;             //!< Owning process ID
    nvmlVgpuInstance_t vgpuInstance;    //!< Owning vGPU instance ID (only valid on vGPU hosts, otherwise zero)
    nvmlEncoderType_t  codecType;       //!< Video encoder type
    unsigned int       hResolution;     //!< Current encode horizontal resolution
    unsigned int       vResolution;     //!< Current encode vertical resolution
    unsigned int       averageFps;      //!< Moving average encode frames per second
    unsigned int       averageLatency;  //!< Moving average encode latency in microseconds
}nvmlEncoderSessionInfo_t;

/** @} */

/***************************************************************************************************/
/** @defgroup nvmlFBCStructs Frame Buffer Capture Structures
*  @{
*/
/***************************************************************************************************/

/**
 * Represents frame buffer capture session type
 */
typedef enum nvmlFBCSessionType_enum
{
    NVML_FBC_SESSION_TYPE_UNKNOWN = 0,     //!< Unknwon
    NVML_FBC_SESSION_TYPE_TOSYS,           //!< ToSys
    NVML_FBC_SESSION_TYPE_CUDA,            //!< Cuda
    NVML_FBC_SESSION_TYPE_VID,             //!< Vid
    NVML_FBC_SESSION_TYPE_HWENC            //!< HEnc
} nvmlFBCSessionType_t;

/**
 * Structure to hold frame buffer capture sessions stats
 */
typedef struct nvmlFBCStats_st
{
    unsigned int      sessionsCount;    //!< Total no of sessions
    unsigned int      averageFPS;       //!< Moving average new frames captured per second
    unsigned int      averageLatency;   //!< Moving average new frame capture latency in microseconds
} nvmlFBCStats_t;

#define NVML_NVFBC_SESSION_FLAG_DIFFMAP_ENABLED                0x00000001    //!< Bit specifying differential map state.
#define NVML_NVFBC_SESSION_FLAG_CLASSIFICATIONMAP_ENABLED      0x00000002    //!< Bit specifying classification map state.
#define NVML_NVFBC_SESSION_FLAG_CAPTURE_WITH_WAIT_NO_WAIT      0x00000004    //!< Bit specifying if capture was requested as non-blocking call.
#define NVML_NVFBC_SESSION_FLAG_CAPTURE_WITH_WAIT_INFINITE     0x00000008    //!< Bit specifying if capture was requested as blocking call.
#define NVML_NVFBC_SESSION_FLAG_CAPTURE_WITH_WAIT_TIMEOUT      0x00000010    //!< Bit specifying if capture was requested as blocking call with timeout period.

/**
 * Structure to hold FBC session data
 */
typedef struct nvmlFBCSessionInfo_st
{
    unsigned int          sessionId;                           //!< Unique session ID
    unsigned int          pid;                                 //!< Owning process ID
    nvmlVgpuInstance_t    vgpuInstance;                        //!< Owning vGPU instance ID (only valid on vGPU hosts, otherwise zero)
    unsigned int          displayOrdinal;                      //!< Display identifier
    nvmlFBCSessionType_t  sessionType;                         //!< Type of frame buffer capture session
    unsigned int          sessionFlags;                        //!< Session flags (one or more of NVML_NVFBC_SESSION_FLAG_XXX).
    unsigned int          hMaxResolution;                      //!< Max horizontal resolution supported by the capture session
    unsigned int          vMaxResolution;                      //!< Max vertical resolution supported by the capture session
    unsigned int          hResolution;                         //!< Horizontal resolution requested by caller in capture call
    unsigned int          vResolution;                         //!< Vertical resolution requested by caller in capture call
    unsigned int          averageFPS;                          //!< Moving average new frames captured per second
    unsigned int          averageLatency;                      //!< Moving average new frame capture latency in microseconds
} nvmlFBCSessionInfo_t;

/** @} */

/***************************************************************************************************/
/** @defgroup nvmlDrainDefs definitions related to the drain state
 *  @{
 */
/***************************************************************************************************/

/**
 *  Is the GPU device to be removed from the kernel by nvmlDeviceRemoveGpu()
 */
typedef enum nvmlDetachGpuState_enum
{
    NVML_DETACH_GPU_KEEP         = 0,
    NVML_DETACH_GPU_REMOVE
} nvmlDetachGpuState_t;

/**
 *  Parent bridge PCIe link state requested by nvmlDeviceRemoveGpu()
 */
typedef enum nvmlPcieLinkState_enum
{
    NVML_PCIE_LINK_KEEP         = 0,
    NVML_PCIE_LINK_SHUT_DOWN
} nvmlPcieLinkState_t;

/** @} */

/***************************************************************************************************/
/** @defgroup nvmlInitializationAndCleanup Initialization and Cleanup
 * This chapter describes the methods that handle NVML initialization and cleanup.
 * It is the user's responsibility to call \ref nvmlInit_v2() before calling any other methods, and
 * nvmlShutdown() once NVML is no longer being used.
 *  @{
 */
/***************************************************************************************************/

#define NVML_INIT_FLAG_NO_GPUS      1   //!< Don't fail nvmlInit() when no GPUs are found
#define NVML_INIT_FLAG_NO_ATTACH    2   //!< Don't attach GPUs

/**
 * Initialize NVML, but don't initialize any GPUs yet.
 *
 * \note nvmlInit_v3 introduces a "flags" argument, that allows passing boolean values
 *       modifying the behaviour of nvmlInit().
 * \note In NVML 5.319 new nvmlInit_v2 has replaced nvmlInit"_v1" (default in NVML 4.304 and older) that
 *       did initialize all GPU devices in the system.
 *
 * This allows NVML to communicate with a GPU
 * when other GPUs in the system are unstable or in a bad state.  When using this API, GPUs are
 * discovered and initialized in nvmlDeviceGetHandleBy* functions instead.
 *
 * \note To contrast nvmlInit_v2 with nvmlInit"_v1", NVML 4.304 nvmlInit"_v1" will fail when any detected GPU is in
 *       a bad or unstable state.
 *
 * For all products.
 *
 * This method, should be called once before invoking any other methods in the library.
 * A reference count of the number of initializations is maintained.  Shutdown only occurs
 * when the reference count reaches zero.
 *
 * @return
 *         - \ref NVML_SUCCESS                   if NVML has been properly initialized
 *         - \ref NVML_ERROR_DRIVER_NOT_LOADED   if NVIDIA driver is not running
 *         - \ref NVML_ERROR_NO_PERMISSION       if NVML does not have permission to talk to the driver
 *         - \ref NVML_ERROR_UNKNOWN             on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlInit_v2(void);

/**
 * nvmlInitWithFlags is a variant of nvmlInit(), that allows passing a set of boolean values
 *       modifying the behaviour of nvmlInit().
 *       Other than the "flags" parameter it is completely similar to \ref nvmlInit_v2.
 *
 * For all products.
 *
 * @param flags                                 behaviour modifier flags
 *
 * @return
 *         - \ref NVML_SUCCESS                   if NVML has been properly initialized
 *         - \ref NVML_ERROR_DRIVER_NOT_LOADED   if NVIDIA driver is not running
 *         - \ref NVML_ERROR_NO_PERMISSION       if NVML does not have permission to talk to the driver
 *         - \ref NVML_ERROR_UNKNOWN             on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlInitWithFlags(unsigned int flags);

/**
 * Shut down NVML by releasing all GPU resources previously allocated with \ref nvmlInit_v2().
 *
 * For all products.
 *
 * This method should be called after NVML work is done, once for each call to \ref nvmlInit_v2()
 * A reference count of the number of initializations is maintained.  Shutdown only occurs
 * when the reference count reaches zero.  For backwards compatibility, no error is reported if
 * nvmlShutdown() is called more times than nvmlInit().
 *
 * @return
 *         - \ref NVML_SUCCESS                 if NVML has been properly shut down
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlShutdown(void);

/** @} */

/***************************************************************************************************/
/** @defgroup nvmlErrorReporting Error reporting
 * This chapter describes helper functions for error reporting routines.
 *  @{
 */
/***************************************************************************************************/

/**
 * Helper method for converting NVML error codes into readable strings.
 *
 * For all products.
 *
 * @param result                               NVML error code to convert
 *
 * @return String representation of the error.
 *
 */
const DECLDIR char* nvmlErrorString(nvmlReturn_t result);
/** @} */


/***************************************************************************************************/
/** @defgroup nvmlConstants Constants
 *  @{
 */
/***************************************************************************************************/

/**
 * Buffer size guaranteed to be large enough for \ref nvmlDeviceGetInforomVersion and \ref nvmlDeviceGetInforomImageVersion
 */
#define NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE       16

/**
 * Buffer size guaranteed to be large enough for storing GPU identifiers.
 */
#define NVML_DEVICE_UUID_BUFFER_SIZE                  80

/**
 * Buffer size guaranteed to be large enough for \ref nvmlDeviceGetUUID
 */
#define NVML_DEVICE_UUID_V2_BUFFER_SIZE               96

/**
 * Buffer size guaranteed to be large enough for \ref nvmlDeviceGetBoardPartNumber
 */
#define NVML_DEVICE_PART_NUMBER_BUFFER_SIZE           80

/**
 * Buffer size guaranteed to be large enough for \ref nvmlSystemGetDriverVersion
 */
#define NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE        80

/**
 * Buffer size guaranteed to be large enough for \ref nvmlSystemGetNVMLVersion
 */
#define NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE          80

/**
 * Buffer size guaranteed to be large enough for storing GPU device names.
 */
#define NVML_DEVICE_NAME_BUFFER_SIZE                  64

/**
 * Buffer size guaranteed to be large enough for \ref nvmlDeviceGetName
 */
#define NVML_DEVICE_NAME_V2_BUFFER_SIZE               96

/**
 * Buffer size guaranteed to be large enough for \ref nvmlDeviceGetSerial
 */
#define NVML_DEVICE_SERIAL_BUFFER_SIZE                30

/**
 * Buffer size guaranteed to be large enough for \ref nvmlDeviceGetVbiosVersion
 */
#define NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE         32

/** @} */

/***************************************************************************************************/
/** @defgroup nvmlSystemQueries System Queries
 * This chapter describes the queries that NVML can perform against the local system. These queries
 * are not device-specific.
 *  @{
 */
/***************************************************************************************************/

/**
 * Retrieves the version of the system's graphics driver.
 *
 * For all products.
 *
 * The version identifier is an alphanumeric string.  It will not exceed 80 characters in length
 * (including the NULL terminator).  See \ref nvmlConstants::NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE.
 *
 * @param version                              Reference in which to return the version identifier
 * @param length                               The maximum allowed length of the string returned in \a version
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a version has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a version is NULL
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
 */
nvmlReturn_t DECLDIR nvmlSystemGetDriverVersion(char *version, unsigned int length);

/**
 * Retrieves the version of the NVML library.
 *
 * For all products.
 *
 * The version identifier is an alphanumeric string.  It will not exceed 80 characters in length
 * (including the NULL terminator).  See \ref nvmlConstants::NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE.
 *
 * @param version                              Reference in which to return the version identifier
 * @param length                               The maximum allowed length of the string returned in \a version
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a version has been set
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a version is NULL
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
 */
nvmlReturn_t DECLDIR nvmlSystemGetNVMLVersion(char *version, unsigned int length);

/**
 * Retrieves the version of the CUDA driver.
 *
 * For all products.
 *
 * The CUDA driver version returned will be retreived from the currently installed version of CUDA.
 * If the cuda library is not found, this function will return a known supported version number.
 *
 * @param cudaDriverVersion                    Reference in which to return the version identifier
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a cudaDriverVersion has been set
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a cudaDriverVersion is NULL
 */
nvmlReturn_t DECLDIR nvmlSystemGetCudaDriverVersion(int *cudaDriverVersion);

/**
 * Retrieves the version of the CUDA driver from the shared library.
 *
 * For all products.
 *
 * The returned CUDA driver version by calling cuDriverGetVersion()
 *
 * @param cudaDriverVersion                    Reference in which to return the version identifier
 *
 * @return
 *         - \ref NVML_SUCCESS                  if \a cudaDriverVersion has been set
 *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a cudaDriverVersion is NULL
 *         - \ref NVML_ERROR_LIBRARY_NOT_FOUND  if \a libcuda.so.1 or libcuda.dll is not found
 *         - \ref NVML_ERROR_FUNCTION_NOT_FOUND if \a cuDriverGetVersion() is not found in the shared library
 */
nvmlReturn_t DECLDIR nvmlSystemGetCudaDriverVersion_v2(int *cudaDriverVersion);

/**
 * Macros for converting the CUDA driver version number to Major and Minor version numbers.
 */
#define NVML_CUDA_DRIVER_VERSION_MAJOR(v) ((v)/1000)
#define NVML_CUDA_DRIVER_VERSION_MINOR(v) (((v)%1000)/10)

/**
 * Gets name of the process with provided process id
 *
 * For all products.
 *
 * Returned process name is cropped to provided length.
 * name string is encoded in ANSI.
 *
 * @param pid                                  The identifier of the process
 * @param name                                 Reference in which to return the process name
 * @param length                               The maximum allowed length of the string returned in \a name
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a name has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a name is NULL or \a length is 0.
 *         - \ref NVML_ERROR_NOT_FOUND         if process doesn't exists
 *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlSystemGetProcessName(unsigned int pid, char *name, unsigned int length);

/** @} */

/***************************************************************************************************/
/** @defgroup nvmlUnitQueries Unit Queries
 * This chapter describes that queries that NVML can perform against each unit. For S-class systems only.
 * In each case the device is identified with an nvmlUnit_t handle. This handle is obtained by
 * calling \ref nvmlUnitGetHandleByIndex().
 *  @{
 */
/***************************************************************************************************/

 /**
 * Retrieves the number of units in the system.
 *
 * For S-class products.
 *
 * @param unitCount                            Reference in which to return the number of units
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a unitCount has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unitCount is NULL
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlUnitGetCount(unsigned int *unitCount);

/**
 * Acquire the handle for a particular unit, based on its index.
 *
 * For S-class products.
 *
 * Valid indices are derived from the \a unitCount returned by \ref nvmlUnitGetCount().
 *   For example, if \a unitCount is 2 the valid indices are 0 and 1, corresponding to UNIT 0 and UNIT 1.
 *
 * The order in which NVML enumerates units has no guarantees of consistency between reboots.
 *
 * @param index                                The index of the target unit, >= 0 and < \a unitCount
 * @param unit                                 Reference in which to return the unit handle
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a unit has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a index is invalid or \a unit is NULL
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t *unit);

/**
 * Retrieves the static information associated with a unit.
 *
 * For S-class products.
 *
 * See \ref nvmlUnitInfo_t for details on available unit info.
 *
 * @param unit                                 The identifier of the target unit
 * @param info                                 Reference in which to return the unit information
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a info has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit is invalid or \a info is NULL
 */
nvmlReturn_t DECLDIR nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t *info);

/**
 * Retrieves the LED state associated with this unit.
 *
 * For S-class products.
 *
 * See \ref nvmlLedState_t for details on allowed states.
 *
 * @param unit                                 The identifier of the target unit
 * @param state                                Reference in which to return the current LED state
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a state has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit is invalid or \a state is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this is not an S-class product
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlUnitSetLedState()
 */
nvmlReturn_t DECLDIR nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t *state);

/**
 * Retrieves the PSU stats for the unit.
 *
 * For S-class products.
 *
 * See \ref nvmlPSUInfo_t for details on available PSU info.
 *
 * @param unit                                 The identifier of the target unit
 * @param psu                                  Reference in which to return the PSU information
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a psu has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit is invalid or \a psu is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this is not an S-class product
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t *psu);

/**
 * Retrieves the temperature readings for the unit, in degrees C.
 *
 * For S-class products.
 *
 * Depending on the product, readings may be available for intake (type=0),
 * exhaust (type=1) and board (type=2).
 *
 * @param unit                                 The identifier of the target unit
 * @param type                                 The type of reading to take
 * @param temp                                 Reference in which to return the intake temperature
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a temp has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit or \a type is invalid or \a temp is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this is not an S-class product
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type, unsigned int *temp);

/**
 * Retrieves the fan speed readings for the unit.
 *
 * For S-class products.
 *
 * See \ref nvmlUnitFanSpeeds_t for details on available fan speed info.
 *
 * @param unit                                 The identifier of the target unit
 * @param fanSpeeds                            Reference in which to return the fan speed information
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a fanSpeeds has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit is invalid or \a fanSpeeds is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this is not an S-class product
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit, nvmlUnitFanSpeeds_t *fanSpeeds);

/**
 * Retrieves the set of GPU devices that are attached to the specified unit.
 *
 * For S-class products.
 *
 * The \a deviceCount argument is expected to be set to the size of the input \a devices array.
 *
 * @param unit                                 The identifier of the target unit
 * @param deviceCount                          Reference in which to provide the \a devices array size, and
 *                                             to return the number of attached GPU devices
 * @param devices                              Reference in which to return the references to the attached GPU devices
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a deviceCount and \a devices have been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a deviceCount indicates that the \a devices array is too small
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit is invalid, either of \a deviceCount or \a devices is NULL
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int *deviceCount, nvmlDevice_t *devices);

/**
 * Retrieves the IDs and firmware versions for any Host Interface Cards (HICs) in the system.
 *
 * For S-class products.
 *
 * The \a hwbcCount argument is expected to be set to the size of the input \a hwbcEntries array.
 * The HIC must be connected to an S-class system for it to be reported by this function.
 *
 * @param hwbcCount                            Size of hwbcEntries array
 * @param hwbcEntries                          Array holding information about hwbc
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a hwbcCount and \a hwbcEntries have been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if either \a hwbcCount or \a hwbcEntries is NULL
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a hwbcCount indicates that the \a hwbcEntries array is too small
 */
nvmlReturn_t DECLDIR nvmlSystemGetHicVersion(unsigned int *hwbcCount, nvmlHwbcEntry_t *hwbcEntries);
/** @} */

/***************************************************************************************************/
/** @defgroup nvmlDeviceQueries Device Queries
 * This chapter describes that queries that NVML can perform against each device.
 * In each case the device is identified with an nvmlDevice_t handle. This handle is obtained by
 * calling one of \ref nvmlDeviceGetHandleByIndex_v2(), \ref nvmlDeviceGetHandleBySerial(),
 * \ref nvmlDeviceGetHandleByPciBusId_v2(). or \ref nvmlDeviceGetHandleByUUID().
 *  @{
 */
/***************************************************************************************************/

 /**
 * Retrieves the number of compute devices in the system. A compute device is a single GPU.
 *
 * For all products.
 *
 * Note: New nvmlDeviceGetCount_v2 (default in NVML 5.319) returns count of all devices in the system
 *       even if nvmlDeviceGetHandleByIndex_v2 returns NVML_ERROR_NO_PERMISSION for such device.
 *       Update your code to handle this error, or use NVML 4.304 or older nvml header file.
 *       For backward binary compatibility reasons _v1 version of the API is still present in the shared
 *       library.
 *       Old _v1 version of nvmlDeviceGetCount doesn't count devices that NVML has no permission to talk to.
 *
 * @param deviceCount                          Reference in which to return the number of accessible devices
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a deviceCount has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a deviceCount is NULL
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetCount_v2(unsigned int *deviceCount);

/**
 * Get attributes (engine counts etc.) for the given NVML device handle.
 *
 * @note This API currently only supports MIG device handles.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               NVML device handle
 * @param attributes                           Device attributes
 *
 * @return
 *        - \ref NVML_SUCCESS                  if \a device attributes were successfully retrieved
 *        - \ref NVML_ERROR_INVALID_ARGUMENT   if \a device handle is invalid
 *        - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *        - \ref NVML_ERROR_NOT_SUPPORTED      if this query is not supported by the device
 *        - \ref NVML_ERROR_UNKNOWN            on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetAttributes_v2(nvmlDevice_t device, nvmlDeviceAttributes_t *attributes);

/**
 * Acquire the handle for a particular device, based on its index.
 *
 * For all products.
 *
 * Valid indices are derived from the \a accessibleDevices count returned by
 *   \ref nvmlDeviceGetCount_v2(). For example, if \a accessibleDevices is 2 the valid indices
 *   are 0 and 1, corresponding to GPU 0 and GPU 1.
 *
 * The order in which NVML enumerates devices has no guarantees of consistency between reboots. For that reason it
 *   is recommended that devices be looked up by their PCI ids or UUID. See
 *   \ref nvmlDeviceGetHandleByUUID() and \ref nvmlDeviceGetHandleByPciBusId_v2().
 *
 * Note: The NVML index may not correlate with other APIs, such as the CUDA device index.
 *
 * Starting from NVML 5, this API causes NVML to initialize the target GPU
 * NVML may initialize additional GPUs if:
 *  - The target GPU is an SLI slave
 *
 * Note: New nvmlDeviceGetCount_v2 (default in NVML 5.319) returns count of all devices in the system
 *       even if nvmlDeviceGetHandleByIndex_v2 returns NVML_ERROR_NO_PERMISSION for such device.
 *       Update your code to handle this error, or use NVML 4.304 or older nvml header file.
 *       For backward binary compatibility reasons _v1 version of the API is still present in the shared
 *       library.
 *       Old _v1 version of nvmlDeviceGetCount doesn't count devices that NVML has no permission to talk to.
 *
 *       This means that nvmlDeviceGetHandleByIndex_v2 and _v1 can return different devices for the same index.
 *       If you don't touch macros that map old (_v1) versions to _v2 versions at the top of the file you don't
 *       need to worry about that.
 *
 * @param index                                The index of the target GPU, >= 0 and < \a accessibleDevices
 * @param device                               Reference in which to return the device handle
 *
 * @return
 *         - \ref NVML_SUCCESS                  if \a device has been set
 *         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a index is invalid or \a device is NULL
 *         - \ref NVML_ERROR_INSUFFICIENT_POWER if any attached devices have improperly attached external power cables
 *         - \ref NVML_ERROR_NO_PERMISSION      if the user doesn't have permission to talk to this device
 *         - \ref NVML_ERROR_IRQ_ISSUE          if NVIDIA kernel detected an interrupt issue with the attached GPUs
 *         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
 *
 * @see nvmlDeviceGetIndex
 * @see nvmlDeviceGetCount
 */
nvmlReturn_t DECLDIR nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device);

/**
 * Acquire the handle for a particular device, based on its board serial number.
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * This number corresponds to the value printed directly on the board, and to the value returned by
 *   \ref nvmlDeviceGetSerial().
 *
 * @deprecated Since more than one GPU can exist on a single board this function is deprecated in favor
 *             of \ref nvmlDeviceGetHandleByUUID.
 *             For dual GPU boards this function will return NVML_ERROR_INVALID_ARGUMENT.
 *
 * Starting from NVML 5, this API causes NVML to initialize the target GPU
 * NVML may initialize additional GPUs as it searches for the target GPU
 *
 * @param serial                               The board serial number of the target GPU
 * @param device                               Reference in which to return the device handle
 *
 * @return
 *         - \ref NVML_SUCCESS                  if \a device has been set
 *         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a serial is invalid, \a device is NULL or more than one
 *                                              device has the same serial (dual GPU boards)
 *         - \ref NVML_ERROR_NOT_FOUND          if \a serial does not match a valid device on the system
 *         - \ref NVML_ERROR_INSUFFICIENT_POWER if any attached devices have improperly attached external power cables
 *         - \ref NVML_ERROR_IRQ_ISSUE          if NVIDIA kernel detected an interrupt issue with the attached GPUs
 *         - \ref NVML_ERROR_GPU_IS_LOST        if any GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
 *
 * @see nvmlDeviceGetSerial
 * @see nvmlDeviceGetHandleByUUID
 */
nvmlReturn_t DECLDIR nvmlDeviceGetHandleBySerial(const char *serial, nvmlDevice_t *device);

/**
 * Acquire the handle for a particular device, based on its globally unique immutable UUID associated with each device.
 *
 * For all products.
 *
 * @param uuid                                 The UUID of the target GPU or MIG instance
 * @param device                               Reference in which to return the device handle or MIG device handle
 *
 * Starting from NVML 5, this API causes NVML to initialize the target GPU
 * NVML may initialize additional GPUs as it searches for the target GPU
 *
 * @return
 *         - \ref NVML_SUCCESS                  if \a device has been set
 *         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a uuid is invalid or \a device is null
 *         - \ref NVML_ERROR_NOT_FOUND          if \a uuid does not match a valid device on the system
 *         - \ref NVML_ERROR_INSUFFICIENT_POWER if any attached devices have improperly attached external power cables
 *         - \ref NVML_ERROR_IRQ_ISSUE          if NVIDIA kernel detected an interrupt issue with the attached GPUs
 *         - \ref NVML_ERROR_GPU_IS_LOST        if any GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
 *
 * @see nvmlDeviceGetUUID
 */
nvmlReturn_t DECLDIR nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *device);

/**
 * Acquire the handle for a particular device, based on its PCI bus id.
 *
 * For all products.
 *
 * This value corresponds to the nvmlPciInfo_t::busId returned by \ref nvmlDeviceGetPciInfo_v3().
 *
 * Starting from NVML 5, this API causes NVML to initialize the target GPU
 * NVML may initialize additional GPUs if:
 *  - The target GPU is an SLI slave
 *
 * \note NVML 4.304 and older version of nvmlDeviceGetHandleByPciBusId"_v1" returns NVML_ERROR_NOT_FOUND
 *       instead of NVML_ERROR_NO_PERMISSION.
 *
 * @param pciBusId                             The PCI bus id of the target GPU
 * @param device                               Reference in which to return the device handle
 *
 * @return
 *         - \ref NVML_SUCCESS                  if \a device has been set
 *         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a pciBusId is invalid or \a device is NULL
 *         - \ref NVML_ERROR_NOT_FOUND          if \a pciBusId does not match a valid device on the system
 *         - \ref NVML_ERROR_INSUFFICIENT_POWER if the attached device has improperly attached external power cables
 *         - \ref NVML_ERROR_NO_PERMISSION      if the user doesn't have permission to talk to this device
 *         - \ref NVML_ERROR_IRQ_ISSUE          if NVIDIA kernel detected an interrupt issue with the attached GPUs
 *         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, nvmlDevice_t *device);

/**
 * Retrieves the name of this device.
 *
 * For all products.
 *
 * The name is an alphanumeric string that denotes a particular product, e.g. Tesla &tm; C2070. It will not
 * exceed 96 characters in length (including the NULL terminator).  See \ref
 * nvmlConstants::NVML_DEVICE_NAME_V2_BUFFER_SIZE.
 *
 * When used with MIG device handles the API returns MIG device names which can be used to identify devices
 * based on their attributes.
 *
 * @param device                               The identifier of the target device
 * @param name                                 Reference in which to return the product name
 * @param length                               The maximum allowed length of the string returned in \a name
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a name has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a name is NULL
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length);

/**
 * Retrieves the brand of this device.
 *
 * For all products.
 *
 * The type is a member of \ref nvmlBrandType_t defined above.
 *
 * @param device                               The identifier of the target device
 * @param type                                 Reference in which to return the product brand type
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a name has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a type is NULL
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t *type);

/**
 * Retrieves the NVML index of this device.
 *
 * For all products.
 *
 * Valid indices are derived from the \a accessibleDevices count returned by
 *   \ref nvmlDeviceGetCount_v2(). For example, if \a accessibleDevices is 2 the valid indices
 *   are 0 and 1, corresponding to GPU 0 and GPU 1.
 *
 * The order in which NVML enumerates devices has no guarantees of consistency between reboots. For that reason it
 *   is recommended that devices be looked up by their PCI ids or GPU UUID. See
 *   \ref nvmlDeviceGetHandleByPciBusId_v2() and \ref nvmlDeviceGetHandleByUUID().
 *
 * When used with MIG device handles this API returns indices that can be
 * passed to \ref nvmlDeviceGetMigDeviceHandleByIndex to retrieve an identical handle.
 * MIG device indices are unique within a device.
 *
 * Note: The NVML index may not correlate with other APIs, such as the CUDA device index.
 *
 * @param device                               The identifier of the target device
 * @param index                                Reference in which to return the NVML index of the device
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a index has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a index is NULL
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceGetHandleByIndex()
 * @see nvmlDeviceGetCount()
 */
nvmlReturn_t DECLDIR nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index);

/**
 * Retrieves the globally unique board serial number associated with this device's board.
 *
 * For all products with an inforom.
 *
 * The serial number is an alphanumeric string that will not exceed 30 characters (including the NULL terminator).
 * This number matches the serial number tag that is physically attached to the board.  See \ref
 * nvmlConstants::NVML_DEVICE_SERIAL_BUFFER_SIZE.
 *
 * @param device                               The identifier of the target device
 * @param serial                               Reference in which to return the board/module serial number
 * @param length                               The maximum allowed length of the string returned in \a serial
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a serial has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a serial is NULL
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetSerial(nvmlDevice_t device, char *serial, unsigned int length);


/***************************************************************************************************/

/** @defgroup nvmlAffinity CPU and Memory Affinity
 *  This chapter describes NVML operations that are associated with CPU and memory
 *  affinity.
 *  @{
 */
/***************************************************************************************************/

//! Scope of NUMA node for affinity queries
#define NVML_AFFINITY_SCOPE_NODE     0
//! Scope of processor socket for affinity queries
#define NVML_AFFINITY_SCOPE_SOCKET   1

typedef unsigned int nvmlAffinityScope_t;

/**
 * Retrieves an array of unsigned ints (sized to nodeSetSize) of bitmasks with
 * the ideal memory affinity within node or socket for the device.
 * For example, if NUMA node 0, 1 are ideal within the socket for the device and nodeSetSize ==  1,
 *     result[0] = 0x3
 *
 * \note If requested scope is not applicable to the target topology, the API
 *       will fall back to reporting the memory affinity for the immediate non-I/O
 *       ancestor of the device.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               The identifier of the target device
 * @param nodeSetSize                          The size of the nodeSet array that is safe to access
 * @param nodeSet                              Array reference in which to return a bitmask of NODEs, 64 NODEs per
 *                                             unsigned long on 64-bit machines, 32 on 32-bit machines
 * @param scope                                Scope that change the default behavior
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a NUMA node Affinity has been filled
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, nodeSetSize == 0, nodeSet is NULL or scope is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */

nvmlReturn_t DECLDIR nvmlDeviceGetMemoryAffinity(nvmlDevice_t device, unsigned int nodeSetSize, unsigned long *nodeSet, nvmlAffinityScope_t scope);

/**
 * Retrieves an array of unsigned ints (sized to cpuSetSize) of bitmasks with the
 * ideal CPU affinity within node or socket for the device.
 * For example, if processors 0, 1, 32, and 33 are ideal for the device and cpuSetSize == 2,
 *     result[0] = 0x3, result[1] = 0x3
 *
 * \note If requested scope is not applicable to the target topology, the API
 *       will fall back to reporting the CPU affinity for the immediate non-I/O
 *       ancestor of the device.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               The identifier of the target device
 * @param cpuSetSize                           The size of the cpuSet array that is safe to access
 * @param cpuSet                               Array reference in which to return a bitmask of CPUs, 64 CPUs per
 *                                                 unsigned long on 64-bit machines, 32 on 32-bit machines
 * @param scope                                Scope that change the default behavior
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a cpuAffinity has been filled
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, cpuSetSize == 0, cpuSet is NULL or sope is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */

nvmlReturn_t DECLDIR nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet, nvmlAffinityScope_t scope);

/**
 * Retrieves an array of unsigned ints (sized to cpuSetSize) of bitmasks with the ideal CPU affinity for the device
 * For example, if processors 0, 1, 32, and 33 are ideal for the device and cpuSetSize == 2,
 *     result[0] = 0x3, result[1] = 0x3
 * This is equivalent to calling \ref nvmlDeviceGetCpuAffinityWithinScope with \ref NVML_AFFINITY_SCOPE_NODE.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               The identifier of the target device
 * @param cpuSetSize                           The size of the cpuSet array that is safe to access
 * @param cpuSet                               Array reference in which to return a bitmask of CPUs, 64 CPUs per
 *                                                 unsigned long on 64-bit machines, 32 on 32-bit machines
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a cpuAffinity has been filled
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, cpuSetSize == 0, or cpuSet is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetCpuAffinity(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet);

/**
 * Sets the ideal affinity for the calling thread and device using the guidelines
 * given in nvmlDeviceGetCpuAffinity().  Note, this is a change as of version 8.0.
 * Older versions set the affinity for a calling process and all children.
 * Currently supports up to 1024 processors.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               The identifier of the target device
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the calling process has been successfully bound
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceSetCpuAffinity(nvmlDevice_t device);

/**
 * Clear all affinity bindings for the calling thread.  Note, this is a change as of version
 * 8.0 as older versions cleared the affinity for a calling process and all children.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               The identifier of the target device
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the calling process has been successfully unbound
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceClearCpuAffinity(nvmlDevice_t device);

/**
 * Retrieve the common ancestor for two devices
 * For all products.
 * Supported on Linux only.
 *
 * @param device1                              The identifier of the first device
 * @param device2                              The identifier of the second device
 * @param pathInfo                             A \ref nvmlGpuTopologyLevel_t that gives the path type
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a pathInfo has been set
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device1, or \a device2 is invalid, or \a pathInfo is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device or OS does not support this feature
 *         - \ref NVML_ERROR_UNKNOWN           an error has occurred in underlying topology discovery
 */

/** @} */
nvmlReturn_t DECLDIR nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t *pathInfo);

/**
 * Retrieve the set of GPUs that are nearest to a given device at a specific interconnectivity level
 * For all products.
 * Supported on Linux only.
 *
 * @param device                               The identifier of the first device
 * @param level                                The \ref nvmlGpuTopologyLevel_t level to search for other GPUs
 * @param count                                When zero, is set to the number of matching GPUs such that \a deviceArray
 *                                             can be malloc'd.  When non-zero, \a deviceArray will be filled with \a count
 *                                             number of device handles.
 * @param deviceArray                          An array of device handles for GPUs found at \a level
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a deviceArray or \a count (if initially zero) has been set
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a level, or \a count is invalid, or \a deviceArray is NULL with a non-zero \a count
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device or OS does not support this feature
 *         - \ref NVML_ERROR_UNKNOWN           an error has occurred in underlying topology discovery
 */
nvmlReturn_t DECLDIR nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int *count, nvmlDevice_t *deviceArray);

/**
 * Retrieve the set of GPUs that have a CPU affinity with the given CPU number
 * For all products.
 * Supported on Linux only.
 *
 * @param cpuNumber                            The CPU number
 * @param count                                When zero, is set to the number of matching GPUs such that \a deviceArray
 *                                             can be malloc'd.  When non-zero, \a deviceArray will be filled with \a count
 *                                             number of device handles.
 * @param deviceArray                          An array of device handles for GPUs found with affinity to \a cpuNumber
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a deviceArray or \a count (if initially zero) has been set
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a cpuNumber, or \a count is invalid, or \a deviceArray is NULL with a non-zero \a count
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device or OS does not support this feature
 *         - \ref NVML_ERROR_UNKNOWN           an error has occurred in underlying topology discovery
 */
nvmlReturn_t DECLDIR nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int *count, nvmlDevice_t *deviceArray);

/**
 * Retrieve the status for a given p2p capability index between a given pair of GPU
 *
 * @param device1                              The first device
 * @param device2                              The second device
 * @param p2pIndex                             p2p Capability Index being looked for between \a device1 and \a device2
 * @param p2pStatus                            Reference in which to return the status of the \a p2pIndex
 *                                             between \a device1 and \a device2
 * @return
 *         - \ref NVML_SUCCESS         if \a p2pStatus has been populated
 *         - \ref NVML_ERROR_INVALID_ARGUMENT     if \a device1 or \a device2 or \a p2pIndex is invalid or \a p2pStatus is NULL
 *         - \ref NVML_ERROR_UNKNOWN              on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex,nvmlGpuP2PStatus_t *p2pStatus);

/**
 * Retrieves the globally unique immutable UUID associated with this device, as a 5 part hexadecimal string,
 * that augments the immutable, board serial identifier.
 *
 * For all products.
 *
 * The UUID is a globally unique identifier. It is the only available identifier for pre-Fermi-architecture products.
 * It does NOT correspond to any identifier printed on the board.  It will not exceed 96 characters in length
 * (including the NULL terminator).  See \ref nvmlConstants::NVML_DEVICE_UUID_V2_BUFFER_SIZE.
 *
 * When used with MIG device handles the API returns globally unique UUIDs which can be used to identify MIG
 * devices across both GPU and MIG devices. UUIDs are immutable for the lifetime of a MIG device.
 *
 * @param device                               The identifier of the target device
 * @param uuid                                 Reference in which to return the GPU UUID
 * @param length                               The maximum allowed length of the string returned in \a uuid
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a uuid has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a uuid is NULL
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length);

/**
 * Retrieve the MDEV UUID of a vGPU instance.
 *
 * The MDEV UUID is a globally unique identifier of the mdev device assigned to the VM, and is returned as a 5-part hexadecimal string,
 * not exceeding 80 characters in length (including the NULL terminator).
 * MDEV UUID is displayed only on KVM platform.
 * See \ref nvmlConstants::NVML_DEVICE_UUID_BUFFER_SIZE.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             Identifier of the target vGPU instance
 * @param mdevUuid                 Pointer to caller-supplied buffer to hold MDEV UUID
 * @param size                     Size of buffer in bytes
 *
 * @return
 *         - \ref NVML_SUCCESS                 successful completion
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_NOT_SUPPORTED     on any hypervisor other than KVM
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a mdevUuid is NULL
 *         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a size is too small
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetMdevUUID(nvmlVgpuInstance_t vgpuInstance, char *mdevUuid, unsigned int size);

/**
 * Retrieves minor number for the device. The minor number for the device is such that the Nvidia device node file for
 * each GPU will have the form /dev/nvidia[minor number].
 *
 * For all products.
 * Supported only for Linux
 *
 * @param device                                The identifier of the target device
 * @param minorNumber                           Reference in which to return the minor number for the device
 * @return
 *         - \ref NVML_SUCCESS                 if the minor number is successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a minorNumber is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int *minorNumber);

/**
 * Retrieves the the device board part number which is programmed into the board's InfoROM
 *
 * For all products.
 *
 * @param device                                Identifier of the target device
 * @param partNumber                            Reference to the buffer to return
 * @param length                                Length of the buffer reference
 *
 * @return
 *         - \ref NVML_SUCCESS                  if \a partNumber has been set
 *         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref NVML_ERROR_NOT_SUPPORTED      if the needed VBIOS fields have not been filled
 *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a device is invalid or \a serial is NULL
 *         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char* partNumber, unsigned int length);

/**
 * Retrieves the version information for the device's infoROM object.
 *
 * For all products with an inforom.
 *
 * Fermi and higher parts have non-volatile on-board memory for persisting device info, such as aggregate
 * ECC counts. The version of the data structures in this memory may change from time to time. It will not
 * exceed 16 characters in length (including the NULL terminator).
 * See \ref nvmlConstants::NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE.
 *
 * See \ref nvmlInforomObject_t for details on the available infoROM objects.
 *
 * @param device                               The identifier of the target device
 * @param object                               The target infoROM object
 * @param version                              Reference in which to return the infoROM version
 * @param length                               The maximum allowed length of the string returned in \a version
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a version has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a version is NULL
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have an infoROM
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceGetInforomImageVersion
 */
nvmlReturn_t DECLDIR nvmlDeviceGetInforomVersion(nvmlDevice_t device, nvmlInforomObject_t object, char *version, unsigned int length);

/**
 * Retrieves the global infoROM image version
 *
 * For all products with an inforom.
 *
 * Image version just like VBIOS version uniquely describes the exact version of the infoROM flashed on the board
 * in contrast to infoROM object version which is only an indicator of supported features.
 * Version string will not exceed 16 characters in length (including the NULL terminator).
 * See \ref nvmlConstants::NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE.
 *
 * @param device                               The identifier of the target device
 * @param version                              Reference in which to return the infoROM image version
 * @param length                               The maximum allowed length of the string returned in \a version
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a version has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a version is NULL
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have an infoROM
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceGetInforomVersion
 */
nvmlReturn_t DECLDIR nvmlDeviceGetInforomImageVersion(nvmlDevice_t device, char *version, unsigned int length);

/**
 * Retrieves the checksum of the configuration stored in the device's infoROM.
 *
 * For all products with an inforom.
 *
 * Can be used to make sure that two GPUs have the exact same configuration.
 * Current checksum takes into account configuration stored in PWR and ECC infoROM objects.
 * Checksum can change between driver releases or when user changes configuration (e.g. disable/enable ECC)
 *
 * @param device                               The identifier of the target device
 * @param checksum                             Reference in which to return the infoROM configuration checksum
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a checksum has been set
 *         - \ref NVML_ERROR_CORRUPTED_INFOROM if the device's checksum couldn't be retrieved due to infoROM corruption
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a checksum is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetInforomConfigurationChecksum(nvmlDevice_t device, unsigned int *checksum);

/**
 * Reads the infoROM from the flash and verifies the checksums.
 *
 * For all products with an inforom.
 *
 * @param device                               The identifier of the target device
 *
 * @return
 *         - \ref NVML_SUCCESS                 if infoROM is not corrupted
 *         - \ref NVML_ERROR_CORRUPTED_INFOROM if the device's infoROM is corrupted
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceValidateInforom(nvmlDevice_t device);

/**
 * Retrieves the display mode for the device.
 *
 * For all products.
 *
 * This method indicates whether a physical display (e.g. monitor) is currently connected to
 * any of the device's connectors.
 *
 * See \ref nvmlEnableState_t for details on allowed modes.
 *
 * @param device                               The identifier of the target device
 * @param display                              Reference in which to return the display mode
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a display has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a display is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t *display);

/**
 * Retrieves the display active state for the device.
 *
 * For all products.
 *
 * This method indicates whether a display is initialized on the device.
 * For example whether X Server is attached to this device and has allocated memory for the screen.
 *
 * Display can be active even when no monitor is physically attached.
 *
 * See \ref nvmlEnableState_t for details on allowed modes.
 *
 * @param device                               The identifier of the target device
 * @param isActive                             Reference in which to return the display active state
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a isActive has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a isActive is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t *isActive);

/**
 * Retrieves the persistence mode associated with this device.
 *
 * For all products.
 * For Linux only.
 *
 * When driver persistence mode is enabled the driver software state is not torn down when the last
 * client disconnects. By default this feature is disabled.
 *
 * See \ref nvmlEnableState_t for details on allowed modes.
 *
 * @param device                               The identifier of the target device
 * @param mode                                 Reference in which to return the current driver persistence mode
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a mode has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceSetPersistenceMode()
 */
nvmlReturn_t DECLDIR nvmlDeviceGetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t *mode);

/**
 * Retrieves the PCI attributes of this device.
 *
 * For all products.
 *
 * See \ref nvmlPciInfo_t for details on the available PCI info.
 *
 * @param device                               The identifier of the target device
 * @param pci                                  Reference in which to return the PCI info
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a pci has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a pci is NULL
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci);

/**
 * Retrieves the maximum PCIe link generation possible with this device and system
 *
 * I.E. for a generation 2 PCIe device attached to a generation 1 PCIe bus the max link generation this function will
 * report is generation 1.
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param maxLinkGen                           Reference in which to return the max PCIe link generation
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a maxLinkGen has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a maxLinkGen is null
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if PCIe link information is not available
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int *maxLinkGen);

/**
 * Retrieves the maximum PCIe link width possible with this device and system
 *
 * I.E. for a device with a 16x PCIe bus width attached to a 8x PCIe system bus this function will report
 * a max link width of 8.
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param maxLinkWidth                         Reference in which to return the max PCIe link generation
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a maxLinkWidth has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a maxLinkWidth is null
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if PCIe link information is not available
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int *maxLinkWidth);

/**
 * Retrieves the current PCIe link generation
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param currLinkGen                          Reference in which to return the current PCIe link generation
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a currLinkGen has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a currLinkGen is null
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if PCIe link information is not available
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int *currLinkGen);

/**
 * Retrieves the current PCIe link width
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param currLinkWidth                        Reference in which to return the current PCIe link generation
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a currLinkWidth has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a currLinkWidth is null
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if PCIe link information is not available
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int *currLinkWidth);

/**
 * Retrieve PCIe utilization information.
 * This function is querying a byte counter over a 20ms interval and thus is the
 *   PCIe throughput over that interval.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * This method is not supported in virtual machines running virtual GPU (vGPU).
 *
 * @param device                               The identifier of the target device
 * @param counter                              The specific counter that should be queried \ref nvmlPcieUtilCounter_t
 * @param value                                Reference in which to return throughput in KB/s
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a value has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a counter is invalid, or \a value is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int *value);

/**
 * Retrieve the PCIe replay counter.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param value                                Reference in which to return the counter's value
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a value has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a value is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int *value);

/**
 * Retrieves the current clock speeds for the device.
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * See \ref nvmlClockType_t for details on available clock information.
 *
 * @param device                               The identifier of the target device
 * @param type                                 Identify which clock domain to query
 * @param clock                                Reference in which to return the clock speed in MHz
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a clock has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clock is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device cannot report the specified clock
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock);

/**
 * Retrieves the maximum clock speeds for the device.
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * See \ref nvmlClockType_t for details on available clock information.
 *
 * \note On GPUs from Fermi family current P0 clocks (reported by \ref nvmlDeviceGetClockInfo) can differ from max clocks
 *       by few MHz.
 *
 * @param device                               The identifier of the target device
 * @param type                                 Identify which clock domain to query
 * @param clock                                Reference in which to return the clock speed in MHz
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a clock has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clock is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device cannot report the specified clock
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock);

/**
 * Retrieves the current setting of a clock that applications will use unless an overspec situation occurs.
 * Can be changed using \ref nvmlDeviceSetApplicationsClocks.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param clockType                            Identify which clock domain to query
 * @param clockMHz                             Reference in which to return the clock in MHz
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a clockMHz has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clockMHz is NULL or \a clockType is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz);

/**
 * Retrieves the default applications clock that GPU boots with or
 * defaults to after \ref nvmlDeviceResetApplicationsClocks call.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param clockType                            Identify which clock domain to query
 * @param clockMHz                             Reference in which to return the default clock in MHz
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a clockMHz has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clockMHz is NULL or \a clockType is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * \see nvmlDeviceGetApplicationsClock
 */
nvmlReturn_t DECLDIR nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz);

/**
 * Resets the application clock to the default value
 *
 * This is the applications clock that will be used after system reboot or driver reload.
 * Default value is constant, but the current value an be changed using \ref nvmlDeviceSetApplicationsClocks.
 *
 * On Pascal and newer hardware, if clocks were previously locked with \ref nvmlDeviceSetApplicationsClocks,
 * this call will unlock clocks. This returns clocks their default behavior ofautomatically boosting above
 * base clocks as thermal limits allow.
 *
 * @see nvmlDeviceGetApplicationsClock
 * @see nvmlDeviceSetApplicationsClocks
 *
 * For Fermi &tm; or newer non-GeForce fully supported devices and Maxwell or newer GeForce devices.
 *
 * @param device                               The identifier of the target device
 *
 * @return
 *         - \ref NVML_SUCCESS                 if new settings were successfully set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceResetApplicationsClocks(nvmlDevice_t device);

/**
 * Retrieves the clock speed for the clock specified by the clock type and clock ID.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param clockType                            Identify which clock domain to query
 * @param clockId                              Identify which clock in the domain to query
 * @param clockMHz                             Reference in which to return the clock in MHz
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a clockMHz has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clockMHz is NULL or \a clockType is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int *clockMHz);

/**
 * Retrieves the customer defined maximum boost clock speed specified by the given clock type.
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param clockType                            Identify which clock domain to query
 * @param clockMHz                             Reference in which to return the clock in MHz
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a clockMHz has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clockMHz is NULL or \a clockType is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device or the \a clockType on this device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz);

/**
 * Retrieves the list of possible memory clocks that can be used as an argument for \ref nvmlDeviceSetApplicationsClocks.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param count                                Reference in which to provide the \a clocksMHz array size, and
 *                                             to return the number of elements
 * @param clocksMHz                            Reference in which to return the clock in MHz
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a count and \a clocksMHz have been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a count is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a count is too small (\a count is set to the number of
 *                                                required elements)
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceSetApplicationsClocks
 * @see nvmlDeviceGetSupportedGraphicsClocks
 */
nvmlReturn_t DECLDIR nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int *count, unsigned int *clocksMHz);

/**
 * Retrieves the list of possible graphics clocks that can be used as an argument for \ref nvmlDeviceSetApplicationsClocks.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param memoryClockMHz                       Memory clock for which to return possible graphics clocks
 * @param count                                Reference in which to provide the \a clocksMHz array size, and
 *                                             to return the number of elements
 * @param clocksMHz                            Reference in which to return the clocks in MHz
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a count and \a clocksMHz have been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_NOT_FOUND         if the specified \a memoryClockMHz is not a supported frequency
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clock is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a count is too small
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceSetApplicationsClocks
 * @see nvmlDeviceGetSupportedMemoryClocks
 */
nvmlReturn_t DECLDIR nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int *count, unsigned int *clocksMHz);

/**
 * Retrieve the current state of Auto Boosted clocks on a device and store it in \a isEnabled
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates
 * to maximize performance as thermal limits allow.
 *
 * On Pascal and newer hardware, Auto Aoosted clocks are controlled through application clocks.
 * Use \ref nvmlDeviceSetApplicationsClocks and \ref nvmlDeviceResetApplicationsClocks to control Auto Boost
 * behavior.
 *
 * @param device                               The identifier of the target device
 * @param isEnabled                            Where to store the current state of Auto Boosted clocks of the target device
 * @param defaultIsEnabled                     Where to store the default Auto Boosted clocks behavior of the target device that the device will
 *                                                 revert to when no applications are using the GPU
 *
 * @return
 *         - \ref NVML_SUCCESS                 If \a isEnabled has been been set with the Auto Boosted clocks state of \a device
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a isEnabled is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support Auto Boosted clocks
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 */
nvmlReturn_t DECLDIR nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t *isEnabled, nvmlEnableState_t *defaultIsEnabled);

/**
 * Try to set the current state of Auto Boosted clocks on a device.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates
 * to maximize performance as thermal limits allow. Auto Boosted clocks should be disabled if fixed clock
 * rates are desired.
 *
 * Non-root users may use this API by default but can be restricted by root from using this API by calling
 * \ref nvmlDeviceSetAPIRestriction with apiType=NVML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS.
 * Note: Persistence Mode is required to modify current Auto Boost settings, therefore, it must be enabled.
 *
 * On Pascal and newer hardware, Auto Boosted clocks are controlled through application clocks.
 * Use \ref nvmlDeviceSetApplicationsClocks and \ref nvmlDeviceResetApplicationsClocks to control Auto Boost
 * behavior.
 *
 * @param device                               The identifier of the target device
 * @param enabled                              What state to try to set Auto Boosted clocks of the target device to
 *
 * @return
 *         - \ref NVML_SUCCESS                 If the Auto Boosted clocks were successfully set to the state specified by \a enabled
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support Auto Boosted clocks
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 */
nvmlReturn_t DECLDIR nvmlDeviceSetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled);

/**
 * Try to set the default state of Auto Boosted clocks on a device. This is the default state that Auto Boosted clocks will
 * return to when no compute running processes (e.g. CUDA application which have an active context) are running
 *
 * For Kepler &tm; or newer non-GeForce fully supported devices and Maxwell or newer GeForce devices.
 * Requires root/admin permissions.
 *
 * Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates
 * to maximize performance as thermal limits allow. Auto Boosted clocks should be disabled if fixed clock
 * rates are desired.
 *
 * On Pascal and newer hardware, Auto Boosted clocks are controlled through application clocks.
 * Use \ref nvmlDeviceSetApplicationsClocks and \ref nvmlDeviceResetApplicationsClocks to control Auto Boost
 * behavior.
 *
 * @param device                               The identifier of the target device
 * @param enabled                              What state to try to set default Auto Boosted clocks of the target device to
 * @param flags                                Flags that change the default behavior. Currently Unused.
 *
 * @return
 *         - \ref NVML_SUCCESS                 If the Auto Boosted clock's default state was successfully set to the state specified by \a enabled
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_NO_PERMISSION     If the calling user does not have permission to change Auto Boosted clock's default state.
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support Auto Boosted clocks
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 */
nvmlReturn_t DECLDIR nvmlDeviceSetDefaultAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled, unsigned int flags);


/**
 * Retrieves the intended operating speed of the device's fan.
 *
 * Note: The reported speed is the intended fan speed.  If the fan is physically blocked and unable to spin, the
 * output will not match the actual fan speed.
 *
 * For all discrete products with dedicated fans.
 *
 * The fan speed is expressed as a percentage of the product's maximum noise tolerance fan speed.
 * This value may exceed 100% in certain cases.
 *
 * @param device                               The identifier of the target device
 * @param speed                                Reference in which to return the fan speed percentage
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a speed has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a speed is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have a fan
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed);


/**
 * Retrieves the intended operating speed of the device's specified fan.
 *
 * Note: The reported speed is the intended fan speed. If the fan is physically blocked and unable to spin, the
 * output will not match the actual fan speed.
 *
 * For all discrete products with dedicated fans.
 *
 * The fan speed is expressed as a percentage of the product's maximum noise tolerance fan speed.
 * This value may exceed 100% in certain cases.
 *
 * @param device                                The identifier of the target device
 * @param fan                                   The index of the target fan, zero indexed.
 * @param speed                                 Reference in which to return the fan speed percentage
 *
 * @return
 *        - \ref NVML_SUCCESS                   if \a speed has been set
 *        - \ref NVML_ERROR_UNINITIALIZED       if the library has not been successfully initialized
 *        - \ref NVML_ERROR_INVALID_ARGUMENT    if \a device is invalid, \a fan is not an acceptable index, or \a speed is NULL
 *        - \ref NVML_ERROR_NOT_SUPPORTED       if the device does not have a fan or is newer than Maxwell
 *        - \ref NVML_ERROR_GPU_IS_LOST         if the target GPU has fallen off the bus or is otherwise inaccessible
 *        - \ref NVML_ERROR_UNKNOWN             on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int * speed);

/**
 * Retrieves the number of fans on the device.
 *
 * For all discrete products with dedicated fans.
 *
 * @param device                               The identifier of the target device
 * @param numFans                              The number of fans
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a fan number query was successful
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a numFans is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have a fan
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetNumFans(nvmlDevice_t device, unsigned int *numFans);

/**
 * Retrieves the current temperature readings for the device, in degrees C.
 *
 * For all products.
 *
 * See \ref nvmlTemperatureSensors_t for details on available temperature sensors.
 *
 * @param device                               The identifier of the target device
 * @param sensorType                           Flag that indicates which sensor reading to retrieve
 * @param temp                                 Reference in which to return the temperature reading
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a temp has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a sensorType is invalid or \a temp is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have the specified sensor
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetTemperature(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int *temp);

/**
 * Retrieves the temperature threshold for the GPU with the specified threshold type in degrees C.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * See \ref nvmlTemperatureThresholds_t for details on available temperature thresholds.
 *
 * @param device                               The identifier of the target device
 * @param thresholdType                        The type of threshold value queried
 * @param temp                                 Reference in which to return the temperature reading
 * @return
 *         - \ref NVML_SUCCESS                 if \a temp has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a thresholdType is invalid or \a temp is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have a temperature sensor or is unsupported
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int *temp);

/**
 * Sets the temperature threshold for the GPU with the specified threshold type in degrees C.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * See \ref nvmlTemperatureThresholds_t for details on available temperature thresholds.
 *
 * @param device                               The identifier of the target device
 * @param thresholdType                        The type of threshold value to be set
 * @param temp                                 Reference which hold the value to be set
 * @return
 *         - \ref NVML_SUCCESS                 if \a temp has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a thresholdType is invalid or \a temp is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have a temperature sensor or is unsupported
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceSetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int *temp);

/**
 * Retrieves the current performance state for the device.
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * See \ref nvmlPstates_t for details on allowed performance states.
 *
 * @param device                               The identifier of the target device
 * @param pState                               Reference in which to return the performance state reading
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a pState has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a pState is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t *pState);

/**
 * Retrieves current clocks throttling reasons.
 *
 * For all fully supported products.
 *
 * \note More than one bit can be enabled at the same time. Multiple reasons can be affecting clocks at once.
 *
 * @param device                                The identifier of the target device
 * @param clocksThrottleReasons                 Reference in which to return bitmask of active clocks throttle
 *                                                  reasons
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a clocksThrottleReasons has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clocksThrottleReasons is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlClocksThrottleReasons
 * @see nvmlDeviceGetSupportedClocksThrottleReasons
 */
nvmlReturn_t DECLDIR nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice_t device, unsigned long long *clocksThrottleReasons);

/**
 * Retrieves bitmask of supported clocks throttle reasons that can be returned by
 * \ref nvmlDeviceGetCurrentClocksThrottleReasons
 *
 * For all fully supported products.
 *
 * This method is not supported in virtual machines running virtual GPU (vGPU).
 *
 * @param device                               The identifier of the target device
 * @param supportedClocksThrottleReasons       Reference in which to return bitmask of supported
 *                                              clocks throttle reasons
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a supportedClocksThrottleReasons has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a supportedClocksThrottleReasons is NULL
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlClocksThrottleReasons
 * @see nvmlDeviceGetCurrentClocksThrottleReasons
 */
nvmlReturn_t DECLDIR nvmlDeviceGetSupportedClocksThrottleReasons(nvmlDevice_t device, unsigned long long *supportedClocksThrottleReasons);

/**
 * Deprecated: Use \ref nvmlDeviceGetPerformanceState. This function exposes an incorrect generalization.
 *
 * Retrieve the current performance state for the device.
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * See \ref nvmlPstates_t for details on allowed performance states.
 *
 * @param device                               The identifier of the target device
 * @param pState                               Reference in which to return the performance state reading
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a pState has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a pState is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t *pState);

/**
 * This API has been deprecated.
 *
 * Retrieves the power management mode associated with this device.
 *
 * For products from the Fermi family.
 *     - Requires \a NVML_INFOROM_POWER version 3.0 or higher.
 *
 * For from the Kepler or newer families.
 *     - Does not require \a NVML_INFOROM_POWER object.
 *
 * This flag indicates whether any power management algorithm is currently active on the device. An
 * enabled state does not necessarily mean the device is being actively throttled -- only that
 * that the driver will do so if the appropriate conditions are met.
 *
 * See \ref nvmlEnableState_t for details on allowed modes.
 *
 * @param device                               The identifier of the target device
 * @param mode                                 Reference in which to return the current power management mode
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a mode has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetPowerManagementMode(nvmlDevice_t device, nvmlEnableState_t *mode);

/**
 * Retrieves the power management limit associated with this device.
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * The power limit defines the upper boundary for the card's power draw. If
 * the card's total power draw reaches this limit the power management algorithm kicks in.
 *
 * This reading is only available if power management mode is supported.
 * See \ref nvmlDeviceGetPowerManagementMode.
 *
 * @param device                               The identifier of the target device
 * @param limit                                Reference in which to return the power management limit in milliwatts
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a limit has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a limit is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int *limit);

/**
 * Retrieves information about possible values of power management limits on this device.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param minLimit                             Reference in which to return the minimum power management limit in milliwatts
 * @param maxLimit                             Reference in which to return the maximum power management limit in milliwatts
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a minLimit and \a maxLimit have been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a minLimit or \a maxLimit is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceSetPowerManagementLimit
 */
nvmlReturn_t DECLDIR nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int *minLimit, unsigned int *maxLimit);

/**
 * Retrieves default power management limit on this device, in milliwatts.
 * Default power management limit is a power management limit that the device boots with.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param defaultLimit                         Reference in which to return the default power management limit in milliwatts
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a defaultLimit has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a defaultLimit is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int *defaultLimit);

/**
 * Retrieves power usage for this GPU in milliwatts and its associated circuitry (e.g. memory)
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * On Fermi and Kepler GPUs the reading is accurate to within +/- 5% of current power draw.
 *
 * It is only available if power management mode is supported. See \ref nvmlDeviceGetPowerManagementMode.
 *
 * @param device                               The identifier of the target device
 * @param power                                Reference in which to return the power usage information
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a power has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a power is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support power readings
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power);

/**
 * Retrieves total energy consumption for this GPU in millijoules (mJ) since the driver was last reloaded
 *
 * For Volta &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param energy                               Reference in which to return the energy consumption information
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a energy has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a energy is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support energy readings
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device, unsigned long long *energy);

/**
 * Get the effective power limit that the driver enforces after taking into account all limiters
 *
 * Note: This can be different from the \ref nvmlDeviceGetPowerManagementLimit if other limits are set elsewhere
 * This includes the out of band power limit interface
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                           The device to communicate with
 * @param limit                            Reference in which to return the power management limit in milliwatts
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a limit has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a limit is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int *limit);

/**
 * Retrieves the current GOM and pending GOM (the one that GPU will switch to after reboot).
 *
 * For GK110 M-class and X-class Tesla &tm; products from the Kepler family.
 * Modes \ref NVML_GOM_LOW_DP and \ref NVML_GOM_ALL_ON are supported on fully supported GeForce products.
 * Not supported on Quadro &reg; and Tesla &tm; C-class products.
 *
 * @param device                               The identifier of the target device
 * @param current                              Reference in which to return the current GOM
 * @param pending                              Reference in which to return the pending GOM
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a mode has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a current or \a pending is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlGpuOperationMode_t
 * @see nvmlDeviceSetGpuOperationMode
 */
nvmlReturn_t DECLDIR nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t *current, nvmlGpuOperationMode_t *pending);

/**
 * Retrieves the amount of used, free, reserved and total memory available on the device, in bytes.
 * The reserved amount is supported on version 2 only.
 *
 * For all products.
 *
 * Enabling ECC reduces the amount of total available memory, due to the extra required parity bits.
 * Under WDDM most device memory is allocated and managed on startup by Windows.
 *
 * Under Linux and Windows TCC, the reported amount of used memory is equal to the sum of memory allocated
 * by all active channels on the device.
 *
 * See \ref nvmlMemory_v2_t for details on available memory info.
 *
 * @note In MIG mode, if device handle is provided, the API returns aggregate
 *       information, only if the caller has appropriate privileges. Per-instance
 *       information can be queried by using specific MIG device handles.
 * 
 * @note nvmlDeviceGetMemoryInfo_v2 adds additional memory information.
 *
 * @param device                               The identifier of the target device
 * @param memory                               Reference in which to return the memory information
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a memory has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a memory is NULL
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory);
nvmlReturn_t DECLDIR nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory);

/**
 * Retrieves the current compute mode for the device.
 *
 * For all products.
 *
 * See \ref nvmlComputeMode_t for details on allowed compute modes.
 *
 * @param device                               The identifier of the target device
 * @param mode                                 Reference in which to return the current compute mode
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a mode has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceSetComputeMode()
 */
nvmlReturn_t DECLDIR nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t *mode);

/**
 * Retrieves the CUDA compute capability of the device.
 *
 * For all products.
 *
 * Returns the major and minor compute capability version numbers of the
 * device.  The major and minor versions are equivalent to the
 * CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR and
 * CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR attributes that would be
 * returned by CUDA's cuDeviceGetAttribute().
 *
 * @param device                               The identifier of the target device
 * @param major                                Reference in which to return the major CUDA compute capability
 * @param minor                                Reference in which to return the minor CUDA compute capability
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a major and \a minor have been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a major or \a minor are NULL
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major, int *minor);

/**
 * Retrieves the current and pending ECC modes for the device.
 *
 * For Fermi &tm; or newer fully supported devices.
 * Only applicable to devices with ECC.
 * Requires \a NVML_INFOROM_ECC version 1.0 or higher.
 *
 * Changing ECC modes requires a reboot. The "pending" ECC mode refers to the target mode following
 * the next reboot.
 *
 * See \ref nvmlEnableState_t for details on allowed modes.
 *
 * @param device                               The identifier of the target device
 * @param current                              Reference in which to return the current ECC mode
 * @param pending                              Reference in which to return the pending ECC mode
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a current and \a pending have been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or either \a current or \a pending is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceSetEccMode()
 */
nvmlReturn_t DECLDIR nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t *current, nvmlEnableState_t *pending);

/**
 * Retrieves the device boardId from 0-N.
 * Devices with the same boardId indicate GPUs connected to the same PLX.  Use in conjunction with
 *  \ref nvmlDeviceGetMultiGpuBoard() to decide if they are on the same board as well.
 *  The boardId returned is a unique ID for the current configuration.  Uniqueness and ordering across
 *  reboots and system configurations is not guaranteed (i.e. if a Tesla K40c returns 0x100 and
 *  the two GPUs on a Tesla K10 in the same system returns 0x200 it is not guaranteed they will
 *  always return those values but they will always be different from each other).
 *
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param boardId                              Reference in which to return the device's board ID
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a boardId has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a boardId is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int *boardId);

/**
 * Retrieves whether the device is on a Multi-GPU Board
 * Devices that are on multi-GPU boards will set \a multiGpuBool to a non-zero value.
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param multiGpuBool                         Reference in which to return a zero or non-zero value
 *                                                 to indicate whether the device is on a multi GPU board
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a multiGpuBool has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a multiGpuBool is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int *multiGpuBool);

/**
 * Retrieves the total ECC error counts for the device.
 *
 * For Fermi &tm; or newer fully supported devices.
 * Only applicable to devices with ECC.
 * Requires \a NVML_INFOROM_ECC version 1.0 or higher.
 * Requires ECC Mode to be enabled.
 *
 * The total error count is the sum of errors across each of the separate memory systems, i.e. the total set of
 * errors across the entire device.
 *
 * See \ref nvmlMemoryErrorType_t for a description of available error types.\n
 * See \ref nvmlEccCounterType_t for a description of available counter types.
 *
 * @param device                               The identifier of the target device
 * @param errorType                            Flag that specifies the type of the errors.
 * @param counterType                          Flag that specifies the counter-type of the errors.
 * @param eccCounts                            Reference in which to return the specified ECC errors
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a eccCounts has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a errorType or \a counterType is invalid, or \a eccCounts is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceClearEccErrorCounts()
 */
nvmlReturn_t DECLDIR nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long *eccCounts);

/**
 * Retrieves the detailed ECC error counts for the device.
 *
 * @deprecated   This API supports only a fixed set of ECC error locations
 *               On different GPU architectures different locations are supported
 *               See \ref nvmlDeviceGetMemoryErrorCounter
 *
 * For Fermi &tm; or newer fully supported devices.
 * Only applicable to devices with ECC.
 * Requires \a NVML_INFOROM_ECC version 2.0 or higher to report aggregate location-based ECC counts.
 * Requires \a NVML_INFOROM_ECC version 1.0 or higher to report all other ECC counts.
 * Requires ECC Mode to be enabled.
 *
 * Detailed errors provide separate ECC counts for specific parts of the memory system.
 *
 * Reports zero for unsupported ECC error counters when a subset of ECC error counters are supported.
 *
 * See \ref nvmlMemoryErrorType_t for a description of available bit types.\n
 * See \ref nvmlEccCounterType_t for a description of available counter types.\n
 * See \ref nvmlEccErrorCounts_t for a description of provided detailed ECC counts.
 *
 * @param device                               The identifier of the target device
 * @param errorType                            Flag that specifies the type of the errors.
 * @param counterType                          Flag that specifies the counter-type of the errors.
 * @param eccCounts                            Reference in which to return the specified ECC errors
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a eccCounts has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a errorType or \a counterType is invalid, or \a eccCounts is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceClearEccErrorCounts()
 */
nvmlReturn_t DECLDIR nvmlDeviceGetDetailedEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlEccErrorCounts_t *eccCounts);

/**
 * Retrieves the requested memory error counter for the device.
 *
 * For Fermi &tm; or newer fully supported devices.
 * Requires \a NVML_INFOROM_ECC version 2.0 or higher to report aggregate location-based memory error counts.
 * Requires \a NVML_INFOROM_ECC version 1.0 or higher to report all other memory error counts.
 *
 * Only applicable to devices with ECC.
 *
 * Requires ECC Mode to be enabled.
 *
 * @note On MIG-enabled GPUs, per instance information can be queried using specific
 *       MIG device handles. Per instance information is currently only supported for
 *       non-DRAM uncorrectable volatile errors. Querying volatile errors using device
 *       handles is currently not supported.
 *
 * See \ref nvmlMemoryErrorType_t for a description of available memory error types.\n
 * See \ref nvmlEccCounterType_t for a description of available counter types.\n
 * See \ref nvmlMemoryLocation_t for a description of available counter locations.\n
 *
 * @param device                               The identifier of the target device
 * @param errorType                            Flag that specifies the type of error.
 * @param counterType                          Flag that specifies the counter-type of the errors.
 * @param locationType                         Specifies the location of the counter.
 * @param count                                Reference in which to return the ECC counter
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a count has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a bitTyp,e \a counterType or \a locationType is
 *                                             invalid, or \a count is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support ECC error reporting in the specified memory
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType,
                                                   nvmlEccCounterType_t counterType,
                                                   nvmlMemoryLocation_t locationType, unsigned long long *count);

/**
 * Retrieves the current utilization rates for the device's major subsystems.
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * See \ref nvmlUtilization_t for details on available utilization rates.
 *
 * \note During driver initialization when ECC is enabled one can see high GPU and Memory Utilization readings.
 *       This is caused by ECC Memory Scrubbing mechanism that is performed during driver initialization.
 *
 * @note On MIG-enabled GPUs, querying device utilization rates is not currently supported.
 *
 * @param device                               The identifier of the target device
 * @param utilization                          Reference in which to return the utilization information
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a utilization has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a utilization is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t *utilization);

/**
 * Retrieves the current utilization and sampling size in microseconds for the Encoder
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @note On MIG-enabled GPUs, querying encoder utilization is not currently supported.
 *
 * @param device                               The identifier of the target device
 * @param utilization                          Reference to an unsigned int for encoder utilization info
 * @param samplingPeriodUs                     Reference to an unsigned int for the sampling period in US
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a utilization has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a utilization is NULL, or \a samplingPeriodUs is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs);

/**
 * Retrieves the current capacity of the device's encoder, as a percentage of maximum encoder capacity with valid values in the range 0-100.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param device                            The identifier of the target device
 * @param encoderQueryType                  Type of encoder to query
 * @param encoderCapacity                   Reference to an unsigned int for the encoder capacity
 *
 * @return
 *         - \ref NVML_SUCCESS                  if \a encoderCapacity is fetched
 *         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a encoderCapacity is NULL, or \a device or \a encoderQueryType
 *                                              are invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED      if device does not support the encoder specified in \a encodeQueryType
 *         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetEncoderCapacity (nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int *encoderCapacity);

/**
 * Retrieves the current encoder statistics for a given device.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param device                            The identifier of the target device
 * @param sessionCount                      Reference to an unsigned int for count of active encoder sessions
 * @param averageFps                        Reference to an unsigned int for trailing average FPS of all active sessions
 * @param averageLatency                    Reference to an unsigned int for encode latency in microseconds
 *
 * @return
 *         - \ref NVML_SUCCESS                  if \a sessionCount, \a averageFps and \a averageLatency is fetched
 *         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a sessionCount, or \a device or \a averageFps,
 *                                              or \a averageLatency is NULL
 *         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetEncoderStats (nvmlDevice_t device, unsigned int *sessionCount,
                                                unsigned int *averageFps, unsigned int *averageLatency);

/**
 * Retrieves information about active encoder sessions on a target device.
 *
 * An array of active encoder sessions is returned in the caller-supplied buffer pointed at by \a sessionInfos. The
 * array elememt count is passed in \a sessionCount, and \a sessionCount is used to return the number of sessions
 * written to the buffer.
 *
 * If the supplied buffer is not large enough to accomodate the active session array, the function returns
 * NVML_ERROR_INSUFFICIENT_SIZE, with the element count of nvmlEncoderSessionInfo_t array required in \a sessionCount.
 * To query the number of active encoder sessions, call this function with *sessionCount = 0.  The code will return
 * NVML_SUCCESS with number of active encoder sessions updated in *sessionCount.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param device                            The identifier of the target device
 * @param sessionCount                      Reference to caller supplied array size, and returns the number of sessions.
 * @param sessionInfos                      Reference in which to return the session information
 *
 * @return
 *         - \ref NVML_SUCCESS                  if \a sessionInfos is fetched
 *         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE  if \a sessionCount is too small, array element count is returned in \a sessionCount
 *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a sessionCount is NULL.
 *         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_NOT_SUPPORTED      if this query is not supported by \a device
 *         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfos);

/**
 * Retrieves the current utilization and sampling size in microseconds for the Decoder
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @note On MIG-enabled GPUs, querying decoder utilization is not currently supported.
 *
 * @param device                               The identifier of the target device
 * @param utilization                          Reference to an unsigned int for decoder utilization info
 * @param samplingPeriodUs                     Reference to an unsigned int for the sampling period in US
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a utilization has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a utilization is NULL, or \a samplingPeriodUs is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs);

/**
* Retrieves the active frame buffer capture sessions statistics for a given device.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param device                            The identifier of the target device
* @param fbcStats                          Reference to nvmlFBCStats_t structure contianing NvFBC stats
*
* @return
*         - \ref NVML_SUCCESS                  if \a fbcStats is fetched
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a fbcStats is NULL
*         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t DECLDIR nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t *fbcStats);

/**
* Retrieves information about active frame buffer capture sessions on a target device.
*
* An array of active FBC sessions is returned in the caller-supplied buffer pointed at by \a sessionInfo. The
* array element count is passed in \a sessionCount, and \a sessionCount is used to return the number of sessions
* written to the buffer.
*
* If the supplied buffer is not large enough to accomodate the active session array, the function returns
* NVML_ERROR_INSUFFICIENT_SIZE, with the element count of nvmlFBCSessionInfo_t array required in \a sessionCount.
* To query the number of active FBC sessions, call this function with *sessionCount = 0.  The code will return
* NVML_SUCCESS with number of active FBC sessions updated in *sessionCount.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @note hResolution, vResolution, averageFPS and averageLatency data for a FBC session returned in \a sessionInfo may
*       be zero if there are no new frames captured since the session started.
*
* @param device                            The identifier of the target device
* @param sessionCount                      Reference to caller supplied array size, and returns the number of sessions.
* @param sessionInfo                       Reference in which to return the session information
*
* @return
*         - \ref NVML_SUCCESS                  if \a sessionInfo is fetched
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE  if \a sessionCount is too small, array element count is returned in \a sessionCount
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a sessionCount is NULL.
*         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t DECLDIR nvmlDeviceGetFBCSessions(nvmlDevice_t device, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo);

/**
 * Retrieves the current and pending driver model for the device.
 *
 * For Fermi &tm; or newer fully supported devices.
 * For windows only.
 *
 * On Windows platforms the device driver can run in either WDDM or WDM (TCC) mode. If a display is attached
 * to the device it must run in WDDM mode. TCC mode is preferred if a display is not attached.
 *
 * See \ref nvmlDriverModel_t for details on available driver models.
 *
 * @param device                               The identifier of the target device
 * @param current                              Reference in which to return the current driver model
 * @param pending                              Reference in which to return the pending driver model
 *
 * @return
 *         - \ref NVML_SUCCESS                 if either \a current and/or \a pending have been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or both \a current and \a pending are NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the platform is not windows
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceSetDriverModel()
 */
nvmlReturn_t DECLDIR nvmlDeviceGetDriverModel(nvmlDevice_t device, nvmlDriverModel_t *current, nvmlDriverModel_t *pending);

/**
 * Get VBIOS version of the device.
 *
 * For all products.
 *
 * The VBIOS version may change from time to time. It will not exceed 32 characters in length
 * (including the NULL terminator).  See \ref nvmlConstants::NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE.
 *
 * @param device                               The identifier of the target device
 * @param version                              Reference to which to return the VBIOS version
 * @param length                               The maximum allowed length of the string returned in \a version
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a version has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a version is NULL
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char *version, unsigned int length);

/**
 * Get Bridge Chip Information for all the bridge chips on the board.
 *
 * For all fully supported products.
 * Only applicable to multi-GPU products.
 *
 * @param device                                The identifier of the target device
 * @param bridgeHierarchy                       Reference to the returned bridge chip Hierarchy
 *
 * @return
 *         - \ref NVML_SUCCESS                 if bridge chip exists
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a bridgeInfo is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if bridge chip not supported on the device
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 */
nvmlReturn_t DECLDIR nvmlDeviceGetBridgeChipInfo(nvmlDevice_t device, nvmlBridgeChipHierarchy_t *bridgeHierarchy);

/**
 * Get information about processes with a compute context on a device
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * This function returns information only about compute running processes (e.g. CUDA application which have
 * active context). Any graphics applications (e.g. using OpenGL, DirectX) won't be listed by this function.
 *
 * To query the current number of running compute processes, call this function with *infoCount = 0. The
 * return code will be NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if none are running. For this call
 * \a infos is allowed to be NULL.
 *
 * The usedGpuMemory field returned is all of the memory used by the application.
 *
 * Keep in mind that information returned by this call is dynamic and the number of elements might change in
 * time. Allocate more space for \a infos table in case new compute processes are spawned.
 *
 * @note In MIG mode, if device handle is provided, the API returns aggregate information, only if
 *       the caller has appropriate privileges. Per-instance information can be queried by using
 *       specific MIG device handles.
 *       Querying per-instance information using MIG device handles is not supported if the device is in vGPU Host virtualization mode.
 *
 * @param device                               The device handle or MIG device handle
 * @param infoCount                            Reference in which to provide the \a infos array size, and
 *                                             to return the number of returned elements
 * @param infos                                Reference in which to return the process information
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a infoCount and \a infos have been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a infoCount indicates that the \a infos array is too small
 *                                             \a infoCount will contain minimal amount of space necessary for
 *                                             the call to complete
 *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, either of \a infoCount or \a infos is NULL
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by \a device
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see \ref nvmlSystemGetProcessName
 */
nvmlReturn_t DECLDIR nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos);

/**
 * Get information about processes with a graphics context on a device
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * This function returns information only about graphics based processes
 * (eg. applications using OpenGL, DirectX)
 *
 * To query the current number of running graphics processes, call this function with *infoCount = 0. The
 * return code will be NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if none are running. For this call
 * \a infos is allowed to be NULL.
 *
 * The usedGpuMemory field returned is all of the memory used by the application.
 *
 * Keep in mind that information returned by this call is dynamic and the number of elements might change in
 * time. Allocate more space for \a infos table in case new graphics processes are spawned.
 *
 * @note In MIG mode, if device handle is provided, the API returns aggregate information, only if
 *       the caller has appropriate privileges. Per-instance information can be queried by using
 *       specific MIG device handles.
 *       Querying per-instance information using MIG device handles is not supported if the device is in vGPU Host virtualization mode.
 *
 * @param device                               The device handle or MIG device handle
 * @param infoCount                            Reference in which to provide the \a infos array size, and
 *                                             to return the number of returned elements
 * @param infos                                Reference in which to return the process information
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a infoCount and \a infos have been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a infoCount indicates that the \a infos array is too small
 *                                             \a infoCount will contain minimal amount of space necessary for
 *                                             the call to complete
 *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, either of \a infoCount or \a infos is NULL
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by \a device
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see \ref nvmlSystemGetProcessName
 */
nvmlReturn_t DECLDIR nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos);

/**
 * Get information about processes with a MPS compute context on a device
 *
 * For Volta &tm; or newer fully supported devices.
 *
 * This function returns information only about compute running processes (e.g. CUDA application which have
 * active context) utilizing MPS. Any graphics applications (e.g. using OpenGL, DirectX) won't be listed by
 * this function.
 *
 * To query the current number of running compute processes, call this function with *infoCount = 0. The
 * return code will be NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if none are running. For this call
 * \a infos is allowed to be NULL.
 *
 * The usedGpuMemory field returned is all of the memory used by the application.
 *
 * Keep in mind that information returned by this call is dynamic and the number of elements might change in
 * time. Allocate more space for \a infos table in case new compute processes are spawned.
 * 
 * @note In MIG mode, if device handle is provided, the API returns aggregate information, only if
 *       the caller has appropriate privileges. Per-instance information can be queried by using
 *       specific MIG device handles.
 *       Querying per-instance information using MIG device handles is not supported if the device is in vGPU Host virtualization mode.
 *
 * @param device                               The device handle or MIG device handle
 * @param infoCount                            Reference in which to provide the \a infos array size, and
 *                                             to return the number of returned elements
 * @param infos                                Reference in which to return the process information
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a infoCount and \a infos have been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a infoCount indicates that the \a infos array is too small
 *                                             \a infoCount will contain minimal amount of space necessary for
 *                                             the call to complete
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, either of \a infoCount or \a infos is NULL
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by \a device
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see \ref nvmlSystemGetProcessName
 */
nvmlReturn_t DECLDIR nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos);

/**
 * Check if the GPU devices are on the same physical board.
 *
 * For all fully supported products.
 *
 * @param device1                               The first GPU device
 * @param device2                               The second GPU device
 * @param onSameBoard                           Reference in which to return the status.
 *                                              Non-zero indicates that the GPUs are on the same board.
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a onSameBoard has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a dev1 or \a dev2 are invalid or \a onSameBoard is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this check is not supported by the device
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the either GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceOnSameBoard(nvmlDevice_t device1, nvmlDevice_t device2, int *onSameBoard);

/**
 * Retrieves the root/admin permissions on the target API. See \a nvmlRestrictedAPI_t for the list of supported APIs.
 * If an API is restricted only root users can call that API. See \a nvmlDeviceSetAPIRestriction to change current permissions.
 *
 * For all fully supported products.
 *
 * @param device                               The identifier of the target device
 * @param apiType                              Target API type for this operation
 * @param isRestricted                         Reference in which to return the current restriction
 *                                             NVML_FEATURE_ENABLED indicates that the API is root-only
 *                                             NVML_FEATURE_DISABLED indicates that the API is accessible to all users
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a isRestricted has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a apiType incorrect or \a isRestricted is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device or the device does not support
 *                                                 the feature that is being queried (E.G. Enabling/disabling Auto Boosted clocks is
 *                                                 not supported by the device)
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlRestrictedAPI_t
 */
nvmlReturn_t DECLDIR nvmlDeviceGetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t *isRestricted);

/**
 * Gets recent samples for the GPU.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * Based on type, this method can be used to fetch the power, utilization or clock samples maintained in the buffer by
 * the driver.
 *
 * Power, Utilization and Clock samples are returned as type "unsigned int" for the union nvmlValue_t.
 *
 * To get the size of samples that user needs to allocate, the method is invoked with samples set to NULL.
 * The returned samplesCount will provide the number of samples that can be queried. The user needs to
 * allocate the buffer with size as samplesCount * sizeof(nvmlSample_t).
 *
 * lastSeenTimeStamp represents CPU timestamp in microseconds. Set it to 0 to fetch all the samples maintained by the
 * underlying buffer. Set lastSeenTimeStamp to one of the timeStamps retrieved from the date of the previous query
 * to get more recent samples.
 *
 * This method fetches the number of entries which can be accommodated in the provided samples array, and the
 * reference samplesCount is updated to indicate how many samples were actually retrieved. The advantage of using this
 * method for samples in contrast to polling via existing methods is to get get higher frequency data at lower polling cost.
 *
 * @note On MIG-enabled GPUs, querying the following sample types, NVML_GPU_UTILIZATION_SAMPLES, NVML_MEMORY_UTILIZATION_SAMPLES
 *       NVML_ENC_UTILIZATION_SAMPLES and NVML_DEC_UTILIZATION_SAMPLES, is not currently supported.
 *
 * @param device                        The identifier for the target device
 * @param type                          Type of sampling event
 * @param lastSeenTimeStamp             Return only samples with timestamp greater than lastSeenTimeStamp.
 * @param sampleValType                 Output parameter to represent the type of sample value as described in nvmlSampleVal_t
 * @param sampleCount                   Reference to provide the number of elements which can be queried in samples array
 * @param samples                       Reference in which samples are returned

 * @return
 *         - \ref NVML_SUCCESS                 if samples are successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a samplesCount is NULL or
 *                                             reference to \a sampleCount is 0 for non null \a samples
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_NOT_FOUND         if sample entries are not found
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetSamples(nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp,
        nvmlValueType_t *sampleValType, unsigned int *sampleCount, nvmlSample_t *samples);

/**
 * Gets Total, Available and Used size of BAR1 memory.
 *
 * BAR1 is used to map the FB (device memory) so that it can be directly accessed by the CPU or by 3rd party
 * devices (peer-to-peer on the PCIE bus).
 *
 * @note In MIG mode, if device handle is provided, the API returns aggregate
 *       information, only if the caller has appropriate privileges. Per-instance
 *       information can be queried by using specific MIG device handles.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param bar1Memory                           Reference in which BAR1 memory
 *                                             information is returned.
 *
 * @return
 *         - \ref NVML_SUCCESS                 if BAR1 memory is successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a bar1Memory is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 */
nvmlReturn_t DECLDIR nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device, nvmlBAR1Memory_t *bar1Memory);

/**
 * Gets the duration of time during which the device was throttled (lower than requested clocks) due to power
 * or thermal constraints.
 *
 * The method is important to users who are tying to understand if their GPUs throttle at any point during their applications. The
 * difference in violation times at two different reference times gives the indication of GPU throttling event.
 *
 * Violation for thermal capping is not supported at this time.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param perfPolicyType                       Represents Performance policy which can trigger GPU throttling
 * @param violTime                             Reference to which violation time related information is returned
 *
 *
 * @return
 *         - \ref NVML_SUCCESS                 if violation time is successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a perfPolicyType is invalid, or \a violTime is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *
 */
nvmlReturn_t DECLDIR nvmlDeviceGetViolationStatus(nvmlDevice_t device, nvmlPerfPolicyType_t perfPolicyType, nvmlViolationTime_t *violTime);

/**
 * Gets the device's interrupt number
 *
 * @param device                               The identifier of the target device
 * @param irqNum                               The interrupt number associated with the specified device
 *
 * @return
 *         - \ref NVML_SUCCESS                 if irq number is successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a irqNum is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *
 */
nvmlReturn_t DECLDIR nvmlDeviceGetIrqNum(nvmlDevice_t device, unsigned int *irqNum);

/**
 * Gets the device's core count
 *
 * @param device                               The identifier of the target device
 * @param numCores                             The number of cores for the specified device
 *
 * @return
 *         - \ref NVML_SUCCESS                 if Gpu core count is successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a numCores is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *
 */
nvmlReturn_t DECLDIR nvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int *numCores);

/**
 * Gets the devices power source
 *
 * @param device                               The identifier of the target device
 * @param powerSource                          The power source of the device
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the current power source was successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a powerSource is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *
 */
nvmlReturn_t DECLDIR nvmlDeviceGetPowerSource(nvmlDevice_t device, nvmlPowerSource_t *powerSource);

/**
 * Gets the device's memory bus width
 *
 * @param device                               The identifier of the target device
 * @param maxSpeed                             The devices's memory bus width
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the memory bus width is successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a busWidth is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *
 */
nvmlReturn_t DECLDIR nvmlDeviceGetMemoryBusWidth(nvmlDevice_t device, unsigned int *busWidth);

/**
 * Gets the device's PCIE Max Link speed in MBPS
 *
 * @param device                               The identifier of the target device
 * @param maxSpeed                             The devices's PCIE Max Link speed in MBPS
 *
 * @return
 *         - \ref NVML_SUCCESS                 if Pcie Max Link Speed is successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a maxSpeed is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *
 */
nvmlReturn_t DECLDIR nvmlDeviceGetPcieLinkMaxSpeed(nvmlDevice_t device, unsigned int *maxSpeed);

/**
 * Gets the device's Adaptive Clock status
 *
 * @param device                               The identifier of the target device
 * @param adaptiveClockStatus                  The current adaptive clocking status
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the current adaptive clocking status is successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a adaptiveClockStatus is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *
 */
nvmlReturn_t DECLDIR nvmlDeviceGetAdaptiveClockInfoStatus(nvmlDevice_t device, unsigned int *adaptiveClockStatus);

/**
 * @}
 */

/** @addtogroup nvmlAccountingStats
 *  @{
 */

/**
 * Queries the state of per process accounting mode.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * See \ref nvmlDeviceGetAccountingStats for more details.
 * See \ref nvmlDeviceSetAccountingMode
 *
 * @param device                               The identifier of the target device
 * @param mode                                 Reference in which to return the current accounting mode
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the mode has been successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode are NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetAccountingMode(nvmlDevice_t device, nvmlEnableState_t *mode);

/**
 * Queries process's accounting stats.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * Accounting stats capture GPU utilization and other statistics across the lifetime of a process.
 * Accounting stats can be queried during life time of the process and after its termination.
 * The time field in \ref nvmlAccountingStats_t is reported as 0 during the lifetime of the process and
 * updated to actual running time after its termination.
 * Accounting stats are kept in a circular buffer, newly created processes overwrite information about old
 * processes.
 *
 * See \ref nvmlAccountingStats_t for description of each returned metric.
 * List of processes that can be queried can be retrieved from \ref nvmlDeviceGetAccountingPids.
 *
 * @note Accounting Mode needs to be on. See \ref nvmlDeviceGetAccountingMode.
 * @note Only compute and graphics applications stats can be queried. Monitoring applications stats can't be
 *         queried since they don't contribute to GPU utilization.
 * @note In case of pid collision stats of only the latest process (that terminated last) will be reported
 *
 * @warning On Kepler devices per process statistics are accurate only if there's one process running on a GPU.
 *
 * @param device                               The identifier of the target device
 * @param pid                                  Process Id of the target process to query stats for
 * @param stats                                Reference in which to return the process's accounting stats
 *
 * @return
 *         - \ref NVML_SUCCESS                 if stats have been successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a stats are NULL
 *         - \ref NVML_ERROR_NOT_FOUND         if process stats were not found
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if \a device doesn't support this feature or accounting mode is disabled
 *                                              or on vGPU host.
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceGetAccountingBufferSize
 */
nvmlReturn_t DECLDIR nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid, nvmlAccountingStats_t *stats);

/**
 * Queries list of processes that can be queried for accounting stats. The list of processes returned
 * can be in running or terminated state.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * To just query the number of processes ready to be queried, call this function with *count = 0 and
 * pids=NULL. The return code will be NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if list is empty.
 *
 * For more details see \ref nvmlDeviceGetAccountingStats.
 *
 * @note In case of PID collision some processes might not be accessible before the circular buffer is full.
 *
 * @param device                               The identifier of the target device
 * @param count                                Reference in which to provide the \a pids array size, and
 *                                               to return the number of elements ready to be queried
 * @param pids                                 Reference in which to return list of process ids
 *
 * @return
 *         - \ref NVML_SUCCESS                 if pids were successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a count is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if \a device doesn't support this feature or accounting mode is disabled
 *                                              or on vGPU host.
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a count is too small (\a count is set to
 *                                                 expected value)
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceGetAccountingBufferSize
 */
nvmlReturn_t DECLDIR nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int *count, unsigned int *pids);

/**
 * Returns the number of processes that the circular buffer with accounting pids can hold.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * This is the maximum number of processes that accounting information will be stored for before information
 * about oldest processes will get overwritten by information about new processes.
 *
 * @param device                               The identifier of the target device
 * @param bufferSize                           Reference in which to provide the size (in number of elements)
 *                                               of the circular buffer for accounting stats.
 *
 * @return
 *         - \ref NVML_SUCCESS                 if buffer size was successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a bufferSize is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature or accounting mode is disabled
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceGetAccountingStats
 * @see nvmlDeviceGetAccountingPids
 */
nvmlReturn_t DECLDIR nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int *bufferSize);

/** @} */

/** @addtogroup nvmlDeviceQueries
 *  @{
 */

/**
 * Returns the list of retired pages by source, including pages that are pending retirement
 * The address information provided from this API is the hardware address of the page that was retired.  Note
 * that this does not match the virtual address used in CUDA, but will match the address information in XID 63
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                            The identifier of the target device
 * @param cause                             Filter page addresses by cause of retirement
 * @param pageCount                         Reference in which to provide the \a addresses buffer size, and
 *                                          to return the number of retired pages that match \a cause
 *                                          Set to 0 to query the size without allocating an \a addresses buffer
 * @param addresses                         Buffer to write the page addresses into
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a pageCount was populated and \a addresses was filled
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a pageCount indicates the buffer is not large enough to store all the
 *                                             matching page addresses.  \a pageCount is set to the needed size.
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a pageCount is NULL, \a cause is invalid, or
 *                                             \a addresses is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetRetiredPages(nvmlDevice_t device, nvmlPageRetirementCause_t cause,
    unsigned int *pageCount, unsigned long long *addresses);

/**
 * Returns the list of retired pages by source, including pages that are pending retirement
 * The address information provided from this API is the hardware address of the page that was retired.  Note
 * that this does not match the virtual address used in CUDA, but will match the address information in XID 63
 *
 * \note nvmlDeviceGetRetiredPages_v2 adds an additional timestamps paramter to return the time of each page's
 *       retirement.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                            The identifier of the target device
 * @param cause                             Filter page addresses by cause of retirement
 * @param pageCount                         Reference in which to provide the \a addresses buffer size, and
 *                                          to return the number of retired pages that match \a cause
 *                                          Set to 0 to query the size without allocating an \a addresses buffer
 * @param addresses                         Buffer to write the page addresses into
 * @param timestamps                        Buffer to write the timestamps of page retirement, additional for _v2
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a pageCount was populated and \a addresses was filled
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a pageCount indicates the buffer is not large enough to store all the
 *                                             matching page addresses.  \a pageCount is set to the needed size.
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a pageCount is NULL, \a cause is invalid, or
 *                                             \a addresses is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device, nvmlPageRetirementCause_t cause,
    unsigned int *pageCount, unsigned long long *addresses, unsigned long long *timestamps);

/**
 * Check if any pages are pending retirement and need a reboot to fully retire.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                            The identifier of the target device
 * @param isPending                         Reference in which to return the pending status
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a isPending was populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a isPending is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device, nvmlEnableState_t *isPending);

/**
 * Get number of remapped rows. The number of rows reported will be based on
 * the cause of the remapping. isPending indicates whether or not there are
 * pending remappings. A reset will be required to actually remap the row.
 * failureOccurred will be set if a row remapping ever failed in the past. A
 * pending remapping won't affect future work on the GPU since
 * error-containment and dynamic page blacklisting will take care of that.
 *
 * @note On MIG-enabled GPUs with active instances, querying the number of
 * remapped rows is not supported
 *
 * For Ampere &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param corrRows                             Reference for number of rows remapped due to correctable errors
 * @param uncRows                              Reference for number of rows remapped due to uncorrectable errors
 * @param isPending                            Reference for whether or not remappings are pending
 * @param failureOccurred                      Reference that is set when a remapping has failed in the past
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a corrRows, \a uncRows, \a isPending or \a failureOccurred is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     If MIG is enabled or if the device doesn't support this feature
 *         - \ref NVML_ERROR_UNKNOWN           Unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int *corrRows, unsigned int *uncRows,
                                               unsigned int *isPending, unsigned int *failureOccurred);

/**
 * Get the row remapper histogram. Returns the remap availability for each bank
 * on the GPU.
 *
 * @param device                               Device handle
 * @param values                               Histogram values
 *
 * @return
 *        - \ref NVML_SUCCESS                  On success
 *        - \ref NVML_ERROR_UNKNOWN            On any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device, nvmlRowRemapperHistogramValues_t *values);

/**
 * Get architecture for device
 *
 * @param device                               The identifier of the target device
 * @param arch                                 Reference where architecture is returned, if call successful.
 *                                             Set to NVML_DEVICE_ARCH_* upon success
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device or \a arch (output refererence) are invalid
 */
nvmlReturn_t DECLDIR nvmlDeviceGetArchitecture(nvmlDevice_t device, nvmlDeviceArchitecture_t *arch);

/** @} */

/***************************************************************************************************/
/** @defgroup nvmlUnitCommands Unit Commands
 *  This chapter describes NVML operations that change the state of the unit. For S-class products.
 *  Each of these requires root/admin access. Non-admin users will see an NVML_ERROR_NO_PERMISSION
 *  error code when invoking any of these methods.
 *  @{
 */
/***************************************************************************************************/

/**
 * Set the LED state for the unit. The LED can be either green (0) or amber (1).
 *
 * For S-class products.
 * Requires root/admin permissions.
 *
 * This operation takes effect immediately.
 *
 *
 * <b>Current S-Class products don't provide unique LEDs for each unit. As such, both front
 * and back LEDs will be toggled in unison regardless of which unit is specified with this command.</b>
 *
 * See \ref nvmlLedColor_t for available colors.
 *
 * @param unit                                 The identifier of the target unit
 * @param color                                The target LED color
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the LED color has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit or \a color is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this is not an S-class product
 *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlUnitGetLedState()
 */
nvmlReturn_t DECLDIR nvmlUnitSetLedState(nvmlUnit_t unit, nvmlLedColor_t color);

/** @} */

/***************************************************************************************************/
/** @defgroup nvmlDeviceCommands Device Commands
 *  This chapter describes NVML operations that change the state of the device.
 *  Each of these requires root/admin access. Non-admin users will see an NVML_ERROR_NO_PERMISSION
 *  error code when invoking any of these methods.
 *  @{
 */
/***************************************************************************************************/

/**
 * Set the persistence mode for the device.
 *
 * For all products.
 * For Linux only.
 * Requires root/admin permissions.
 *
 * The persistence mode determines whether the GPU driver software is torn down after the last client
 * exits.
 *
 * This operation takes effect immediately. It is not persistent across reboots. After each reboot the
 * persistence mode is reset to "Disabled".
 *
 * See \ref nvmlEnableState_t for available modes.
 *
 * After calling this API with mode set to NVML_FEATURE_DISABLED on a device that has its own NUMA
 * memory, the given device handle will no longer be valid, and to continue to interact with this
 * device, a new handle should be obtained from one of the nvmlDeviceGetHandleBy*() APIs. This
 * limitation is currently only applicable to devices that have a coherent NVLink connection to
 * system memory.
 *
 * @param device                               The identifier of the target device
 * @param mode                                 The target persistence mode
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the persistence mode was set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceGetPersistenceMode()
 */
nvmlReturn_t DECLDIR nvmlDeviceSetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t mode);

/**
 * Set the compute mode for the device.
 *
 * For all products.
 * Requires root/admin permissions.
 *
 * The compute mode determines whether a GPU can be used for compute operations and whether it can
 * be shared across contexts.
 *
 * This operation takes effect immediately. Under Linux it is not persistent across reboots and
 * always resets to "Default". Under windows it is persistent.
 *
 * Under windows compute mode may only be set to DEFAULT when running in WDDM
 *
 * @note On MIG-enabled GPUs, compute mode would be set to DEFAULT and changing it is not supported.
 *
 * See \ref nvmlComputeMode_t for details on available compute modes.
 *
 * @param device                               The identifier of the target device
 * @param mode                                 The target compute mode
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the compute mode was set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceGetComputeMode()
 */
nvmlReturn_t DECLDIR nvmlDeviceSetComputeMode(nvmlDevice_t device, nvmlComputeMode_t mode);

/**
 * Set the ECC mode for the device.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Only applicable to devices with ECC.
 * Requires \a NVML_INFOROM_ECC version 1.0 or higher.
 * Requires root/admin permissions.
 *
 * The ECC mode determines whether the GPU enables its ECC support.
 *
 * This operation takes effect after the next reboot.
 *
 * See \ref nvmlEnableState_t for details on available modes.
 *
 * @param device                               The identifier of the target device
 * @param ecc                                  The target ECC mode
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the ECC mode was set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a ecc is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceGetEccMode()
 */
nvmlReturn_t DECLDIR nvmlDeviceSetEccMode(nvmlDevice_t device, nvmlEnableState_t ecc);

/**
 * Clear the ECC error and other memory error counts for the device.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Only applicable to devices with ECC.
 * Requires \a NVML_INFOROM_ECC version 2.0 or higher to clear aggregate location-based ECC counts.
 * Requires \a NVML_INFOROM_ECC version 1.0 or higher to clear all other ECC counts.
 * Requires root/admin permissions.
 * Requires ECC Mode to be enabled.
 *
 * Sets all of the specified ECC counters to 0, including both detailed and total counts.
 *
 * This operation takes effect immediately.
 *
 * See \ref nvmlMemoryErrorType_t for details on available counter types.
 *
 * @param device                               The identifier of the target device
 * @param counterType                          Flag that indicates which type of errors should be cleared.
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the error counts were cleared
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a counterType is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see
 *      - nvmlDeviceGetDetailedEccErrors()
 *      - nvmlDeviceGetTotalEccErrors()
 */
nvmlReturn_t DECLDIR nvmlDeviceClearEccErrorCounts(nvmlDevice_t device, nvmlEccCounterType_t counterType);

/**
 * Set the driver model for the device.
 *
 * For Fermi &tm; or newer fully supported devices.
 * For windows only.
 * Requires root/admin permissions.
 *
 * On Windows platforms the device driver can run in either WDDM or WDM (TCC) mode. If a display is attached
 * to the device it must run in WDDM mode.
 *
 * It is possible to force the change to WDM (TCC) while the display is still attached with a force flag (nvmlFlagForce).
 * This should only be done if the host is subsequently powered down and the display is detached from the device
 * before the next reboot.
 *
 * This operation takes effect after the next reboot.
 *
 * Windows driver model may only be set to WDDM when running in DEFAULT compute mode.
 *
 * Change driver model to WDDM is not supported when GPU doesn't support graphics acceleration or
 * will not support it after reboot. See \ref nvmlDeviceSetGpuOperationMode.
 *
 * See \ref nvmlDriverModel_t for details on available driver models.
 * See \ref nvmlFlagDefault and \ref nvmlFlagForce
 *
 * @param device                               The identifier of the target device
 * @param driverModel                          The target driver model
 * @param flags                                Flags that change the default behavior
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the driver model has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a driverModel is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the platform is not windows or the device does not support this feature
 *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceGetDriverModel()
 */
nvmlReturn_t DECLDIR nvmlDeviceSetDriverModel(nvmlDevice_t device, nvmlDriverModel_t driverModel, unsigned int flags);

typedef enum nvmlClockLimitId_enum {
    NVML_CLOCK_LIMIT_ID_RANGE_START = 0xffffff00,
    NVML_CLOCK_LIMIT_ID_TDP,
    NVML_CLOCK_LIMIT_ID_UNLIMITED
} nvmlClockLimitId_t;

/**
 * Set clocks that device will lock to.
 *
 * Sets the clocks that the device will be running at to the value in the range of minGpuClockMHz to maxGpuClockMHz.
 * Setting this will supercede application clock values and take effect regardless if a cuda app is running.
 * See /ref nvmlDeviceSetApplicationsClocks
 *
 * Can be used as a setting to request constant performance.
 *
 * This can be called with a pair of integer clock frequencies in MHz, or a pair of /ref nvmlClockLimitId_t values.
 * See the table below for valid combinations of these values.
 *
 * minGpuClock | maxGpuClock | Effect
 * ------------+-------------+--------------------------------------------------
 *     tdp     |     tdp     | Lock clock to TDP
 *  unlimited  |     tdp     | Upper bound is TDP but clock may drift below this
 *     tdp     |  unlimited  | Lower bound is TDP but clock may boost above this
 *  unlimited  |  unlimited  | Unlocked (== nvmlDeviceResetGpuLockedClocks)
 *
 * If one arg takes one of these values, the other must be one of these values as
 * well. Mixed numeric and symbolic calls return NVML_ERROR_INVALID_ARGUMENT.
 *
 * Requires root/admin permissions.
 *
 * After system reboot or driver reload applications clocks go back to their default value.
 * See \ref nvmlDeviceResetGpuLockedClocks.
 *
 * For Volta &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param minGpuClockMHz                       Requested minimum gpu clock in MHz
 * @param maxGpuClockMHz                       Requested maximum gpu clock in MHz
 *
 * @return
 *         - \ref NVML_SUCCESS                 if new settings were successfully set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a minGpuClockMHz and \a maxGpuClockMHz
 *                                                 is not a valid clock combination
 *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz);

/**
 * Resets the gpu clock to the default value
 *
 * This is the gpu clock that will be used after system reboot or driver reload.
 * Default values are idle clocks, but the current values can be changed using \ref nvmlDeviceSetApplicationsClocks.
 *
 * @see nvmlDeviceSetGpuLockedClocks
 *
 * For Volta &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 *
 * @return
 *         - \ref NVML_SUCCESS                 if new settings were successfully set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device);

/**
 * Set memory clocks that device will lock to.
 *
 * Sets the device's memory clocks to the value in the range of minMemClockMHz to maxMemClockMHz.
 * Setting this will supersede application clock values and take effect regardless of whether a cuda app is running.
 * See /ref nvmlDeviceSetApplicationsClocks
 *
 * Can be used as a setting to request constant performance.
 *
 * Requires root/admin permissions.
 *
 * After system reboot or driver reload applications clocks go back to their default value.
 * See \ref nvmlDeviceResetMemoryLockedClocks.
 *
 * For Ampere &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param minMemClockMHz                       Requested minimum memory clock in MHz
 * @param maxMemClockMHz                       Requested maximum memory clock in MHz
 *
 * @return
 *         - \ref NVML_SUCCESS                 if new settings were successfully set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a minGpuClockMHz and \a maxGpuClockMHz
 *                                                 is not a valid clock combination
 *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t device, unsigned int minMemClockMHz, unsigned int maxMemClockMHz);

/**
 * Resets the memory clock to the default value
 *
 * This is the memory clock that will be used after system reboot or driver reload.
 * Default values are idle clocks, but the current values can be changed using \ref nvmlDeviceSetApplicationsClocks.
 *
 * @see nvmlDeviceSetMemoryLockedClocks
 *
 * For Ampere &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 *
 * @return
 *         - \ref NVML_SUCCESS                 if new settings were successfully set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceResetMemoryLockedClocks(nvmlDevice_t device);

/**
 * Set clocks that applications will lock to.
 *
 * Sets the clocks that compute and graphics applications will be running at.
 * e.g. CUDA driver requests these clocks during context creation which means this property
 * defines clocks at which CUDA applications will be running unless some overspec event
 * occurs (e.g. over power, over thermal or external HW brake).
 *
 * Can be used as a setting to request constant performance.
 *
 * On Pascal and newer hardware, this will automatically disable automatic boosting of clocks.
 *
 * On K80 and newer Kepler and Maxwell GPUs, users desiring fixed performance should also call
 * \ref nvmlDeviceSetAutoBoostedClocksEnabled to prevent clocks from automatically boosting
 * above the clock value being set.
 *
 * For Kepler &tm; or newer non-GeForce fully supported devices and Maxwell or newer GeForce devices.
 * Requires root/admin permissions.
 *
 * See \ref nvmlDeviceGetSupportedMemoryClocks and \ref nvmlDeviceGetSupportedGraphicsClocks
 * for details on how to list available clocks combinations.
 *
 * After system reboot or driver reload applications clocks go back to their default value.
 * See \ref nvmlDeviceResetApplicationsClocks.
 *
 * @param device                               The identifier of the target device
 * @param memClockMHz                          Requested memory clock in MHz
 * @param graphicsClockMHz                     Requested graphics clock in MHz
 *
 * @return
 *         - \ref NVML_SUCCESS                 if new settings were successfully set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a memClockMHz and \a graphicsClockMHz
 *                                                 is not a valid clock combination
 *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceSetApplicationsClocks(nvmlDevice_t device, unsigned int memClockMHz, unsigned int graphicsClockMHz);

/**
 * Retrieves the frequency monitor fault status for the device.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Requires root user.
 *
 * See \ref nvmlClkMonStatus_t for details on decoding the status output.
 *
 * @param device                               The identifier of the target device
 * @param status                               Reference in which to return the clkmon fault status
 * 
 * @return 
 *         - \ref NVML_SUCCESS                 if \a status has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a status is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceGetClkMonStatus()
 */
nvmlReturn_t DECLDIR nvmlDeviceGetClkMonStatus(nvmlDevice_t device, nvmlClkMonStatus_t *status);

/**
 * Set new power limit of this device.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Requires root/admin permissions.
 *
 * See \ref nvmlDeviceGetPowerManagementLimitConstraints to check the allowed ranges of values.
 *
 * \note Limit is not persistent across reboots or driver unloads.
 * Enable persistent mode to prevent driver from unloading when no application is using the device.
 *
 * @param device                               The identifier of the target device
 * @param limit                                Power management limit in milliwatts to set
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a limit has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a defaultLimit is out of range
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceGetPowerManagementLimitConstraints
 * @see nvmlDeviceGetPowerManagementDefaultLimit
 */
nvmlReturn_t DECLDIR nvmlDeviceSetPowerManagementLimit(nvmlDevice_t device, unsigned int limit);

/**
 * Sets new GOM. See \a nvmlGpuOperationMode_t for details.
 *
 * For GK110 M-class and X-class Tesla &tm; products from the Kepler family.
 * Modes \ref NVML_GOM_LOW_DP and \ref NVML_GOM_ALL_ON are supported on fully supported GeForce products.
 * Not supported on Quadro &reg; and Tesla &tm; C-class products.
 * Requires root/admin permissions.
 *
 * Changing GOMs requires a reboot.
 * The reboot requirement might be removed in the future.
 *
 * Compute only GOMs don't support graphics acceleration. Under windows switching to these GOMs when
 * pending driver model is WDDM is not supported. See \ref nvmlDeviceSetDriverModel.
 *
 * @param device                               The identifier of the target device
 * @param mode                                 Target GOM
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a mode has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode incorrect
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support GOM or specific mode
 *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlGpuOperationMode_t
 * @see nvmlDeviceGetGpuOperationMode
 */
nvmlReturn_t DECLDIR nvmlDeviceSetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t mode);

/**
 * Changes the root/admin restructions on certain APIs. See \a nvmlRestrictedAPI_t for the list of supported APIs.
 * This method can be used by a root/admin user to give non-root/admin access to certain otherwise-restricted APIs.
 * The new setting lasts for the lifetime of the NVIDIA driver; it is not persistent. See \a nvmlDeviceGetAPIRestriction
 * to query the current restriction settings.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Requires root/admin permissions.
 *
 * @param device                               The identifier of the target device
 * @param apiType                              Target API type for this operation
 * @param isRestricted                         The target restriction
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a isRestricted has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a apiType incorrect
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support changing API restrictions or the device does not support
 *                                                 the feature that api restrictions are being set for (E.G. Enabling/disabling auto
 *                                                 boosted clocks is not supported by the device)
 *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlRestrictedAPI_t
 */
nvmlReturn_t DECLDIR nvmlDeviceSetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t isRestricted);

/**
 * @}
 */

/** @addtogroup nvmlAccountingStats
 *  @{
 */

/**
 * Enables or disables per process accounting.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Requires root/admin permissions.
 *
 * @note This setting is not persistent and will default to disabled after driver unloads.
 *       Enable persistence mode to be sure the setting doesn't switch off to disabled.
 *
 * @note Enabling accounting mode has no negative impact on the GPU performance.
 *
 * @note Disabling accounting clears all accounting pids information.
 *
 * @note On MIG-enabled GPUs, accounting mode would be set to DISABLED and changing it is not supported.
 *
 * See \ref nvmlDeviceGetAccountingMode
 * See \ref nvmlDeviceGetAccountingStats
 * See \ref nvmlDeviceClearAccountingPids
 *
 * @param device                               The identifier of the target device
 * @param mode                                 The target accounting mode
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the new mode has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a mode are invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceSetAccountingMode(nvmlDevice_t device, nvmlEnableState_t mode);

/**
 * Clears accounting information about all processes that have already terminated.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Requires root/admin permissions.
 *
 * See \ref nvmlDeviceGetAccountingMode
 * See \ref nvmlDeviceGetAccountingStats
 * See \ref nvmlDeviceSetAccountingMode
 *
 * @param device                               The identifier of the target device
 *
 * @return
 *         - \ref NVML_SUCCESS                 if accounting information has been cleared
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device are invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceClearAccountingPids(nvmlDevice_t device);

/** @} */

/***************************************************************************************************/
/** @defgroup NvLink NvLink Methods
 * This chapter describes methods that NVML can perform on NVLINK enabled devices.
 *  @{
 */
/***************************************************************************************************/

/**
 * Retrieves the state of the device's NvLink for the link specified
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param link                                 Specifies the NvLink link to be queried
 * @param isActive                             \a nvmlEnableState_t where NVML_FEATURE_ENABLED indicates that
 *                                             the link is active and NVML_FEATURE_DISABLED indicates it
 *                                             is inactive
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a isActive has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a link is invalid or \a isActive is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive);

/**
 * Retrieves the version of the device's NvLink for the link specified
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param link                                 Specifies the NvLink link to be queried
 * @param version                              Requested NvLink version
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a version has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a link is invalid or \a version is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link, unsigned int *version);

/**
 * Retrieves the requested capability from the device's NvLink for the link specified
 * Please refer to the \a nvmlNvLinkCapability_t structure for the specific caps that can be queried
 * The return value should be treated as a boolean.
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param link                                 Specifies the NvLink link to be queried
 * @param capability                           Specifies the \a nvmlNvLinkCapability_t to be queried
 * @param capResult                            A boolean for the queried capability indicating that feature is available
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a capResult has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a link, or \a capability is invalid or \a capResult is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link,
                                                   nvmlNvLinkCapability_t capability, unsigned int *capResult);

/**
 * Retrieves the PCI information for the remote node on a NvLink link
 * Note: pciSubSystemId is not filled in this function and is indeterminate
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param link                                 Specifies the NvLink link to be queried
 * @param pci                                  \a nvmlPciInfo_t of the remote node for the specified link
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a pci has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a link is invalid or \a pci is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci);

/**
 * Retrieves the specified error counter value
 * Please refer to \a nvmlNvLinkErrorCounter_t for error counters that are available
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param link                                 Specifies the NvLink link to be queried
 * @param counter                              Specifies the NvLink counter to be queried
 * @param counterValue                         Returned counter value
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a counter has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a link, or \a counter is invalid or \a counterValue is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device, unsigned int link,
                                                     nvmlNvLinkErrorCounter_t counter, unsigned long long *counterValue);

/**
 * Resets all error counters to zero
 * Please refer to \a nvmlNvLinkErrorCounter_t for the list of error counters that are reset
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param link                                 Specifies the NvLink link to be queried
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the reset is successful
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a link is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceResetNvLinkErrorCounters(nvmlDevice_t device, unsigned int link);

/**
 * Deprecated: Setting utilization counter control is no longer supported.
 *
 * Set the NVLINK utilization counter control information for the specified counter, 0 or 1.
 * Please refer to \a nvmlNvLinkUtilizationControl_t for the structure definition.  Performs a reset
 * of the counters if the reset parameter is non-zero.
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param counter                              Specifies the counter that should be set (0 or 1).
 * @param link                                 Specifies the NvLink link to be queried
 * @param control                              A reference to the \a nvmlNvLinkUtilizationControl_t to set
 * @param reset                                Resets the counters on set if non-zero
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the control has been set successfully
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a counter, \a link, or \a control is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceSetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter,
                                                           nvmlNvLinkUtilizationControl_t *control, unsigned int reset);

/**
 * Deprecated: Getting utilization counter control is no longer supported.
 *
 * Get the NVLINK utilization counter control information for the specified counter, 0 or 1.
 * Please refer to \a nvmlNvLinkUtilizationControl_t for the structure definition
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param counter                              Specifies the counter that should be set (0 or 1).
 * @param link                                 Specifies the NvLink link to be queried
 * @param control                              A reference to the \a nvmlNvLinkUtilizationControl_t to place information
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the control has been set successfully
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a counter, \a link, or \a control is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter,
                                                           nvmlNvLinkUtilizationControl_t *control);


/**
 * Deprecated: Use \ref nvmlDeviceGetFieldValues with NVML_FI_DEV_NVLINK_THROUGHPUT_* as field values instead.
 *
 * Retrieve the NVLINK utilization counter based on the current control for a specified counter.
 * In general it is good practice to use \a nvmlDeviceSetNvLinkUtilizationControl
 *  before reading the utilization counters as they have no default state
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param link                                 Specifies the NvLink link to be queried
 * @param counter                              Specifies the counter that should be read (0 or 1).
 * @param rxcounter                            Receive counter return value
 * @param txcounter                            Transmit counter return value
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a rxcounter and \a txcounter have been successfully set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a counter, or \a link is invalid or \a rxcounter or \a txcounter are NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter,
                                                           unsigned long long *rxcounter, unsigned long long *txcounter);

/**
 * Deprecated: Freezing NVLINK utilization counters is no longer supported.
 *
 * Freeze the NVLINK utilization counters
 * Both the receive and transmit counters are operated on by this function
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param link                                 Specifies the NvLink link to be queried
 * @param counter                              Specifies the counter that should be frozen (0 or 1).
 * @param freeze                               NVML_FEATURE_ENABLED = freeze the receive and transmit counters
 *                                             NVML_FEATURE_DISABLED = unfreeze the receive and transmit counters
 *
 * @return
 *         - \ref NVML_SUCCESS                 if counters were successfully frozen or unfrozen
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a link, \a counter, or \a freeze is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceFreezeNvLinkUtilizationCounter (nvmlDevice_t device, unsigned int link,
                                            unsigned int counter, nvmlEnableState_t freeze);

/**
 * Deprecated: Resetting NVLINK utilization counters is no longer supported.
 *
 * Reset the NVLINK utilization counters
 * Both the receive and transmit counters are operated on by this function
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param link                                 Specifies the NvLink link to be reset
 * @param counter                              Specifies the counter that should be reset (0 or 1)
 *
 * @return
 *         - \ref NVML_SUCCESS                 if counters were successfully reset
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a link, or \a counter is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceResetNvLinkUtilizationCounter (nvmlDevice_t device, unsigned int link, unsigned int counter);

/**
* Get the NVLink device type of the remote device connected over the given link.
*
* @param device                                The device handle of the target GPU
* @param link                                  The NVLink link index on the target GPU
* @param pNvLinkDeviceType                     Pointer in which the output remote device type is returned
*
* @return
*         - \ref NVML_SUCCESS                  if \a pNvLinkDeviceType has been set
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_NOT_SUPPORTED      if NVLink is not supported
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a device or \a link is invalid, or
*                                              \a pNvLinkDeviceType is NULL
*         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is
*                                              otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t DECLDIR nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t *pNvLinkDeviceType);

/** @} */

/***************************************************************************************************/
/** @defgroup nvmlEvents Event Handling Methods
 * This chapter describes methods that NVML can perform against each device to register and wait for
 * some event to occur.
 *  @{
 */
/***************************************************************************************************/

/**
 * Create an empty set of events.
 * Event set should be freed by \ref nvmlEventSetFree
 *
 * For Fermi &tm; or newer fully supported devices.
 * @param set                                  Reference in which to return the event handle
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the event has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a set is NULL
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlEventSetFree
 */
nvmlReturn_t DECLDIR nvmlEventSetCreate(nvmlEventSet_t *set);

/**
 * Starts recording of events on a specified devices and add the events to specified \ref nvmlEventSet_t
 *
 * For Fermi &tm; or newer fully supported devices.
 * Ecc events are available only on ECC enabled devices (see \ref nvmlDeviceGetTotalEccErrors)
 * Power capping events are available only on Power Management enabled devices (see \ref nvmlDeviceGetPowerManagementMode)
 *
 * For Linux only.
 *
 * \b IMPORTANT: Operations on \a set are not thread safe
 *
 * This call starts recording of events on specific device.
 * All events that occurred before this call are not recorded.
 * Checking if some event occurred can be done with \ref nvmlEventSetWait_v2
 *
 * If function reports NVML_ERROR_UNKNOWN, event set is in undefined state and should be freed.
 * If function reports NVML_ERROR_NOT_SUPPORTED, event set can still be used. None of the requested eventTypes
 *     are registered in that case.
 *
 * @param device                               The identifier of the target device
 * @param eventTypes                           Bitmask of \ref nvmlEventType to record
 * @param set                                  Set to which add new event types
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the event has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a eventTypes is invalid or \a set is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the platform does not support this feature or some of requested event types
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlEventType
 * @see nvmlDeviceGetSupportedEventTypes
 * @see nvmlEventSetWait
 * @see nvmlEventSetFree
 */
nvmlReturn_t DECLDIR nvmlDeviceRegisterEvents(nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set);

/**
 * Returns information about events supported on device
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * Events are not supported on Windows. So this function returns an empty mask in \a eventTypes on Windows.
 *
 * @param device                               The identifier of the target device
 * @param eventTypes                           Reference in which to return bitmask of supported events
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the eventTypes has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a eventType is NULL
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlEventType
 * @see nvmlDeviceRegisterEvents
 */
nvmlReturn_t DECLDIR nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device, unsigned long long *eventTypes);

/**
 * Waits on events and delivers events
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * If some events are ready to be delivered at the time of the call, function returns immediately.
 * If there are no events ready to be delivered, function sleeps till event arrives
 * but not longer than specified timeout. This function in certain conditions can return before
 * specified timeout passes (e.g. when interrupt arrives)
 *
 * On Windows, in case of xid error, the function returns the most recent xid error type seen by the system.
 * If there are multiple xid errors generated before nvmlEventSetWait is invoked then the last seen xid error
 * type is returned for all xid error events.
 *
 * On Linux, every xid error event would return the associated event data and other information if applicable.
 *
 * In MIG mode, if device handle is provided, the API reports all the events for the available instances,
 * only if the caller has appropriate privileges. In absence of required privileges, only the events which
 * affect all the instances (i.e. whole device) are reported.
 *
 * This API does not currently support per-instance event reporting using MIG device handles.
 *
 * @param set                                  Reference to set of events to wait on
 * @param data                                 Reference in which to return event data
 * @param timeoutms                            Maximum amount of wait time in milliseconds for registered event
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the data has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a data is NULL
 *         - \ref NVML_ERROR_TIMEOUT           if no event arrived in specified timeout or interrupt arrived
 *         - \ref NVML_ERROR_GPU_IS_LOST       if a GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlEventType
 * @see nvmlDeviceRegisterEvents
 */
nvmlReturn_t DECLDIR nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t * data, unsigned int timeoutms);

/**
 * Releases events in the set
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * @param set                                  Reference to events to be released
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the event has been successfully released
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlDeviceRegisterEvents
 */
nvmlReturn_t DECLDIR nvmlEventSetFree(nvmlEventSet_t set);

/** @} */

/***************************************************************************************************/
/** @defgroup nvmlZPI Drain states
 * This chapter describes methods that NVML can perform against each device to control their drain state
 * and recognition by NVML and NVIDIA kernel driver. These methods can be used with out-of-band tools to
 * power on/off GPUs, enable robust reset scenarios, etc.
 *  @{
 */
/***************************************************************************************************/

/**
 * Modify the drain state of a GPU.  This method forces a GPU to no longer accept new incoming requests.
 * Any new NVML process will no longer see this GPU.  Persistence mode for this GPU must be turned off before
 * this call is made.
 * Must be called as administrator.
 * For Linux only.
 *
 * For Pascal &tm; or newer fully supported devices.
 * Some Kepler devices supported.
 *
 * @param pciInfo                              The PCI address of the GPU drain state to be modified
 * @param newState                             The drain state that should be entered, see \ref nvmlEnableState_t
 *
 * @return
 *         - \ref NVML_SUCCESS                 if counters were successfully reset
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a nvmlIndex or \a newState is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_NO_PERMISSION     if the calling process has insufficient permissions to perform operation
 *         - \ref NVML_ERROR_IN_USE            if the device has persistence mode turned on
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceModifyDrainState (nvmlPciInfo_t *pciInfo, nvmlEnableState_t newState);

/**
 * Query the drain state of a GPU.  This method is used to check if a GPU is in a currently draining
 * state.
 * For Linux only.
 *
 * For Pascal &tm; or newer fully supported devices.
 * Some Kepler devices supported.
 *
 * @param pciInfo                              The PCI address of the GPU drain state to be queried
 * @param currentState                         The current drain state for this GPU, see \ref nvmlEnableState_t
 *
 * @return
 *         - \ref NVML_SUCCESS                 if counters were successfully reset
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a nvmlIndex or \a currentState is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceQueryDrainState (nvmlPciInfo_t *pciInfo, nvmlEnableState_t *currentState);

/**
 * This method will remove the specified GPU from the view of both NVML and the NVIDIA kernel driver
 * as long as no other processes are attached. If other processes are attached, this call will return
 * NVML_ERROR_IN_USE and the GPU will be returned to its original "draining" state. Note: the
 * only situation where a process can still be attached after nvmlDeviceModifyDrainState() is called
 * to initiate the draining state is if that process was using, and is still using, a GPU before the
 * call was made. Also note, persistence mode counts as an attachment to the GPU thus it must be disabled
 * prior to this call.
 *
 * For long-running NVML processes please note that this will change the enumeration of current GPUs.
 * For example, if there are four GPUs present and GPU1 is removed, the new enumeration will be 0-2.
 * Also, device handles after the removed GPU will not be valid and must be re-established.
 * Must be run as administrator.
 * For Linux only.
 *
 * For Pascal &tm; or newer fully supported devices.
 * Some Kepler devices supported.
 *
 * @param pciInfo                              The PCI address of the GPU to be removed
 * @param gpuState                             Whether the GPU is to be removed, from the OS
 *                                             see \ref nvmlDetachGpuState_t
 * @param linkState                            Requested upstream PCIe link state, see \ref nvmlPcieLinkState_t
 *
 * @return
 *         - \ref NVML_SUCCESS                 if counters were successfully reset
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a nvmlIndex is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref NVML_ERROR_IN_USE            if the device is still in use and cannot be removed
 */
nvmlReturn_t DECLDIR nvmlDeviceRemoveGpu_v2(nvmlPciInfo_t *pciInfo, nvmlDetachGpuState_t gpuState, nvmlPcieLinkState_t linkState);

/**
 * Request the OS and the NVIDIA kernel driver to rediscover a portion of the PCI subsystem looking for GPUs that
 * were previously removed. The portion of the PCI tree can be narrowed by specifying a domain, bus, and device.
 * If all are zeroes then the entire PCI tree will be searched.  Please note that for long-running NVML processes
 * the enumeration will change based on how many GPUs are discovered and where they are inserted in bus order.
 *
 * In addition, all newly discovered GPUs will be initialized and their ECC scrubbed which may take several seconds
 * per GPU. Also, all device handles are no longer guaranteed to be valid post discovery.
 *
 * Must be run as administrator.
 * For Linux only.
 *
 * For Pascal &tm; or newer fully supported devices.
 * Some Kepler devices supported.
 *
 * @param pciInfo                              The PCI tree to be searched.  Only the domain, bus, and device
 *                                             fields are used in this call.
 *
 * @return
 *         - \ref NVML_SUCCESS                 if counters were successfully reset
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a pciInfo is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the operating system does not support this feature
 *         - \ref NVML_ERROR_OPERATING_SYSTEM  if the operating system is denying this feature
 *         - \ref NVML_ERROR_NO_PERMISSION     if the calling process has insufficient permissions to perform operation
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceDiscoverGpus (nvmlPciInfo_t *pciInfo);

/** @} */

/***************************************************************************************************/
/** @defgroup nvmlFieldValueQueries Field Value Queries
 *  This chapter describes NVML operations that are associated with retrieving Field Values from NVML
 *  @{
 */
/***************************************************************************************************/

/**
 * Request values for a list of fields for a device. This API allows multiple fields to be queried at once.
 * If any of the underlying fieldIds are populated by the same driver call, the results for those field IDs
 * will be populated from a single call rather than making a driver call for each fieldId.
 *
 * @param device                               The device handle of the GPU to request field values for
 * @param valuesCount                          Number of entries in values that should be retrieved
 * @param values                               Array of \a valuesCount structures to hold field values.
 *                                             Each value's fieldId must be populated prior to this call
 *
 * @return
 *         - \ref NVML_SUCCESS                 if any values in \a values were populated. Note that you must
 *                                             check the nvmlReturn field of each value for each individual
 *                                             status
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a values is NULL
 */
nvmlReturn_t DECLDIR nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t *values);


/** @} */

/***************************************************************************************************/
/** @defgroup vGPU Enums, Constants and Structs
 *  @{
 */
/** @} */
/***************************************************************************************************/

/***************************************************************************************************/
/** @defgroup nvmlVirtualGpuQueries vGPU APIs
 * This chapter describes operations that are associated with NVIDIA vGPU Software products.
 *  @{
 */
/***************************************************************************************************/

/**
 * This method is used to get the virtualization mode corresponding to the GPU.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                    Identifier of the target device
 * @param pVirtualMode              Reference to virtualization mode. One of NVML_GPU_VIRTUALIZATION_?
 *
 * @return
 *         - \ref NVML_SUCCESS                  if \a pVirtualMode is fetched
 *         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a device is invalid or \a pVirtualMode is NULL
 *         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t *pVirtualMode);

/**
 * Queries if SR-IOV host operation is supported on a vGPU supported device.
 *
 * Checks whether SR-IOV host capability is supported by the device and the
 * driver, and indicates device is in SR-IOV mode if both of these conditions
 * are true.
 *
 * @param device                                The identifier of the target device
 * @param pHostVgpuMode                         Reference in which to return the current vGPU mode
 *
 * @return
 *         - \ref NVML_SUCCESS                  if device's vGPU mode has been successfully retrieved
 *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a device handle is 0 or \a pVgpuMode is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED      if \a device doesn't support this feature.
 *         - \ref NVML_ERROR_UNKNOWN            if any unexpected error occurred
 */
nvmlReturn_t DECLDIR nvmlDeviceGetHostVgpuMode(nvmlDevice_t device, nvmlHostVgpuMode_t *pHostVgpuMode);

/**
 * This method is used to set the virtualization mode corresponding to the GPU.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                    Identifier of the target device
 * @param virtualMode               virtualization mode. One of NVML_GPU_VIRTUALIZATION_?
 *
 * @return
 *         - \ref NVML_SUCCESS                  if \a pVirtualMode is set
 *         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a device is invalid or \a pVirtualMode is NULL
 *         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_NOT_SUPPORTED      if setting of virtualization mode is not supported.
 *         - \ref NVML_ERROR_NO_PERMISSION      if setting of virtualization mode is not allowed for this client.
 */
nvmlReturn_t DECLDIR nvmlDeviceSetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t virtualMode);

/**
 * Retrieve the vGPU Software licensable features.
 *
 * Identifies whether the system supports vGPU Software Licensing. If it does, return the list of licensable feature(s)
 * and their current license status.
 *
 * @param device                    Identifier of the target device
 * @param pGridLicensableFeatures   Pointer to structure in which vGPU software licensable features are returned
 *
 * @return
 *         - \ref NVML_SUCCESS                 if licensable features are successfully retrieved
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a pGridLicensableFeatures is NULL
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetGridLicensableFeatures_v4(nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures);

/**
 * Retrieves the current utilization and process ID
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * Reads recent utilization of GPU SM (3D/Compute), framebuffer, video encoder, and video decoder for processes running.
 * Utilization values are returned as an array of utilization sample structures in the caller-supplied buffer pointed at
 * by \a utilization. One utilization sample structure is returned per process running, that had some non-zero utilization
 * during the last sample period. It includes the CPU timestamp at which  the samples were recorded. Individual utilization values
 * are returned as "unsigned int" values.
 *
 * To read utilization values, first determine the size of buffer required to hold the samples by invoking the function with
 * \a utilization set to NULL. The caller should allocate a buffer of size
 * processSamplesCount * sizeof(nvmlProcessUtilizationSample_t). Invoke the function again with the allocated buffer passed
 * in \a utilization, and \a processSamplesCount set to the number of entries the buffer is sized for.
 *
 * On successful return, the function updates \a processSamplesCount with the number of process utilization sample
 * structures that were actually written. This may differ from a previously read value as instances are created or
 * destroyed.
 *
 * lastSeenTimeStamp represents the CPU timestamp in microseconds at which utilization samples were last read. Set it to 0
 * to read utilization based on all the samples maintained by the driver's internal sample buffer. Set lastSeenTimeStamp
 * to a timeStamp retrieved from a previous query to read utilization since the previous query.
 *
 * @note On MIG-enabled GPUs, querying process utilization is not currently supported.
 *
 * @param device                    The identifier of the target device
 * @param utilization               Pointer to caller-supplied buffer in which guest process utilization samples are returned
 * @param processSamplesCount       Pointer to caller-supplied array size, and returns number of processes running
 * @param lastSeenTimeStamp         Return only samples with timestamp greater than lastSeenTimeStamp.

 * @return
 *         - \ref NVML_SUCCESS                 if \a utilization has been populated
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a utilization is NULL, or \a samplingPeriodUs is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref NVML_ERROR_NOT_FOUND         if sample entries are not found
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetProcessUtilization(nvmlDevice_t device, nvmlProcessUtilizationSample_t *utilization,
                                              unsigned int *processSamplesCount, unsigned long long lastSeenTimeStamp);

/** @} */

/***************************************************************************************************/
/** @defgroup nvmlVgpu vGPU Management
 * @{
 *
 * This chapter describes APIs supporting NVIDIA vGPU.
 */
/***************************************************************************************************/

/**
 * Retrieve the supported vGPU types on a physical GPU (device).
 *
 * An array of supported vGPU types for the physical GPU indicated by \a device is returned in the caller-supplied buffer
 * pointed at by \a vgpuTypeIds. The element count of nvmlVgpuTypeId_t array is passed in \a vgpuCount, and \a vgpuCount
 * is used to return the number of vGPU types written to the buffer.
 *
 * If the supplied buffer is not large enough to accomodate the vGPU type array, the function returns
 * NVML_ERROR_INSUFFICIENT_SIZE, with the element count of nvmlVgpuTypeId_t array required in \a vgpuCount.
 * To query the number of vGPU types supported for the GPU, call this function with *vgpuCount = 0.
 * The code will return NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if no vGPU types are supported.
 *
 * @param device                   The identifier of the target device
 * @param vgpuCount                Pointer to caller-supplied array size, and returns number of vGPU types
 * @param vgpuTypeIds              Pointer to caller-supplied array in which to return list of vGPU types
 *
 * @return
 *         - \ref NVML_SUCCESS                      successful completion
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE      \a vgpuTypeIds buffer is too small, array element count is returned in \a vgpuCount
 *         - \ref NVML_ERROR_INVALID_ARGUMENT       if \a vgpuCount is NULL or \a device is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED          if vGPU is not supported by the device
 *         - \ref NVML_ERROR_UNKNOWN                on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetSupportedVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds);

/**
 * Retrieve the currently creatable vGPU types on a physical GPU (device).
 *
 * An array of creatable vGPU types for the physical GPU indicated by \a device is returned in the caller-supplied buffer
 * pointed at by \a vgpuTypeIds. The element count of nvmlVgpuTypeId_t array is passed in \a vgpuCount, and \a vgpuCount
 * is used to return the number of vGPU types written to the buffer.
 *
 * The creatable vGPU types for a device may differ over time, as there may be restrictions on what type of vGPU types
 * can concurrently run on a device.  For example, if only one vGPU type is allowed at a time on a device, then the creatable
 * list will be restricted to whatever vGPU type is already running on the device.
 *
 * If the supplied buffer is not large enough to accomodate the vGPU type array, the function returns
 * NVML_ERROR_INSUFFICIENT_SIZE, with the element count of nvmlVgpuTypeId_t array required in \a vgpuCount.
 * To query the number of vGPU types createable for the GPU, call this function with *vgpuCount = 0.
 * The code will return NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if no vGPU types are creatable.
 *
 * @param device                   The identifier of the target device
 * @param vgpuCount                Pointer to caller-supplied array size, and returns number of vGPU types
 * @param vgpuTypeIds              Pointer to caller-supplied array in which to return list of vGPU types
 *
 * @return
 *         - \ref NVML_SUCCESS                      successful completion
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE      \a vgpuTypeIds buffer is too small, array element count is returned in \a vgpuCount
 *         - \ref NVML_ERROR_INVALID_ARGUMENT       if \a vgpuCount is NULL
 *         - \ref NVML_ERROR_NOT_SUPPORTED          if vGPU is not supported by the device
 *         - \ref NVML_ERROR_UNKNOWN                on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetCreatableVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds);

/**
 * Retrieve the class of a vGPU type. It will not exceed 64 characters in length (including the NUL terminator).
 * See \ref nvmlConstants::NVML_DEVICE_NAME_BUFFER_SIZE.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param vgpuTypeClass            Pointer to string array to return class in
 * @param size                     Size of string
 *
 * @return
 *         - \ref NVML_SUCCESS                   successful completion
 *         - \ref NVML_ERROR_INVALID_ARGUMENT    if \a vgpuTypeId is invalid, or \a vgpuTypeClass is NULL
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE   if \a size is too small
 *         - \ref NVML_ERROR_UNKNOWN             on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuTypeGetClass(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeClass, unsigned int *size);

/**
 * Retrieve the vGPU type name.
 *
 * The name is an alphanumeric string that denotes a particular vGPU, e.g. GRID M60-2Q. It will not
 * exceed 64 characters in length (including the NUL terminator).  See \ref
 * nvmlConstants::NVML_DEVICE_NAME_BUFFER_SIZE.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param vgpuTypeName             Pointer to buffer to return name
 * @param size                     Size of buffer
 *
 * @return
 *         - \ref NVML_SUCCESS                 successful completion
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a name is NULL
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a size is too small
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuTypeGetName(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeName, unsigned int *size);

/**
 * Retrieve the GPU Instance Profile ID for the given vGPU type ID.
 * The API will return a valid GPU Instance Profile ID for the MIG capable vGPU types, else INVALID_GPU_INSTANCE_PROFILE_ID is
 * returned.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param gpuInstanceProfileId     GPU Instance Profile ID
 *
 * @return
 *         - \ref NVML_SUCCESS                 successful completion
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if \a device is not in vGPU Host virtualization mode
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a gpuInstanceProfileId is NULL
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuTypeGetGpuInstanceProfileId(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *gpuInstanceProfileId);

/**
 * Retrieve the device ID of a vGPU type.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param deviceID                 Device ID and vendor ID of the device contained in single 32 bit value
 * @param subsystemID              Subsytem ID and subsytem vendor ID of the device contained in single 32 bit value
 *
 * @return
 *         - \ref NVML_SUCCESS                 successful completion
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a deviceId or \a subsystemID are NULL
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuTypeGetDeviceID(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *deviceID, unsigned long long *subsystemID);

/**
 * Retrieve the vGPU framebuffer size in bytes.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param fbSize                   Pointer to framebuffer size in bytes
 *
 * @return
 *         - \ref NVML_SUCCESS                 successful completion
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a fbSize is NULL
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuTypeGetFramebufferSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *fbSize);

/**
 * Retrieve count of vGPU's supported display heads.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param numDisplayHeads          Pointer to number of display heads
 *
 * @return
 *         - \ref NVML_SUCCESS                 successful completion
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a numDisplayHeads is NULL
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuTypeGetNumDisplayHeads(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *numDisplayHeads);

/**
 * Retrieve vGPU display head's maximum supported resolution.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param displayIndex             Zero-based index of display head
 * @param xdim                     Pointer to maximum number of pixels in X dimension
 * @param ydim                     Pointer to maximum number of pixels in Y dimension
 *
 * @return
 *         - \ref NVML_SUCCESS                 successful completion
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a xdim or \a ydim are NULL, or \a displayIndex
 *                                             is out of range.
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuTypeGetResolution(nvmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int *xdim, unsigned int *ydim);

/**
 * Retrieve license requirements for a vGPU type
 *
 * The license type and version required to run the specified vGPU type is returned as an alphanumeric string, in the form
 * "<license name>,<version>", for example "GRID-Virtual-PC,2.0". If a vGPU is runnable with* more than one type of license,
 * the licenses are delimited by a semicolon, for example "GRID-Virtual-PC,2.0;GRID-Virtual-WS,2.0;GRID-Virtual-WS-Ext,2.0".
 *
 * The total length of the returned string will not exceed 128 characters, including the NUL terminator.
 * See \ref nvmlVgpuConstants::NVML_GRID_LICENSE_BUFFER_SIZE.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param vgpuTypeLicenseString    Pointer to buffer to return license info
 * @param size                     Size of \a vgpuTypeLicenseString buffer
 *
 * @return
 *         - \ref NVML_SUCCESS                 successful completion
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a vgpuTypeLicenseString is NULL
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a size is too small
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuTypeGetLicense(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeLicenseString, unsigned int size);

/**
 * Retrieve the static frame rate limit value of the vGPU type
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param frameRateLimit           Reference to return the frame rate limit value
 * @return
 *         - \ref NVML_SUCCESS                 successful completion
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if frame rate limiter is turned off for the vGPU type
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a frameRateLimit is NULL
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuTypeGetFrameRateLimit(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *frameRateLimit);

/**
 * Retrieve the maximum number of vGPU instances creatable on a device for given vGPU type
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                   The identifier of the target device
 * @param vgpuTypeId               Handle to vGPU type
 * @param vgpuInstanceCount        Pointer to get the max number of vGPU instances
 *                                 that can be created on a deicve for given vgpuTypeId
 * @return
 *         - \ref NVML_SUCCESS                 successful completion
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid or is not supported on target device,
 *                                             or \a vgpuInstanceCount is NULL
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuTypeGetMaxInstances(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCount);

/**
 * Retrieve the maximum number of vGPU instances supported per VM for given vGPU type
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param vgpuInstanceCountPerVm   Pointer to get the max number of vGPU instances supported per VM for given \a vgpuTypeId
 * @return
 *         - \ref NVML_SUCCESS                 successful completion
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a vgpuInstanceCountPerVm is NULL
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuTypeGetMaxInstancesPerVm(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCountPerVm);

/**
 * Retrieve the active vGPU instances on a device.
 *
 * An array of active vGPU instances is returned in the caller-supplied buffer pointed at by \a vgpuInstances. The
 * array elememt count is passed in \a vgpuCount, and \a vgpuCount is used to return the number of vGPU instances
 * written to the buffer.
 *
 * If the supplied buffer is not large enough to accomodate the vGPU instance array, the function returns
 * NVML_ERROR_INSUFFICIENT_SIZE, with the element count of nvmlVgpuInstance_t array required in \a vgpuCount.
 * To query the number of active vGPU instances, call this function with *vgpuCount = 0.  The code will return
 * NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if no vGPU Types are supported.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                   The identifier of the target device
 * @param vgpuCount                Pointer which passes in the array size as well as get
 *                                 back the number of types
 * @param vgpuInstances            Pointer to array in which to return list of vGPU instances
 *
 * @return
 *         - \ref NVML_SUCCESS                  successful completion
 *         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a device is invalid, or \a vgpuCount is NULL
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE  if \a size is too small
 *         - \ref NVML_ERROR_NOT_SUPPORTED      if vGPU is not supported by the device
 *         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetActiveVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuInstance_t *vgpuInstances);

/**
 * Retrieve the VM ID associated with a vGPU instance.
 *
 * The VM ID is returned as a string, not exceeding 80 characters in length (including the NUL terminator).
 * See \ref nvmlConstants::NVML_DEVICE_UUID_BUFFER_SIZE.
 *
 * The format of the VM ID varies by platform, and is indicated by the type identifier returned in \a vmIdType.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             Identifier of the target vGPU instance
 * @param vmId                     Pointer to caller-supplied buffer to hold VM ID
 * @param size                     Size of buffer in bytes
 * @param vmIdType                 Pointer to hold VM ID type
 *
 * @return
 *         - \ref NVML_SUCCESS                 successful completion
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vmId or \a vmIdType is NULL, or \a vgpuInstance is 0
 *         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a size is too small
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetVmID(nvmlVgpuInstance_t vgpuInstance, char *vmId, unsigned int size, nvmlVgpuVmIdType_t *vmIdType);

/**
 * Retrieve the UUID of a vGPU instance.
 *
 * The UUID is a globally unique identifier associated with the vGPU, and is returned as a 5-part hexadecimal string,
 * not exceeding 80 characters in length (including the NULL terminator).
 * See \ref nvmlConstants::NVML_DEVICE_UUID_BUFFER_SIZE.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             Identifier of the target vGPU instance
 * @param uuid                     Pointer to caller-supplied buffer to hold vGPU UUID
 * @param size                     Size of buffer in bytes
 *
 * @return
 *         - \ref NVML_SUCCESS                 successful completion
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a uuid is NULL
 *         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a size is too small
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetUUID(nvmlVgpuInstance_t vgpuInstance, char *uuid, unsigned int size);

/**
 * Retrieve the NVIDIA driver version installed in the VM associated with a vGPU.
 *
 * The version is returned as an alphanumeric string in the caller-supplied buffer \a version. The length of the version
 * string will not exceed 80 characters in length (including the NUL terminator).
 * See \ref nvmlConstants::NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE.
 *
 * nvmlVgpuInstanceGetVmDriverVersion() may be called at any time for a vGPU instance. The guest VM driver version is
 * returned as "Not Available" if no NVIDIA driver is installed in the VM, or the VM has not yet booted to the point where the
 * NVIDIA driver is loaded and initialized.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             Identifier of the target vGPU instance
 * @param version                  Caller-supplied buffer to return driver version string
 * @param length                   Size of \a version buffer
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a version has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0
 *         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetVmDriverVersion(nvmlVgpuInstance_t vgpuInstance, char* version, unsigned int length);

/**
 * Retrieve the framebuffer usage in bytes.
 *
 * Framebuffer usage is the amont of vGPU framebuffer memory that is currently in use by the VM.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             The identifier of the target instance
 * @param fbUsage                  Pointer to framebuffer usage in bytes
 *
 * @return
 *         - \ref NVML_SUCCESS                 successful completion
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a fbUsage is NULL
 *         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetFbUsage(nvmlVgpuInstance_t vgpuInstance, unsigned long long *fbUsage);

/**
 * @deprecated Use \ref nvmlVgpuInstanceGetLicenseInfo_v2.
 *
 * Retrieve the current licensing state of the vGPU instance.
 *
 * If the vGPU is currently licensed, \a licensed is set to 1, otherwise it is set to 0.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             Identifier of the target vGPU instance
 * @param licensed                 Reference to return the licensing status
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a licensed has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a licensed is NULL
 *         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetLicenseStatus(nvmlVgpuInstance_t vgpuInstance, unsigned int *licensed);

/**
 * Retrieve the vGPU type of a vGPU instance.
 *
 * Returns the vGPU type ID of vgpu assigned to the vGPU instance.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             Identifier of the target vGPU instance
 * @param vgpuTypeId               Reference to return the vgpuTypeId
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a vgpuTypeId has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a vgpuTypeId is NULL
 *         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetType(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuTypeId_t *vgpuTypeId);

/**
 * Retrieve the frame rate limit set for the vGPU instance.
 *
 * Returns the value of the frame rate limit set for the vGPU instance
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             Identifier of the target vGPU instance
 * @param frameRateLimit           Reference to return the frame rate limit
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a frameRateLimit has been set
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if frame rate limiter is turned off for the vGPU type
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a frameRateLimit is NULL
 *         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetFrameRateLimit(nvmlVgpuInstance_t vgpuInstance, unsigned int *frameRateLimit);

/**
 * Retrieve the current ECC mode of vGPU instance.
 *
 * @param vgpuInstance            The identifier of the target vGPU instance
 * @param eccMode                 Reference in which to return the current ECC mode
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the vgpuInstance's ECC mode has been successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a mode is NULL
 *         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the vGPU doesn't support this feature
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetEccMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *eccMode);

/**
 * Retrieve the encoder capacity of a vGPU instance, as a percentage of maximum encoder capacity with valid values in the range 0-100.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             Identifier of the target vGPU instance
 * @param encoderCapacity          Reference to an unsigned int for the encoder capacity
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a encoderCapacity has been retrived
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a encoderQueryType is invalid
 *         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int *encoderCapacity);

/**
 * Set the encoder capacity of a vGPU instance, as a percentage of maximum encoder capacity with valid values in the range 0-100.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             Identifier of the target vGPU instance
 * @param encoderCapacity          Unsigned int for the encoder capacity value
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a encoderCapacity has been set
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a encoderCapacity is out of range of 0-100.
 *         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceSetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int  encoderCapacity);

/**
 * Retrieves the current encoder statistics of a vGPU Instance
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param vgpuInstance                      Identifier of the target vGPU instance
 * @param sessionCount                      Reference to an unsigned int for count of active encoder sessions
 * @param averageFps                        Reference to an unsigned int for trailing average FPS of all active sessions
 * @param averageLatency                    Reference to an unsigned int for encode latency in microseconds
 *
 * @return
 *         - \ref NVML_SUCCESS                  if \a sessionCount, \a averageFps and \a averageLatency is fetched
 *         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a sessionCount , or \a averageFps or \a averageLatency is NULL
 *                                              or \a vgpuInstance is 0.
 *         - \ref NVML_ERROR_NOT_FOUND          if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetEncoderStats(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount,
                                                     unsigned int *averageFps, unsigned int *averageLatency);

/**
 * Retrieves information about all active encoder sessions on a vGPU Instance.
 *
 * An array of active encoder sessions is returned in the caller-supplied buffer pointed at by \a sessionInfo. The
 * array element count is passed in \a sessionCount, and \a sessionCount is used to return the number of sessions
 * written to the buffer.
 *
 * If the supplied buffer is not large enough to accomodate the active session array, the function returns
 * NVML_ERROR_INSUFFICIENT_SIZE, with the element count of nvmlEncoderSessionInfo_t array required in \a sessionCount.
 * To query the number of active encoder sessions, call this function with *sessionCount = 0. The code will return
 * NVML_SUCCESS with number of active encoder sessions updated in *sessionCount.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param vgpuInstance                      Identifier of the target vGPU instance
 * @param sessionCount                      Reference to caller supplied array size, and returns
 *                                          the number of sessions.
 * @param sessionInfo                       Reference to caller supplied array in which the list
 *                                          of session information us returned.
 *
 * @return
 *         - \ref NVML_SUCCESS                  if \a sessionInfo is fetched
 *         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE  if \a sessionCount is too small, array element count is
                                                returned in \a sessionCount
 *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a sessionCount is NULL, or \a vgpuInstance is 0.
 *         - \ref NVML_ERROR_NOT_FOUND          if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetEncoderSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfo);

/**
* Retrieves the active frame buffer capture sessions statistics of a vGPU Instance
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param vgpuInstance                      Identifier of the target vGPU instance
* @param fbcStats                          Reference to nvmlFBCStats_t structure contianing NvFBC stats
*
* @return
*         - \ref NVML_SUCCESS                  if \a fbcStats is fetched
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a vgpuInstance is 0, or \a fbcStats is NULL
*         - \ref NVML_ERROR_NOT_FOUND          if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetFBCStats(nvmlVgpuInstance_t vgpuInstance, nvmlFBCStats_t *fbcStats);

/**
* Retrieves information about active frame buffer capture sessions on a vGPU Instance.
*
* An array of active FBC sessions is returned in the caller-supplied buffer pointed at by \a sessionInfo. The
* array element count is passed in \a sessionCount, and \a sessionCount is used to return the number of sessions
* written to the buffer.
*
* If the supplied buffer is not large enough to accomodate the active session array, the function returns
* NVML_ERROR_INSUFFICIENT_SIZE, with the element count of nvmlFBCSessionInfo_t array required in \a sessionCount.
* To query the number of active FBC sessions, call this function with *sessionCount = 0.  The code will return
* NVML_SUCCESS with number of active FBC sessions updated in *sessionCount.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @note hResolution, vResolution, averageFPS and averageLatency data for a FBC session returned in \a sessionInfo may
*       be zero if there are no new frames captured since the session started.
*
* @param vgpuInstance                      Identifier of the target vGPU instance
* @param sessionCount                      Reference to caller supplied array size, and returns the number of sessions.
* @param sessionInfo                       Reference in which to return the session information
*
* @return
*         - \ref NVML_SUCCESS                  if \a sessionInfo is fetched
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a vgpuInstance is 0, or \a sessionCount is NULL.
*         - \ref NVML_ERROR_NOT_FOUND          if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE  if \a sessionCount is too small, array element count is returned in \a sessionCount
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetFBCSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo);

/**
* Retrieve the GPU Instance ID for the given vGPU Instance.
* The API will return a valid GPU Instance ID for MIG backed vGPU Instance, else INVALID_GPU_INSTANCE_ID is returned.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuInstance                      Identifier of the target vGPU instance
* @param gpuInstanceId                     GPU Instance ID
*
* @return
*         - \ref NVML_SUCCESS                  successful completion
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a vgpuInstance is 0, or \a gpuInstanceId is NULL.
*         - \ref NVML_ERROR_NOT_FOUND          if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance, unsigned int *gpuInstanceId);

/**
* Retrieves the PCI Id of the given vGPU Instance i.e. the PCI Id of the GPU as seen inside the VM.
*
* The vGPU PCI id is returned as "00000000:00:00.0" if NVIDIA driver is not installed on the vGPU instance.
*
* @param vgpuInstance                         Identifier of the target vGPU instance
* @param vgpuPciId                            Caller-supplied buffer to return vGPU PCI Id string
* @param length                               Size of the vgpuPciId buffer
*
* @return
*         - \ref NVML_SUCCESS                 if vGPU PCI Id is sucessfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a vgpuPciId is NULL
*         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_DRIVER_NOT_LOADED if NVIDIA driver is not running on the vGPU instance
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small, \a length is set to required length
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetGpuPciId(nvmlVgpuInstance_t vgpuInstance, char *vgpuPciId, unsigned int *length);

/** @} */

/***************************************************************************************************/
/** @defgroup nvml vGPU Migration
 * This chapter describes operations that are associated with vGPU Migration.
 *  @{
 */
/***************************************************************************************************/

/**
 * Structure representing range of vGPU versions.
 */
typedef struct nvmlVgpuVersion_st
{
    unsigned int minVersion; //!< Minimum vGPU version.
    unsigned int maxVersion; //!< Maximum vGPU version.
} nvmlVgpuVersion_t;

/**
 * vGPU metadata structure.
 */
typedef struct nvmlVgpuMetadata_st
{
    unsigned int             version;                                                    //!< Current version of the structure
    unsigned int             revision;                                                   //!< Current revision of the structure
    nvmlVgpuGuestInfoState_t guestInfoState;                                             //!< Current state of Guest-dependent fields
    char                     guestDriverVersion[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE]; //!< Version of driver installed in guest
    char                     hostDriverVersion[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];  //!< Version of driver installed in host
    unsigned int             reserved[6];                                                //!< Reserved for internal use
    unsigned int             vgpuVirtualizationCaps;                                     //!< vGPU virtualizaion capabilities bitfileld
    unsigned int             guestVgpuVersion;                                           //!< vGPU version of guest driver
    unsigned int             opaqueDataSize;                                             //!< Size of opaque data field in bytes
    char                     opaqueData[4];                                              //!< Opaque data
} nvmlVgpuMetadata_t;

/**
 * Physical GPU metadata structure
 */
typedef struct nvmlVgpuPgpuMetadata_st
{
    unsigned int            version;                                                    //!< Current version of the structure
    unsigned int            revision;                                                   //!< Current revision of the structure
    char                    hostDriverVersion[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];  //!< Host driver version
    unsigned int            pgpuVirtualizationCaps;                                     //!< Pgpu virtualizaion capabilities bitfileld
    unsigned int            reserved[5];                                                //!< Reserved for internal use
    nvmlVgpuVersion_t       hostSupportedVgpuRange;                                     //!< vGPU version range supported by host driver
    unsigned int            opaqueDataSize;                                             //!< Size of opaque data field in bytes
    char                    opaqueData[4];                                              //!< Opaque data
} nvmlVgpuPgpuMetadata_t;

/**
 * vGPU VM compatibility codes
 */
typedef enum nvmlVgpuVmCompatibility_enum
{
    NVML_VGPU_VM_COMPATIBILITY_NONE         = 0x0,    //!< vGPU is not runnable
    NVML_VGPU_VM_COMPATIBILITY_COLD         = 0x1,    //!< vGPU is runnable from a cold / powered-off state (ACPI S5)
    NVML_VGPU_VM_COMPATIBILITY_HIBERNATE    = 0x2,    //!< vGPU is runnable from a hibernated state (ACPI S4)
    NVML_VGPU_VM_COMPATIBILITY_SLEEP        = 0x4,    //!< vGPU is runnable from a sleeped state (ACPI S3)
    NVML_VGPU_VM_COMPATIBILITY_LIVE         = 0x8     //!< vGPU is runnable from a live/paused (ACPI S0)
} nvmlVgpuVmCompatibility_t;

/**
 *  vGPU-pGPU compatibility limit codes
 */
typedef enum nvmlVgpuPgpuCompatibilityLimitCode_enum
{
    NVML_VGPU_COMPATIBILITY_LIMIT_NONE          = 0x0,           //!< Compatibility is not limited.
    NVML_VGPU_COMPATIBILITY_LIMIT_HOST_DRIVER   = 0x1,           //!< ompatibility is limited by host driver version.
    NVML_VGPU_COMPATIBILITY_LIMIT_GUEST_DRIVER  = 0x2,           //!< Compatibility is limited by guest driver version.
    NVML_VGPU_COMPATIBILITY_LIMIT_GPU           = 0x4,           //!< Compatibility is limited by GPU hardware.
    NVML_VGPU_COMPATIBILITY_LIMIT_OTHER         = 0x80000000     //!< Compatibility is limited by an undefined factor.
} nvmlVgpuPgpuCompatibilityLimitCode_t;

/**
 * vGPU-pGPU compatibility structure
 */
typedef struct nvmlVgpuPgpuCompatibility_st
{
    nvmlVgpuVmCompatibility_t               vgpuVmCompatibility;    //!< Compatibility of vGPU VM. See \ref nvmlVgpuVmCompatibility_t
    nvmlVgpuPgpuCompatibilityLimitCode_t    compatibilityLimitCode; //!< Limiting factor for vGPU-pGPU compatibility. See \ref nvmlVgpuPgpuCompatibilityLimitCode_t
} nvmlVgpuPgpuCompatibility_t;

/**
 * Returns vGPU metadata structure for a running vGPU. The structure contains information about the vGPU and its associated VM
 * such as the currently installed NVIDIA guest driver version, together with host driver version and an opaque data section
 * containing internal state.
 *
 * nvmlVgpuInstanceGetMetadata() may be called at any time for a vGPU instance. Some fields in the returned structure are
 * dependent on information obtained from the guest VM, which may not yet have reached a state where that information
 * is available. The current state of these dependent fields is reflected in the info structure's \ref nvmlVgpuGuestInfoState_t field.
 *
 * The VMM may choose to read and save the vGPU's VM info as persistent metadata associated with the VM, and provide
 * it to Virtual GPU Manager when creating a vGPU for subsequent instances of the VM.
 *
 * The caller passes in a buffer via \a vgpuMetadata, with the size of the buffer in \a bufferSize. If the vGPU Metadata structure
 * is too large to fit in the supplied buffer, the function returns NVML_ERROR_INSUFFICIENT_SIZE with the size needed
 * in \a bufferSize.
 *
 * @param vgpuInstance             vGPU instance handle
 * @param vgpuMetadata             Pointer to caller-supplied buffer into which vGPU metadata is written
 * @param bufferSize               Size of vgpuMetadata buffer
 *
 * @return
 *         - \ref NVML_SUCCESS                   vGPU metadata structure was successfully returned
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE   vgpuMetadata buffer is too small, required size is returned in \a bufferSize
 *         - \ref NVML_ERROR_INVALID_ARGUMENT    if \a bufferSize is NULL or \a vgpuInstance is 0; if \a vgpuMetadata is NULL and the value of \a bufferSize is not 0.
 *         - \ref NVML_ERROR_NOT_FOUND           if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref NVML_ERROR_UNKNOWN             on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetMetadata(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t *vgpuMetadata, unsigned int *bufferSize);

/**
 * Returns a vGPU metadata structure for the physical GPU indicated by \a device. The structure contains information about
 * the GPU and the currently installed NVIDIA host driver version that's controlling it, together with an opaque data section
 * containing internal state.
 *
 * The caller passes in a buffer via \a pgpuMetadata, with the size of the buffer in \a bufferSize. If the \a pgpuMetadata
 * structure is too large to fit in the supplied buffer, the function returns NVML_ERROR_INSUFFICIENT_SIZE with the size needed
 * in \a bufferSize.
 *
 * @param device                The identifier of the target device
 * @param pgpuMetadata          Pointer to caller-supplied buffer into which \a pgpuMetadata is written
 * @param bufferSize            Pointer to size of \a pgpuMetadata buffer
 *
 * @return
 *         - \ref NVML_SUCCESS                   GPU metadata structure was successfully returned
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE   pgpuMetadata buffer is too small, required size is returned in \a bufferSize
 *         - \ref NVML_ERROR_INVALID_ARGUMENT    if \a bufferSize is NULL or \a device is invalid; if \a pgpuMetadata is NULL and the value of \a bufferSize is not 0.
 *         - \ref NVML_ERROR_NOT_SUPPORTED       vGPU is not supported by the system
 *         - \ref NVML_ERROR_UNKNOWN             on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetVgpuMetadata(nvmlDevice_t device, nvmlVgpuPgpuMetadata_t *pgpuMetadata, unsigned int *bufferSize);

/**
 * Takes a vGPU instance metadata structure read from \ref nvmlVgpuInstanceGetMetadata(), and a vGPU metadata structure for a
 * physical GPU read from \ref nvmlDeviceGetVgpuMetadata(), and returns compatibility information of the vGPU instance and the
 * physical GPU.
 *
 * The caller passes in a buffer via \a compatibilityInfo, into which a compatibility information structure is written. The
 * structure defines the states in which the vGPU / VM may be booted on the physical GPU. If the vGPU / VM compatibility
 * with the physical GPU is limited, a limit code indicates the factor limiting compability.
 * (see \ref nvmlVgpuPgpuCompatibilityLimitCode_t for details).
 *
 * Note: vGPU compatibility does not take into account dynamic capacity conditions that may limit a system's ability to
 *       boot a given vGPU or associated VM.
 *
 * @param vgpuMetadata          Pointer to caller-supplied vGPU metadata structure
 * @param pgpuMetadata          Pointer to caller-supplied GPU metadata structure
 * @param compatibilityInfo     Pointer to caller-supplied buffer to hold compatibility info
 *
 * @return
 *         - \ref NVML_SUCCESS                   vGPU metadata structure was successfully returned
 *         - \ref NVML_ERROR_INVALID_ARGUMENT    if \a vgpuMetadata or \a pgpuMetadata or \a bufferSize are NULL
 *         - \ref NVML_ERROR_UNKNOWN             on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlGetVgpuCompatibility(nvmlVgpuMetadata_t *vgpuMetadata, nvmlVgpuPgpuMetadata_t *pgpuMetadata, nvmlVgpuPgpuCompatibility_t *compatibilityInfo);

/**
 * Returns the properties of the physical GPU indicated by the device in an ascii-encoded string format.
 *
 * The caller passes in a buffer via \a pgpuMetadata, with the size of the buffer in \a bufferSize. If the
 * string is too large to fit in the supplied buffer, the function returns NVML_ERROR_INSUFFICIENT_SIZE with the size needed
 * in \a bufferSize.
 *
 * @param device                The identifier of the target device
 * @param pgpuMetadata          Pointer to caller-supplied buffer into which \a pgpuMetadata is written
 * @param bufferSize            Pointer to size of \a pgpuMetadata buffer
 *
 * @return
 *         - \ref NVML_SUCCESS                   GPU metadata structure was successfully returned
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE   \a pgpuMetadata buffer is too small, required size is returned in \a bufferSize
 *         - \ref NVML_ERROR_INVALID_ARGUMENT    if \a bufferSize is NULL or \a device is invalid; if \a pgpuMetadata is NULL and the value of \a bufferSize is not 0.
 *         - \ref NVML_ERROR_NOT_SUPPORTED       if vGPU is not supported by the system
 *         - \ref NVML_ERROR_UNKNOWN             on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetPgpuMetadataString(nvmlDevice_t device, char *pgpuMetadata, unsigned int *bufferSize);

/*
 * Virtual GPU (vGPU) version
 *
 * The NVIDIA vGPU Manager and the guest drivers are tagged with a range of supported vGPU versions. This determines the range of NVIDIA guest driver versions that
 * are compatible for vGPU feature support with a given NVIDIA vGPU Manager. For vGPU feature support, the range of supported versions for the NVIDIA vGPU Manager
 * and the guest driver must overlap. Otherwise, the guest driver fails to load in the VM.
 *
 * When the NVIDIA guest driver loads, either when the VM is booted or when the driver is installed or upgraded, a negotiation occurs between the guest driver
 * and the NVIDIA vGPU Manager to select the highest mutually compatible vGPU version. The negotiated vGPU version stays the same across VM migration.
 */

/**
 * Query the ranges of supported vGPU versions.
 *
 * This function gets the linear range of supported vGPU versions that is preset for the NVIDIA vGPU Manager and the range set by an administrator.
 * If the preset range has not been overridden by \ref nvmlSetVgpuVersion, both ranges are the same.
 *
 * The caller passes pointers to the following \ref nvmlVgpuVersion_t structures, into which the NVIDIA vGPU Manager writes the ranges:
 * 1. \a supported structure that represents the preset range of vGPU versions supported by the NVIDIA vGPU Manager.
 * 2. \a current structure that represents the range of supported vGPU versions set by an administrator. By default, this range is the same as the preset range.
 *
 * @param supported  Pointer to the structure in which the preset range of vGPU versions supported by the NVIDIA vGPU Manager is written
 * @param current    Pointer to the structure in which the range of supported vGPU versions set by an administrator is written
 *
 * @return
 * - \ref NVML_SUCCESS                 The vGPU version range structures were successfully obtained.
 * - \ref NVML_ERROR_NOT_SUPPORTED     The API is not supported.
 * - \ref NVML_ERROR_INVALID_ARGUMENT  The \a supported parameter or the \a current parameter is NULL.
 * - \ref NVML_ERROR_UNKNOWN           An error occurred while the data was being fetched.
 */
nvmlReturn_t DECLDIR nvmlGetVgpuVersion(nvmlVgpuVersion_t *supported, nvmlVgpuVersion_t *current);

/**
 * Override the preset range of vGPU versions supported by the NVIDIA vGPU Manager with a range set by an administrator.
 *
 * This function configures the NVIDIA vGPU Manager with a range of supported vGPU versions set by an administrator. This range must be a subset of the
 * preset range that the NVIDIA vGPU Manager supports. The custom range set by an administrator takes precedence over the preset range and is advertised to
 * the guest VM for negotiating the vGPU version. See \ref nvmlGetVgpuVersion for details of how to query the preset range of versions supported.
 *
 * This function takes a pointer to vGPU version range structure \ref nvmlVgpuVersion_t as input to override the preset vGPU version range that the NVIDIA vGPU Manager supports.
 *
 * After host system reboot or driver reload, the range of supported versions reverts to the range that is preset for the NVIDIA vGPU Manager.
 *
 * @note 1. The range set by the administrator must be a subset of the preset range that the NVIDIA vGPU Manager supports. Otherwise, an error is returned.
 *       2. If the range of supported guest driver versions does not overlap the range set by the administrator, the guest driver fails to load.
 *       3. If the range of supported guest driver versions overlaps the range set by the administrator, the guest driver will load with a negotiated
 *          vGPU version that is the maximum value in the overlapping range.
 *       4. No VMs must be running on the host when this function is called. If a VM is running on the host, the call to this function fails.
 *
 * @param vgpuVersion   Pointer to a caller-supplied range of supported vGPU versions.
 *
 * @return
 * - \ref NVML_SUCCESS                 The preset range of supported vGPU versions was successfully overridden.
 * - \ref NVML_ERROR_NOT_SUPPORTED     The API is not supported.
 * - \ref NVML_ERROR_IN_USE            The range was not overridden because a VM is running on the host.
 * - \ref NVML_ERROR_INVALID_ARGUMENT  The \a vgpuVersion parameter specifies a range that is outside the range supported by the NVIDIA vGPU Manager or if \a vgpuVersion is NULL.
 */
nvmlReturn_t DECLDIR nvmlSetVgpuVersion(nvmlVgpuVersion_t *vgpuVersion);

/** @} */

/***************************************************************************************************/
/** @defgroup nvmlUtil vGPU Utilization and Accounting
 * This chapter describes operations that are associated with vGPU Utilization and Accounting.
 *  @{
 */
/***************************************************************************************************/

/**
 * Retrieves current utilization for vGPUs on a physical GPU (device).
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * Reads recent utilization of GPU SM (3D/Compute), framebuffer, video encoder, and video decoder for vGPU instances running
 * on a device. Utilization values are returned as an array of utilization sample structures in the caller-supplied buffer
 * pointed at by \a utilizationSamples. One utilization sample structure is returned per vGPU instance, and includes the
 * CPU timestamp at which the samples were recorded. Individual utilization values are returned as "unsigned int" values
 * in nvmlValue_t unions. The function sets the caller-supplied \a sampleValType to NVML_VALUE_TYPE_UNSIGNED_INT to
 * indicate the returned value type.
 *
 * To read utilization values, first determine the size of buffer required to hold the samples by invoking the function with
 * \a utilizationSamples set to NULL. The function will return NVML_ERROR_INSUFFICIENT_SIZE, with the current vGPU instance
 * count in \a vgpuInstanceSamplesCount, or NVML_SUCCESS if the current vGPU instance count is zero. The caller should allocate
 * a buffer of size vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t). Invoke the function again with
 * the allocated buffer passed in \a utilizationSamples, and \a vgpuInstanceSamplesCount set to the number of entries the
 * buffer is sized for.
 *
 * On successful return, the function updates \a vgpuInstanceSampleCount with the number of vGPU utilization sample
 * structures that were actually written. This may differ from a previously read value as vGPU instances are created or
 * destroyed.
 *
 * lastSeenTimeStamp represents the CPU timestamp in microseconds at which utilization samples were last read. Set it to 0
 * to read utilization based on all the samples maintained by the driver's internal sample buffer. Set lastSeenTimeStamp
 * to a timeStamp retrieved from a previous query to read utilization since the previous query.
 *
 * @param device                        The identifier for the target device
 * @param lastSeenTimeStamp             Return only samples with timestamp greater than lastSeenTimeStamp.
 * @param sampleValType                 Pointer to caller-supplied buffer to hold the type of returned sample values
 * @param vgpuInstanceSamplesCount      Pointer to caller-supplied array size, and returns number of vGPU instances
 * @param utilizationSamples            Pointer to caller-supplied buffer in which vGPU utilization samples are returned

 * @return
 *         - \ref NVML_SUCCESS                 if utilization samples are successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a vgpuInstanceSamplesCount or \a sampleValType is
 *                                             NULL, or a sample count of 0 is passed with a non-NULL \a utilizationSamples
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if supplied \a vgpuInstanceSamplesCount is too small to return samples for all
 *                                             vGPU instances currently executing on the device
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if vGPU is not supported by the device
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_NOT_FOUND         if sample entries are not found
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetVgpuUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp,
                                                  nvmlValueType_t *sampleValType, unsigned int *vgpuInstanceSamplesCount,
                                                  nvmlVgpuInstanceUtilizationSample_t *utilizationSamples);

/**
 * Retrieves current utilization for processes running on vGPUs on a physical GPU (device).
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * Reads recent utilization of GPU SM (3D/Compute), framebuffer, video encoder, and video decoder for processes running on
 * vGPU instances active on a device. Utilization values are returned as an array of utilization sample structures in the
 * caller-supplied buffer pointed at by \a utilizationSamples. One utilization sample structure is returned per process running
 * on vGPU instances, that had some non-zero utilization during the last sample period. It includes the CPU timestamp at which
 * the samples were recorded. Individual utilization values are returned as "unsigned int" values.
 *
 * To read utilization values, first determine the size of buffer required to hold the samples by invoking the function with
 * \a utilizationSamples set to NULL. The function will return NVML_ERROR_INSUFFICIENT_SIZE, with the current vGPU instance
 * count in \a vgpuProcessSamplesCount. The caller should allocate a buffer of size
 * vgpuProcessSamplesCount * sizeof(nvmlVgpuProcessUtilizationSample_t). Invoke the function again with
 * the allocated buffer passed in \a utilizationSamples, and \a vgpuProcessSamplesCount set to the number of entries the
 * buffer is sized for.
 *
 * On successful return, the function updates \a vgpuSubProcessSampleCount with the number of vGPU sub process utilization sample
 * structures that were actually written. This may differ from a previously read value depending on the number of processes that are active
 * in any given sample period.
 *
 * lastSeenTimeStamp represents the CPU timestamp in microseconds at which utilization samples were last read. Set it to 0
 * to read utilization based on all the samples maintained by the driver's internal sample buffer. Set lastSeenTimeStamp
 * to a timeStamp retrieved from a previous query to read utilization since the previous query.
 *
 * @param device                        The identifier for the target device
 * @param lastSeenTimeStamp             Return only samples with timestamp greater than lastSeenTimeStamp.
 * @param vgpuProcessSamplesCount       Pointer to caller-supplied array size, and returns number of processes running on vGPU instances
 * @param utilizationSamples            Pointer to caller-supplied buffer in which vGPU sub process utilization samples are returned

 * @return
 *         - \ref NVML_SUCCESS                 if utilization samples are successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a vgpuProcessSamplesCount or a sample count of 0 is
 *                                             passed with a non-NULL \a utilizationSamples
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if supplied \a vgpuProcessSamplesCount is too small to return samples for all
 *                                             vGPU instances currently executing on the device
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if vGPU is not supported by the device
 *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref NVML_ERROR_NOT_FOUND         if sample entries are not found
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetVgpuProcessUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp,
                                                         unsigned int *vgpuProcessSamplesCount,
                                                         nvmlVgpuProcessUtilizationSample_t *utilizationSamples);
/**
 * Queries the state of per process accounting mode on vGPU.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param vgpuInstance            The identifier of the target vGPU instance
 * @param mode                    Reference in which to return the current accounting mode
 *
 * @return
 *         - \ref NVML_SUCCESS                 if the mode has been successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a mode is NULL
 *         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the vGPU doesn't support this feature
 *         - \ref NVML_ERROR_DRIVER_NOT_LOADED if NVIDIA driver is not running on the vGPU instance
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetAccountingMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *mode);

/**
 * Queries list of processes running on vGPU that can be queried for accounting stats. The list of processes
 * returned can be in running or terminated state.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * To just query the maximum number of processes that can be queried, call this function with *count = 0 and
 * pids=NULL. The return code will be NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if list is empty.
 *
 * For more details see \ref nvmlVgpuInstanceGetAccountingStats.
 *
 * @note In case of PID collision some processes might not be accessible before the circular buffer is full.
 *
 * @param vgpuInstance            The identifier of the target vGPU instance
 * @param count                   Reference in which to provide the \a pids array size, and
 *                                to return the number of elements ready to be queried
 * @param pids                    Reference in which to return list of process ids
 *
 * @return
 *         - \ref NVML_SUCCESS                 if pids were successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a count is NULL
 *         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the vGPU doesn't support this feature or accounting mode is disabled
 *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a count is too small (\a count is set to expected value)
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see nvmlVgpuInstanceGetAccountingPids
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetAccountingPids(nvmlVgpuInstance_t vgpuInstance, unsigned int *count, unsigned int *pids);

/**
 * Queries process's accounting stats.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * Accounting stats capture GPU utilization and other statistics across the lifetime of a process, and
 * can be queried during life time of the process or after its termination.
 * The time field in \ref nvmlAccountingStats_t is reported as 0 during the lifetime of the process and
 * updated to actual running time after its termination.
 * Accounting stats are kept in a circular buffer, newly created processes overwrite information about old
 * processes.
 *
 * See \ref nvmlAccountingStats_t for description of each returned metric.
 * List of processes that can be queried can be retrieved from \ref nvmlVgpuInstanceGetAccountingPids.
 *
 * @note Accounting Mode needs to be on. See \ref nvmlVgpuInstanceGetAccountingMode.
 * @note Only compute and graphics applications stats can be queried. Monitoring applications stats can't be
 *         queried since they don't contribute to GPU utilization.
 * @note In case of pid collision stats of only the latest process (that terminated last) will be reported
 *
 * @param vgpuInstance            The identifier of the target vGPU instance
 * @param pid                     Process Id of the target process to query stats for
 * @param stats                   Reference in which to return the process's accounting stats
 *
 * @return
 *         - \ref NVML_SUCCESS                 if stats have been successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a stats is NULL
 *         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *                                             or \a stats is not found
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the vGPU doesn't support this feature or accounting mode is disabled
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetAccountingStats(nvmlVgpuInstance_t vgpuInstance, unsigned int pid, nvmlAccountingStats_t *stats);

/**
 * Clears accounting information of the vGPU instance that have already terminated.
 *
 * For Maxwell &tm; or newer fully supported devices.
 * Requires root/admin permissions.
 *
 * @note Accounting Mode needs to be on. See \ref nvmlVgpuInstanceGetAccountingMode.
 * @note Only compute and graphics applications stats are reported and can be cleared since monitoring applications
 *         stats don't contribute to GPU utilization.
 *
 * @param vgpuInstance            The identifier of the target vGPU instance
 *
 * @return
 *         - \ref NVML_SUCCESS                 if accounting information has been cleared
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is invalid
 *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if the vGPU doesn't support this feature or accounting mode is disabled
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceClearAccountingPids(nvmlVgpuInstance_t vgpuInstance);

/**
 * Query the license information of the vGPU instance.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param vgpuInstance              Identifier of the target vGPU instance
 * @param licenseInfo               Pointer to vGPU license information structure
 *
 * @return
 *         - \ref NVML_SUCCESS                 if information is successfully retrieved
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a licenseInfo is NULL
 *         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref NVML_ERROR_DRIVER_NOT_LOADED if NVIDIA driver is not running on the vGPU instance
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetLicenseInfo_v2(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuLicenseInfo_t *licenseInfo);
/** @} */

/***************************************************************************************************/
/** @defgroup nvmlExcludedGpuQueries Excluded GPU Queries
 * This chapter describes NVML operations that are associated with excluded GPUs.
 *  @{
 */
/***************************************************************************************************/

/**
 * Excluded GPU device information
 **/
typedef struct nvmlExcludedDeviceInfo_st
{
    nvmlPciInfo_t pciInfo;                   //!< The PCI information for the excluded GPU
    char uuid[NVML_DEVICE_UUID_BUFFER_SIZE]; //!< The ASCII string UUID for the excluded GPU
} nvmlExcludedDeviceInfo_t;

 /**
 * Retrieves the number of excluded GPU devices in the system.
 *
 * For all products.
 *
 * @param deviceCount                          Reference in which to return the number of excluded devices
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a deviceCount has been set
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a deviceCount is NULL
 */
nvmlReturn_t DECLDIR nvmlGetExcludedDeviceCount(unsigned int *deviceCount);

/**
 * Acquire the device information for an excluded GPU device, based on its index.
 *
 * For all products.
 *
 * Valid indices are derived from the \a deviceCount returned by
 *   \ref nvmlGetExcludedDeviceCount(). For example, if \a deviceCount is 2 the valid indices
 *   are 0 and 1, corresponding to GPU 0 and GPU 1.
 *
 * @param index                                The index of the target GPU, >= 0 and < \a deviceCount
 * @param info                                 Reference in which to return the device information
 *
 * @return
 *         - \ref NVML_SUCCESS                  if \a device has been set
 *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a index is invalid or \a info is NULL
 *
 * @see nvmlGetExcludedDeviceCount
 */
nvmlReturn_t DECLDIR nvmlGetExcludedDeviceInfoByIndex(unsigned int index, nvmlExcludedDeviceInfo_t *info);

/** @} */

/***************************************************************************************************/
/** @defgroup nvmlMultiInstanceGPU Multi Instance GPU Management
 * This chapter describes NVML operations that are associated with Multi Instance GPU management.
 *  @{
 */
/***************************************************************************************************/

/**
 * Disable Multi Instance GPU mode.
 */
#define NVML_DEVICE_MIG_DISABLE 0x0

/**
 * Enable Multi Instance GPU mode.
 */
#define NVML_DEVICE_MIG_ENABLE 0x1

/**
 * GPU instance profiles.
 *
 * These macros should be passed to \ref nvmlDeviceGetGpuInstanceProfileInfo to retrieve the
 * detailed information about a GPU instance such as profile ID, engine counts.
 */
#define NVML_GPU_INSTANCE_PROFILE_1_SLICE      0x0
#define NVML_GPU_INSTANCE_PROFILE_2_SLICE      0x1
#define NVML_GPU_INSTANCE_PROFILE_3_SLICE      0x2
#define NVML_GPU_INSTANCE_PROFILE_4_SLICE      0x3
#define NVML_GPU_INSTANCE_PROFILE_7_SLICE      0x4
#define NVML_GPU_INSTANCE_PROFILE_8_SLICE      0x5
#define NVML_GPU_INSTANCE_PROFILE_6_SLICE      0x6
#define NVML_GPU_INSTANCE_PROFILE_1_SLICE_REV1 0x7
#define NVML_GPU_INSTANCE_PROFILE_COUNT        0x8

typedef struct nvmlGpuInstancePlacement_st
{
    unsigned int start;               //!< Index of first occupied memory slice
    unsigned int size;                //!< Number of memory slices occupied
} nvmlGpuInstancePlacement_t;

/**
 * GPU instance profile information.
 */
typedef struct nvmlGpuInstanceProfileInfo_st
{
    unsigned int id;                  //!< Unique profile ID within the device
    unsigned int isP2pSupported;      //!< Peer-to-Peer support
    unsigned int sliceCount;          //!< GPU Slice count
    unsigned int instanceCount;       //!< GPU instance count
    unsigned int multiprocessorCount; //!< Streaming Multiprocessor count
    unsigned int copyEngineCount;     //!< Copy Engine count
    unsigned int decoderCount;        //!< Decoder Engine count
    unsigned int encoderCount;        //!< Encoder Engine count
    unsigned int jpegCount;           //!< JPEG Engine count
    unsigned int ofaCount;            //!< OFA Engine count
    unsigned long long memorySizeMB;  //!< Memory size in MBytes
} nvmlGpuInstanceProfileInfo_t;

/**
 * GPU instance profile information (v2).
 * 
 * Version 2 adds the \ref nvmlGpuInstanceProfileInfo_v2_t.version field
 * to the start of the structure, and the \ref nvmlGpuInstanceProfileInfo_v2_t.name
 * field to the end. This structure is not backwards-compatible with
 * \ref nvmlGpuInstanceProfileInfo_t.
 */
typedef struct nvmlGpuInstanceProfileInfo_v2_st
{
    unsigned int version;                       //!< Structure version identifier (set to \ref nvmlGpuInstanceProfileInfo_v2)
    unsigned int id;                            //!< Unique profile ID within the device
    unsigned int isP2pSupported;                //!< Peer-to-Peer support
    unsigned int sliceCount;                    //!< GPU Slice count
    unsigned int instanceCount;                 //!< GPU instance count
    unsigned int multiprocessorCount;           //!< Streaming Multiprocessor count
    unsigned int copyEngineCount;               //!< Copy Engine count
    unsigned int decoderCount;                  //!< Decoder Engine count
    unsigned int encoderCount;                  //!< Encoder Engine count
    unsigned int jpegCount;                     //!< JPEG Engine count
    unsigned int ofaCount;                      //!< OFA Engine count
    unsigned long long memorySizeMB;            //!< Memory size in MBytes
    char name[NVML_DEVICE_NAME_V2_BUFFER_SIZE]; //!< Profile name
} nvmlGpuInstanceProfileInfo_v2_t;

/**
 * Version identifier value for \ref nvmlGpuInstanceProfileInfo_v2_t.version.
 */
#define nvmlGpuInstanceProfileInfo_v2 NVML_STRUCT_VERSION(GpuInstanceProfileInfo, 2)

typedef struct nvmlGpuInstanceInfo_st
{
    nvmlDevice_t device;                      //!< Parent device
    unsigned int id;                          //!< Unique instance ID within the device
    unsigned int profileId;                   //!< Unique profile ID within the device
    nvmlGpuInstancePlacement_t placement;     //!< Placement for this instance
} nvmlGpuInstanceInfo_t;

typedef struct nvmlGpuInstance_st* nvmlGpuInstance_t;

/**
 * Compute instance profiles.
 *
 * These macros should be passed to \ref nvmlGpuInstanceGetComputeInstanceProfileInfo to retrieve the
 * detailed information about a compute instance such as profile ID, engine counts
 */
#define NVML_COMPUTE_INSTANCE_PROFILE_1_SLICE 0x0
#define NVML_COMPUTE_INSTANCE_PROFILE_2_SLICE 0x1
#define NVML_COMPUTE_INSTANCE_PROFILE_3_SLICE 0x2
#define NVML_COMPUTE_INSTANCE_PROFILE_4_SLICE 0x3
#define NVML_COMPUTE_INSTANCE_PROFILE_7_SLICE 0x4
#define NVML_COMPUTE_INSTANCE_PROFILE_8_SLICE 0x5
#define NVML_COMPUTE_INSTANCE_PROFILE_6_SLICE 0x6
#define NVML_COMPUTE_INSTANCE_PROFILE_COUNT   0x7

#define NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED 0x0 //!< All the engines except multiprocessors would be shared
#define NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_COUNT  0x1

typedef struct nvmlComputeInstancePlacement_st
{
    unsigned int start;                 //!< Index of first occupied compute slice
    unsigned int size;                  //!< Number of compute slices occupied
} nvmlComputeInstancePlacement_t;

/**
 * Compute instance profile information.
 */
typedef struct nvmlComputeInstanceProfileInfo_st
{
    unsigned int id;                    //!< Unique profile ID within the GPU instance
    unsigned int sliceCount;            //!< GPU Slice count
    unsigned int instanceCount;         //!< Compute instance count
    unsigned int multiprocessorCount;   //!< Streaming Multiprocessor count
    unsigned int sharedCopyEngineCount; //!< Shared Copy Engine count
    unsigned int sharedDecoderCount;    //!< Shared Decoder Engine count
    unsigned int sharedEncoderCount;    //!< Shared Encoder Engine count
    unsigned int sharedJpegCount;       //!< Shared JPEG Engine count
    unsigned int sharedOfaCount;        //!< Shared OFA Engine count
} nvmlComputeInstanceProfileInfo_t;

/**
 * Compute instance profile information (v2).
 * 
 * Version 2 adds the \ref nvmlComputeInstanceProfileInfo_v2_t.version field
 * to the start of the structure, and the \ref nvmlComputeInstanceProfileInfo_v2_t.name
 * field to the end. This structure is not backwards-compatible with
 * \ref nvmlComputeInstanceProfileInfo_t.
 */
typedef struct nvmlComputeInstanceProfileInfo_v2_st
{
    unsigned int version;                       //!< Structure version identifier (set to \ref nvmlComputeInstanceProfileInfo_v2)
    unsigned int id;                            //!< Unique profile ID within the GPU instance
    unsigned int sliceCount;                    //!< GPU Slice count
    unsigned int instanceCount;                 //!< Compute instance count
    unsigned int multiprocessorCount;           //!< Streaming Multiprocessor count
    unsigned int sharedCopyEngineCount;         //!< Shared Copy Engine count
    unsigned int sharedDecoderCount;            //!< Shared Decoder Engine count
    unsigned int sharedEncoderCount;            //!< Shared Encoder Engine count
    unsigned int sharedJpegCount;               //!< Shared JPEG Engine count
    unsigned int sharedOfaCount;                //!< Shared OFA Engine count
    char name[NVML_DEVICE_NAME_V2_BUFFER_SIZE]; //!< Profile name
} nvmlComputeInstanceProfileInfo_v2_t;

/**
 * Version identifier value for \ref nvmlComputeInstanceProfileInfo_v2_t.version.
 */
#define nvmlComputeInstanceProfileInfo_v2 NVML_STRUCT_VERSION(ComputeInstanceProfileInfo, 2)

typedef struct nvmlComputeInstanceInfo_st
{
    nvmlDevice_t device;                      //!< Parent device
    nvmlGpuInstance_t gpuInstance;            //!< Parent GPU instance
    unsigned int id;                          //!< Unique instance ID within the GPU instance
    unsigned int profileId;                   //!< Unique profile ID within the GPU instance
    nvmlComputeInstancePlacement_t placement; //!< Placement for this instance within the GPU instance's compute slice range {0, sliceCount}
} nvmlComputeInstanceInfo_t;

typedef struct nvmlComputeInstance_st* nvmlComputeInstance_t;

/**
 * Set MIG mode for the device.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Requires root user.
 *
 * This mode determines whether a GPU instance can be created.
 *
 * This API may unbind or reset the device to activate the requested mode. Thus, the attributes associated with the
 * device, such as minor number, might change. The caller of this API is expected to query such attributes again.
 *
 * On certain platforms like pass-through virtualization, where reset functionality may not be exposed directly, VM
 * reboot is required. \a activationStatus would return \ref NVML_ERROR_RESET_REQUIRED for such cases.
 *
 * \a activationStatus would return the appropriate error code upon unsuccessful activation. For example, if device
 * unbind fails because the device isn't idle, \ref NVML_ERROR_IN_USE would be returned. The caller of this API
 * is expected to idle the device and retry setting the \a mode.
 *
 * @note On Windows, only disabling MIG mode is supported. \a activationStatus would return \ref
 *       NVML_ERROR_NOT_SUPPORTED as GPU reset is not supported on Windows through this API.
 *
 * @param device                               The identifier of the target device
 * @param mode                                 The mode to be set, \ref NVML_DEVICE_MIG_DISABLE or
 *                                             \ref NVML_DEVICE_MIG_ENABLE
 * @param activationStatus                     The activationStatus status
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device,\a mode or \a activationStatus are invalid
 *         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 *         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't support MIG mode
 */
nvmlReturn_t DECLDIR nvmlDeviceSetMigMode(nvmlDevice_t device, unsigned int mode, nvmlReturn_t *activationStatus);

/**
 * Get MIG mode for the device.
 *
 * For Ampere &tm; or newer fully supported devices.
 *
 * Changing MIG modes may require device unbind or reset. The "pending" MIG mode refers to the target mode following the
 * next activation trigger.
 *
 * @param device                               The identifier of the target device
 * @param currentMode                          Returns the current mode, \ref NVML_DEVICE_MIG_DISABLE or
 *                                             \ref NVML_DEVICE_MIG_ENABLE
 * @param pendingMode                          Returns the pending mode, \ref NVML_DEVICE_MIG_DISABLE or
 *                                             \ref NVML_DEVICE_MIG_ENABLE
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device, \a currentMode or \a pendingMode are invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't support MIG mode
 */
nvmlReturn_t DECLDIR nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int *currentMode, unsigned int *pendingMode);

/**
 * Get GPU instance profile information.
 *
 * Information provided by this API is immutable throughout the lifetime of a MIG mode.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               The identifier of the target device
 * @param profile                              One of the NVML_GPU_INSTANCE_PROFILE_*
 * @param info                                 Returns detailed profile information
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device, \a profile or \a info are invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled or \a profile isn't supported
 *         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
nvmlReturn_t DECLDIR nvmlDeviceGetGpuInstanceProfileInfo(nvmlDevice_t device, unsigned int profile,
                                                         nvmlGpuInstanceProfileInfo_t *info);

/**
 * Versioned wrapper around \ref nvmlDeviceGetGpuInstanceProfileInfo that accepts a versioned
 * \ref nvmlGpuInstanceProfileInfo_v2_t or later output structure.
 * 
 * @note The caller must set the \ref nvmlGpuInstanceProfileInfo_v2_t.version field to the
 * appropriate version prior to calling this function. For example:
 * \code
 *     nvmlGpuInstanceProfileInfo_v2_t profileInfo =
 *         { .version = nvmlGpuInstanceProfileInfo_v2 };
 *     nvmlReturn_t result = nvmlDeviceGetGpuInstanceProfileInfoV(device,
 *                                                                profile,
 *                                                                &profileInfo);
 * \endcode
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               The identifier of the target device
 * @param profile                              One of the NVML_GPU_INSTANCE_PROFILE_*
 * @param info                                 Returns detailed profile information
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device, \a profile, \a info, or \a info->version are invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled or \a profile isn't supported
 *         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
nvmlReturn_t DECLDIR nvmlDeviceGetGpuInstanceProfileInfoV(nvmlDevice_t device, unsigned int profile,
                                                          nvmlGpuInstanceProfileInfo_v2_t *info);

/**
 * Get GPU instance placements.
 *
 * A placement represents the location of a GPU instance within a device. This API only returns all the possible
 * placements for the given profile.
 * A created GPU instance occupies memory slices described by its placement. Creation of new GPU instance will
 * fail if there is overlap with the already occupied memory slices.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param device                               The identifier of the target device
 * @param profileId                            The GPU instance profile ID. See \ref nvmlDeviceGetGpuInstanceProfileInfo
 * @param placements                           Returns placements allowed for the profile. Can be NULL to discover number
 *                                             of allowed placements for this profile. If non-NULL must be large enough
 *                                             to accommodate the placements supported by the profile.
 * @param count                                Returns number of allowed placemenets for the profile.
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device, \a profileId or \a count are invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled or \a profileId isn't supported
 *         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
nvmlReturn_t DECLDIR nvmlDeviceGetGpuInstancePossiblePlacements_v2(nvmlDevice_t device, unsigned int profileId,
                                                                   nvmlGpuInstancePlacement_t *placements,
                                                                   unsigned int *count);

/**
 * Get GPU instance profile capacity.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param device                               The identifier of the target device
 * @param profileId                            The GPU instance profile ID. See \ref nvmlDeviceGetGpuInstanceProfileInfo
 * @param count                                Returns remaining instance count for the profile ID
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device, \a profileId or \a count are invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled or \a profileId isn't supported
 *         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
nvmlReturn_t DECLDIR nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t device, unsigned int profileId,
                                                               unsigned int *count);

/**
 * Create GPU instance.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * If the parent device is unbound, reset or the GPU instance is destroyed explicitly, the GPU instance handle would
 * become invalid. The GPU instance must be recreated to acquire a valid handle.
 *
 * @param device                               The identifier of the target device
 * @param profileId                            The GPU instance profile ID. See \ref nvmlDeviceGetGpuInstanceProfileInfo
 * @param gpuInstance                          Returns the GPU instance handle
 *
 * @return
 *         - \ref NVML_SUCCESS                       Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED           If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT        If \a device, \a profile, \a profileId or \a gpuInstance are invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED           If \a device doesn't have MIG mode enabled or in vGPU guest
 *         - \ref NVML_ERROR_NO_PERMISSION           If user doesn't have permission to perform the operation
 *         - \ref NVML_ERROR_INSUFFICIENT_RESOURCES  If the requested GPU instance could not be created
 */
nvmlReturn_t DECLDIR nvmlDeviceCreateGpuInstance(nvmlDevice_t device, unsigned int profileId,
                                                 nvmlGpuInstance_t *gpuInstance);

/**
 * Create GPU instance with the specified placement.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * If the parent device is unbound, reset or the GPU instance is destroyed explicitly, the GPU instance handle would
 * become invalid. The GPU instance must be recreated to acquire a valid handle.
 *
 * @param device                               The identifier of the target device
 * @param profileId                            The GPU instance profile ID. See \ref nvmlDeviceGetGpuInstanceProfileInfo
 * @param placement                            The requested placement. See \ref nvmlDeviceGetGpuInstancePossiblePlacements_v2
 * @param gpuInstance                          Returns the GPU instance handle
 *
 * @return
 *         - \ref NVML_SUCCESS                       Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED           If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT        If \a device, \a profile, \a profileId, \a placement or \a gpuInstance
 *                                                   are invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED           If \a device doesn't have MIG mode enabled or in vGPU guest
 *         - \ref NVML_ERROR_NO_PERMISSION           If user doesn't have permission to perform the operation
 *         - \ref NVML_ERROR_INSUFFICIENT_RESOURCES  If the requested GPU instance could not be created
 */
nvmlReturn_t DECLDIR nvmlDeviceCreateGpuInstanceWithPlacement(nvmlDevice_t device, unsigned int profileId,
                                                              const nvmlGpuInstancePlacement_t *placement,
                                                              nvmlGpuInstance_t *gpuInstance);
/**
 * Destroy GPU instance.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param gpuInstance                          The GPU instance handle
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a gpuInstance is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled or in vGPU guest
 *         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 *         - \ref NVML_ERROR_IN_USE            If the GPU instance is in use. This error would be returned if processes
 *                                             (e.g. CUDA application) or compute instances are active on the
 *                                             GPU instance.
 */
nvmlReturn_t DECLDIR nvmlGpuInstanceDestroy(nvmlGpuInstance_t gpuInstance);

/**
 * Get GPU instances for given profile ID.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param device                               The identifier of the target device
 * @param profileId                            The GPU instance profile ID. See \ref nvmlDeviceGetGpuInstanceProfileInfo
 * @param gpuInstances                         Returns pre-exiting GPU instances, the buffer must be large enough to
 *                                             accommodate the instances supported by the profile.
 *                                             See \ref nvmlDeviceGetGpuInstanceProfileInfo
 * @param count                                The count of returned GPU instances
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device, \a profileId, \a gpuInstances or \a count are invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled
 *         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
nvmlReturn_t DECLDIR nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId,
                                               nvmlGpuInstance_t *gpuInstances, unsigned int *count);

/**
 * Get GPU instances for given instance ID.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param device                               The identifier of the target device
 * @param id                                   The GPU instance ID
 * @param gpuInstance                          Returns GPU instance
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device, \a id or \a gpuInstance are invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled
 *         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 *         - \ref NVML_ERROR_NOT_FOUND         If the GPU instance is not found.
 */
nvmlReturn_t DECLDIR nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id, nvmlGpuInstance_t *gpuInstance);

/**
 * Get GPU instance information.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param gpuInstance                          The GPU instance handle
 * @param info                                 Return GPU instance information
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a gpuInstance or \a info are invalid
 *         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
nvmlReturn_t DECLDIR nvmlGpuInstanceGetInfo(nvmlGpuInstance_t gpuInstance, nvmlGpuInstanceInfo_t *info);

/**
 * Get compute instance profile information.
 *
 * Information provided by this API is immutable throughout the lifetime of a MIG mode.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param gpuInstance                          The identifier of the target GPU instance
 * @param profile                              One of the NVML_COMPUTE_INSTANCE_PROFILE_*
 * @param engProfile                           One of the NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_*
 * @param info                                 Returns detailed profile information
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a gpuInstance, \a profile, \a engProfile or \a info are invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     If \a profile isn't supported
 *         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
nvmlReturn_t DECLDIR nvmlGpuInstanceGetComputeInstanceProfileInfo(nvmlGpuInstance_t gpuInstance, unsigned int profile,
                                                                  unsigned int engProfile,
                                                                  nvmlComputeInstanceProfileInfo_t *info);

/**
 * Versioned wrapper around \ref nvmlGpuInstanceGetComputeInstanceProfileInfo that accepts a versioned
 * \ref nvmlComputeInstanceProfileInfo_v2_t or later output structure.
 *
 * @note The caller must set the \ref nvmlGpuInstanceProfileInfo_v2_t.version field to the
 * appropriate version prior to calling this function. For example:
 * \code
 *     nvmlComputeInstanceProfileInfo_v2_t profileInfo =
 *         { .version = nvmlComputeInstanceProfileInfo_v2 };
 *     nvmlReturn_t result = nvmlGpuInstanceGetComputeInstanceProfileInfoV(gpuInstance,
 *                                                                         profile,
 *                                                                         engProfile,
 *                                                                         &profileInfo);
 * \endcode
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param gpuInstance                          The identifier of the target GPU instance
 * @param profile                              One of the NVML_COMPUTE_INSTANCE_PROFILE_*
 * @param engProfile                           One of the NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_*
 * @param info                                 Returns detailed profile information
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a gpuInstance, \a profile, \a engProfile, \a info, or \a info->version are invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     If \a profile isn't supported
 *         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
nvmlReturn_t DECLDIR nvmlGpuInstanceGetComputeInstanceProfileInfoV(nvmlGpuInstance_t gpuInstance, unsigned int profile,
                                                                   unsigned int engProfile,
                                                                   nvmlComputeInstanceProfileInfo_v2_t *info);

/**
 * Get compute instance profile capacity.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param gpuInstance                          The identifier of the target GPU instance
 * @param profileId                            The compute instance profile ID.
 *                                             See \ref nvmlGpuInstanceGetComputeInstanceProfileInfo
 * @param count                                Returns remaining instance count for the profile ID
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a gpuInstance, \a profileId or \a availableCount are invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     If \a profileId isn't supported
 *         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
nvmlReturn_t DECLDIR nvmlGpuInstanceGetComputeInstanceRemainingCapacity(nvmlGpuInstance_t gpuInstance,
                                                                        unsigned int profileId, unsigned int *count);

/**
 * Create compute instance.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * If the parent device is unbound, reset or the parent GPU instance is destroyed or the compute instance is destroyed
 * explicitly, the compute instance handle would become invalid. The compute instance must be recreated to acquire
 * a valid handle.
 *
 * @param gpuInstance                          The identifier of the target GPU instance
 * @param profileId                            The compute instance profile ID.
 *                                             See \ref nvmlGpuInstanceGetComputeInstanceProfileInfo
 * @param computeInstance                      Returns the compute instance handle
 *
 * @return
 *         - \ref NVML_SUCCESS                       Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED           If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT        If \a gpuInstance, \a profile, \a profileId or \a computeInstance
 *                                                   are invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED           If \a profileId isn't supported
 *         - \ref NVML_ERROR_NO_PERMISSION           If user doesn't have permission to perform the operation
 *         - \ref NVML_ERROR_INSUFFICIENT_RESOURCES  If the requested compute instance could not be created
 */
nvmlReturn_t DECLDIR nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t gpuInstance, unsigned int profileId,
                                                          nvmlComputeInstance_t *computeInstance);

/**
 * Destroy compute instance.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param computeInstance                      The compute instance handle
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a computeInstance is invalid
 *         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 *         - \ref NVML_ERROR_IN_USE            If the compute instance is in use. This error would be returned if
 *                                             processes (e.g. CUDA application) are active on the compute instance.
 */
nvmlReturn_t DECLDIR nvmlComputeInstanceDestroy(nvmlComputeInstance_t computeInstance);

/**
 * Get compute instances for given profile ID.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param gpuInstance                          The identifier of the target GPU instance
 * @param profileId                            The compute instance profile ID.
 *                                             See \ref nvmlGpuInstanceGetComputeInstanceProfileInfo
 * @param computeInstances                     Returns pre-exiting compute instances, the buffer must be large enough to
 *                                             accommodate the instances supported by the profile.
 *                                             See \ref nvmlGpuInstanceGetComputeInstanceProfileInfo
 * @param count                                The count of returned compute instances
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a gpuInstance, \a profileId, \a computeInstances or \a count
 *                                             are invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     If \a profileId isn't supported
 *         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
nvmlReturn_t DECLDIR nvmlGpuInstanceGetComputeInstances(nvmlGpuInstance_t gpuInstance, unsigned int profileId,
                                                        nvmlComputeInstance_t *computeInstances, unsigned int *count);

/**
 * Get compute instance for given instance ID.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param gpuInstance                          The identifier of the target GPU instance
 * @param id                                   The compute instance ID
 * @param computeInstance                      Returns compute instance
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device, \a ID or \a computeInstance are invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled
 *         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 *         - \ref NVML_ERROR_NOT_FOUND         If the compute instance is not found.
 */
nvmlReturn_t DECLDIR nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t gpuInstance, unsigned int id,
                                                           nvmlComputeInstance_t *computeInstance);

/**
 * Get compute instance information.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param computeInstance                      The compute instance handle
 * @param info                                 Return compute instance information
 *
 * @return
 *         - \ref NVML_SUCCESS                 Upon success
 *         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a computeInstance or \a info are invalid
 *         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
nvmlReturn_t DECLDIR nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t *info);

/**
 * Test if the given handle refers to a MIG device.
 *
 * A MIG device handle is an NVML abstraction which maps to a MIG compute instance.
 * These overloaded references can be used (with some restrictions) interchangeably
 * with a GPU device handle to execute queries at a per-compute instance granularity.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               NVML handle to test
 * @param isMigDevice                          True when handle refers to a MIG device
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a device status was successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device handle or \a isMigDevice reference is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this check is not supported by the device
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int *isMigDevice);

/**
 * Get GPU instance ID for the given MIG device handle.
 *
 * GPU instance IDs are unique per device and remain valid until the GPU instance is destroyed.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               Target MIG device handle
 * @param id                                   GPU instance ID
 *
 * @return
 *         - \ref NVML_SUCCESS                 if instance ID was successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a id reference is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int *id);

/**
 * Get compute instance ID for the given MIG device handle.
 *
 * Compute instance IDs are unique per GPU instance and remain valid until the compute instance
 * is destroyed.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               Target MIG device handle
 * @param id                                   Compute instance ID
 *
 * @return
 *         - \ref NVML_SUCCESS                 if instance ID was successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a id reference is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetComputeInstanceId(nvmlDevice_t device, unsigned int *id);

/**
 * Get the maximum number of MIG devices that can exist under a given parent NVML device.
 *
 * Returns zero if MIG is not supported or enabled.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               Target device handle
 * @param count                                Count of MIG devices
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a count was successfully retrieved
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a count reference is invalid
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int *count);

/**
 * Get MIG device handle for the given index under its parent NVML device.
 *
 * If the compute instance is destroyed either explicitly or by destroying,
 * resetting or unbinding the parent GPU instance or the GPU device itself
 * the MIG device handle would remain invalid and must be requested again
 * using this API. Handles may be reused and their properties can change in
 * the process.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               Reference to the parent GPU device handle
 * @param index                                Index of the MIG device
 * @param migDevice                            Reference to the MIG device handle
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a migDevice handle was successfully created
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a index or \a migDevice reference is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref NVML_ERROR_NOT_FOUND         if no valid MIG device was found at \a index
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index,
                                                         nvmlDevice_t *migDevice);

/**
 * Get parent device handle from a MIG device handle.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param migDevice                            MIG device handle
 * @param device                               Device handle
 *
 * @return
 *         - \ref NVML_SUCCESS                 if \a device handle was successfully created
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a migDevice or \a device is invalid
 *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice, nvmlDevice_t *device);

/**
 * Get the type of the GPU Bus (PCIe, PCI, ...)
 *
 * @param device                               The identifier of the target device
 * @param type                                 The PCI Bus type
 *
 * return
 *         - \ref NVML_SUCCESS                 if the bus \a type is successfully retreived
 *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \device is invalid or \type is NULL
 *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
 */
nvmlReturn_t DECLDIR nvmlDeviceGetBusType(nvmlDevice_t device, nvmlBusType_t *type);

/** @} */

/**
 * NVML API versioning support
 */

#ifdef NVML_NO_UNVERSIONED_FUNC_DEFS
nvmlReturn_t DECLDIR nvmlInit(void);
nvmlReturn_t DECLDIR nvmlDeviceGetCount(unsigned int *deviceCount);
nvmlReturn_t DECLDIR nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device);
nvmlReturn_t DECLDIR nvmlDeviceGetHandleByPciBusId(const char *pciBusId, nvmlDevice_t *device);
nvmlReturn_t DECLDIR nvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t *pci);
nvmlReturn_t DECLDIR nvmlDeviceGetPciInfo_v2(nvmlDevice_t device, nvmlPciInfo_t *pci);
nvmlReturn_t DECLDIR nvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci);
nvmlReturn_t DECLDIR nvmlDeviceGetGridLicensableFeatures(nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures);
nvmlReturn_t DECLDIR nvmlDeviceGetGridLicensableFeatures_v2(nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures);
nvmlReturn_t DECLDIR nvmlDeviceGetGridLicensableFeatures_v3(nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures);
nvmlReturn_t DECLDIR nvmlDeviceRemoveGpu(nvmlPciInfo_t *pciInfo);
nvmlReturn_t DECLDIR nvmlEventSetWait(nvmlEventSet_t set, nvmlEventData_t * data, unsigned int timeoutms);
nvmlReturn_t DECLDIR nvmlDeviceGetAttributes(nvmlDevice_t device, nvmlDeviceAttributes_t *attributes);
nvmlReturn_t DECLDIR nvmlComputeInstanceGetInfo(nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t *info);
nvmlReturn_t DECLDIR nvmlDeviceGetComputeRunningProcesses(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_v1_t *infos);
nvmlReturn_t DECLDIR nvmlDeviceGetComputeRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_v2_t *infos);
nvmlReturn_t DECLDIR nvmlDeviceGetGraphicsRunningProcesses(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_v1_t *infos);
nvmlReturn_t DECLDIR nvmlDeviceGetGraphicsRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_v2_t *infos);
nvmlReturn_t DECLDIR nvmlDeviceGetMPSComputeRunningProcesses(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_v1_t *infos);
nvmlReturn_t DECLDIR nvmlDeviceGetMPSComputeRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_v2_t *infos);
nvmlReturn_t DECLDIR nvmlDeviceGetGpuInstancePossiblePlacements(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t *placements, unsigned int *count);
nvmlReturn_t DECLDIR nvmlVgpuInstanceGetLicenseInfo(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuLicenseInfo_t *licenseInfo);
#endif // #ifdef NVML_NO_UNVERSIONED_FUNC_DEFS

#if defined(NVML_NO_UNVERSIONED_FUNC_DEFS)
// We don't define APIs to run new versions if this guard is present so there is
// no need to undef
#elif defined(__NVML_API_VERSION_INTERNAL)
#undef nvmlDeviceGetGraphicsRunningProcesses
#undef nvmlDeviceGetComputeRunningProcesses
#undef nvmlDeviceGetMPSComputeRunningProcesses
#undef nvmlDeviceGetAttributes
#undef nvmlComputeInstanceGetInfo
#undef nvmlEventSetWait
#undef nvmlDeviceGetGridLicensableFeatures
#undef nvmlDeviceRemoveGpu
#undef nvmlDeviceGetNvLinkRemotePciInfo
#undef nvmlDeviceGetPciInfo
#undef nvmlDeviceGetCount
#undef nvmlDeviceGetHandleByIndex
#undef nvmlDeviceGetHandleByPciBusId
#undef nvmlInit
#undef nvmlBlacklistDeviceInfo_t
#undef nvmlGetBlacklistDeviceCount
#undef nvmlGetBlacklistDeviceInfoByIndex
#undef nvmlDeviceGetGpuInstancePossiblePlacements
#undef nvmlVgpuInstanceGetLicenseInfo
#endif

#ifdef __cplusplus
}
#endif

#endif
