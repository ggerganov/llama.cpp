#pragma once

#include <math.h>

#include <mutex>
#include <string>
#include <unordered_map>

// header file of Qualcomm QNN(Qualcomm Neural Network, aka Qualcomm AI Engine Direct) SDK
// https://qpm.qualcomm.com/#/main/tools/details/qualcomm_ai_engine_direct
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpGraph.h>
#include <QnnBackend.h>
#include <QnnCommon.h>
#include <QnnContext.h>
#include <QnnGraph.h>
#include <QnnInterface.h>
#include <QnnProperty.h>
#include <QnnTensor.h>
#include <QnnTypes.h>
#include <System/QnnSystemInterface.h>

#include "qnn-types.hpp"
#include "utils.hpp"

namespace qnn {

// =================================================================================================
//
// wrapper class of Qualcomm QNN(Qualcomm Neural Network, aka Qualcomm AI Engine Direct) SDK
// ref:https://github.com/pytorch/executorch/tree/main/backends/qualcomm
// =================================================================================================
class qnn_interface {

#define DEFINE_SHIM_FUNCTION_INTERFACE(F, pointer_name)                                            \
    template <typename... Args>                                                                    \
    inline auto qnn_##F(Args... args) const {                                                      \
        return (_qnn_interface->QNN_INTERFACE_VER_NAME.pointer_name)(std::forward<Args>(args)...); \
    }

#define DEFINE_SHIM_FUNCTION_SYS_INTERFACE(F, pointer_name)                                                   \
    template <typename... Args>                                                                               \
    inline auto qnn_##F(Args... args) const {                                                                 \
        return (_qnn_sys_interface->QNN_SYSTEM_INTERFACE_VER_NAME.pointer_name)(std::forward<Args>(args)...); \
    }

    friend class qnn_instance;

public:
    qnn_interface() = default;

    // QnnBackend
    DEFINE_SHIM_FUNCTION_INTERFACE(backend_create, backendCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_free, backendFree);

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_register_op_package, backendRegisterOpPackage);

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_validate_op_config, backendValidateOpConfig);

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_get_api_version, backendGetApiVersion);

    // QnnDevice
    DEFINE_SHIM_FUNCTION_INTERFACE(device_create, deviceCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(device_free, deviceFree);

    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_infrastructure, deviceGetInfrastructure);

    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_platform_info, deviceGetPlatformInfo);

    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_info, deviceGetInfo);

    // QnnContext
    DEFINE_SHIM_FUNCTION_INTERFACE(context_create, contextCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(context_get_binary_size, contextGetBinarySize);

    DEFINE_SHIM_FUNCTION_INTERFACE(context_get_binary, contextGetBinary);

    DEFINE_SHIM_FUNCTION_INTERFACE(context_create_from_binary, contextCreateFromBinary);

    DEFINE_SHIM_FUNCTION_INTERFACE(context_free, contextFree);

    // QnnGraph
    DEFINE_SHIM_FUNCTION_INTERFACE(graph_create, graphCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_add_node, graphAddNode);

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_finalize, graphFinalize);

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_execute, graphExecute);

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_retrieve, graphRetrieve);

    // QnnLog
    DEFINE_SHIM_FUNCTION_INTERFACE(log_create, logCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(log_free, logFree);

    DEFINE_SHIM_FUNCTION_INTERFACE(log_set_log_level, logSetLogLevel);

    // QnnProfile
    DEFINE_SHIM_FUNCTION_INTERFACE(profile_create, profileCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_events, profileGetEvents);

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_sub_events, profileGetSubEvents);

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_event_data, profileGetEventData);

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_free, profileFree);

    // QnnMem
    DEFINE_SHIM_FUNCTION_INTERFACE(mem_register, memRegister);

    DEFINE_SHIM_FUNCTION_INTERFACE(mem_de_register, memDeRegister);

    // QnnProperty
    DEFINE_SHIM_FUNCTION_INTERFACE(property_has_capability, propertyHasCapability);

    // QnnTensor
    DEFINE_SHIM_FUNCTION_INTERFACE(tensor_create_context_tensor, tensorCreateContextTensor);

    DEFINE_SHIM_FUNCTION_INTERFACE(tensor_create_graph_tensor, tensorCreateGraphTensor);

    // QnnSystem
    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_create, systemContextCreate);

    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_get_binary_info, systemContextGetBinaryInfo);

    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_free, systemContextFree);

    void set_qnn_interface(const QnnInterface_t *qnn_interface) { _qnn_interface = qnn_interface; }

    void set_qnn_system_interface(const QnnSystemInterface_t *qnn_sys_interface) {
        _qnn_sys_interface = qnn_sys_interface;
    }

    uint32_t get_backend_id() const { return _qnn_interface->backendId; }

    bool is_loaded() const { return ((_qnn_sys_interface != nullptr) && (_qnn_interface != nullptr)); }

private:
    const QnnInterface_t *_qnn_interface = nullptr;

    const QnnSystemInterface_t *_qnn_sys_interface = nullptr;
};

class qnn_instance {
public:
    using BackendIdType = decltype(QnnInterface_t{}.backendId);

    explicit qnn_instance(const std::string &lib_path, const std::string &backend_name, const std::string &model_name) :
        _lib_path(std::move(lib_path)), _backend_name(std::move(backend_name)), _model_name(std::move(model_name)) {};

    ~qnn_instance() {}

    int qnn_init(const QnnSaver_Config_t **saver_config) {
        BackendIdType backend_id = QNN_BACKEND_ID_NULL;
        QNN_LOG_DEBUG("enter qni_init\n");

        std::lock_guard<std::mutex> lock(_init_mutex);

        if (0 != load_system()) {
            QNN_LOG_WARN("can not load QNN system lib, pls check why?\n");
            return 1;
        } else {
            QNN_LOG_DEBUG("load QNN system lib successfully\n");
        }

        std::string backend_lib_path = _lib_path + _backend_name;
        if (0 == _lib_path_to_backend_id.count(backend_lib_path)) {
            int is_load_ok = load_backend(backend_lib_path, saver_config);
            if (0 != is_load_ok) {
                QNN_LOG_WARN("failed to load QNN backend\n");
                return 2;
            }
        }

        backend_id = _lib_path_to_backend_id[backend_lib_path];
        if (0 == _loaded_backend.count(backend_id) || 0 == _loaded_lib_handle.count(backend_id)) {
            QNN_LOG_WARN(
                "library %s is loaded but loaded backend count=%zu, "
                "loaded lib_handle count=%zu\n",
                backend_lib_path.c_str(), _loaded_backend.count(backend_id), _loaded_lib_handle.count(backend_id));
            return 3;
        }

        _qnn_interface.set_qnn_interface(_loaded_backend[backend_id]);

        _qnn_interface.qnn_log_create(qnn::sdk_logcallback, _qnn_log_level, &_qnn_log_handle);
        if (nullptr == _qnn_log_handle) {
            // NPU backend not work on Qualcomm SoC equipped low-end phone
            QNN_LOG_WARN("why failed to initialize qnn log\n");
            return 4;
        } else {
            QNN_LOG_DEBUG("initialize qnn log successfully\n");
        }

        std::vector<const QnnBackend_Config_t *> temp_backend_config;
        _qnn_interface.qnn_backend_create(
            _qnn_log_handle, temp_backend_config.empty() ? nullptr : temp_backend_config.data(), &_qnn_backend_handle);
        if (nullptr == _qnn_backend_handle) {
            QNN_LOG_WARN("why failed to initialize qnn backend\n");
            return 5;
        } else {
            QNN_LOG_DEBUG("initialize qnn backend successfully\n");
        }

        if (nullptr != _qnn_raw_interface.propertyHasCapability) {
            Qnn_ErrorHandle_t qnn_status = _qnn_raw_interface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
            if (QNN_PROPERTY_NOT_SUPPORTED == qnn_status) {
                QNN_LOG_WARN("device property is not supported\n");
            }
            if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnn_status) {
                QNN_LOG_WARN("device property is not known to backend\n");
            }
        }

        Qnn_ErrorHandle_t qnn_status = QNN_SUCCESS;
        if (_backend_name.find("Htp") != std::variant_npos) {
            const QnnDevice_PlatformInfo_t *p_info = nullptr;
            _qnn_raw_interface.deviceGetPlatformInfo(nullptr, &p_info);
            QNN_LOG_INFO("device counts %d", p_info->v1.numHwDevices);
            QnnDevice_HardwareDeviceInfo_t *infos = p_info->v1.hwDevices;
            QnnHtpDevice_OnChipDeviceInfoExtension_t chipinfo = {};
            for (int i = 0; i < p_info->v1.numHwDevices; i++) {
                QNN_LOG_INFO("deviceID:%d, deviceType:%d, numCores %d", infos[i].v1.deviceId, infos[i].v1.deviceType,
                             infos[i].v1.numCores);
                QnnDevice_DeviceInfoExtension_t devinfo = infos[i].v1.deviceInfoExtension;
                chipinfo = devinfo->onChipDevice;
                QnnHtpDevice_Arch_t htp_arch = chipinfo.arch;
                QNN_LOG_INFO("htp_type:%d(%s)", devinfo->devType,
                             (devinfo->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) ? "ON_CHIP" : "");
                QNN_LOG_INFO("qualcomm soc_model:%d(%s), htp_arch:%d(%s), vtcm_size:%d MB", chipinfo.socModel,
                             qnn::get_chipset_desc(chipinfo.socModel), htp_arch, qnn::get_htparch_desc(htp_arch),
                             chipinfo.vtcmSize);
                _soc_info = { chipinfo.socModel, htp_arch, chipinfo.vtcmSize };
            }
            _qnn_raw_interface.deviceFreePlatformInfo(nullptr, p_info);

            QnnHtpDevice_CustomConfig_t soc_customconfig;
            soc_customconfig.option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
            soc_customconfig.socModel = chipinfo.socModel;
            QnnDevice_Config_t soc_devconfig;
            soc_devconfig.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
            soc_devconfig.customConfig = &soc_customconfig;

            QnnHtpDevice_CustomConfig_t arch_customconfig;
            arch_customconfig.option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
            arch_customconfig.arch.arch = chipinfo.arch;
            arch_customconfig.arch.deviceId = 0; // Id of device to be used. If single device is used by default 0.
            QnnDevice_Config_t arch_devconfig;
            arch_devconfig.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
            arch_devconfig.customConfig = &arch_customconfig;

            const QnnDevice_Config_t *p_deviceconfig[] = { &soc_devconfig, &arch_devconfig, nullptr };
            qnn_status = _qnn_raw_interface.deviceCreate(_qnn_log_handle, p_deviceconfig, &_qnn_device_handle);
        } else {
            qnn_status = _qnn_raw_interface.deviceCreate(_qnn_log_handle, nullptr, &_qnn_device_handle);
        }
        if (QNN_SUCCESS != qnn_status && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnn_status) {
            QNN_LOG_WARN("failed to create QNN device\n");
        } else {
            QNN_LOG_INFO("create QNN device successfully\n");
        }

        if (qnn::sdk_profile_level::profile_off != _profile_level) {
            QNN_LOG_INFO("profiling turned on; level = %d", _profile_level);
            if (qnn::sdk_profile_level::profile_basic == _profile_level) {
                QNN_LOG_INFO("basic profiling requested. creating Qnn Profile object\n");
                if (QNN_PROFILE_NO_ERROR != _qnn_raw_interface.profileCreate(
                                                _qnn_backend_handle, QNN_PROFILE_LEVEL_BASIC, &_qnn_profile_handle)) {
                    QNN_LOG_WARN("unable to create profile handle in the backend\n");
                    return 6;
                } else {
                    QNN_LOG_DEBUG("initialize qnn profile successfully\n");
                }
            } else if (qnn::sdk_profile_level::profile_detail == _profile_level) {
                QNN_LOG_INFO("detailed profiling requested. Creating Qnn Profile object\n");
                if (QNN_PROFILE_NO_ERROR != _qnn_raw_interface.profileCreate(_qnn_backend_handle,
                                                                             QNN_PROFILE_LEVEL_DETAILED,
                                                                             &_qnn_profile_handle)) {
                    QNN_LOG_WARN("unable to create profile handle in the backend\n");
                    return 7;
                } else {
                    QNN_LOG_DEBUG("initialize qnn profile successfully\n");
                }
            }
        }

        _rpc_lib_handle = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
        if (nullptr == _rpc_lib_handle) {
            QNN_LOG_WARN("failed to load qualcomm's rpc lib, error:%s\n", dlerror());
            return 8;
        } else {
            QNN_LOG_DEBUG("load rpcmem lib successfully\n");
            set_rpcmem_initialized(true);
        }
        _pfn_rpc_mem_init = reinterpret_cast<qnn::pfn_rpc_mem_init>(dlsym(_rpc_lib_handle, "rpcmem_init"));
        _pfn_rpc_mem_deinit = reinterpret_cast<qnn::pfn_rpc_mem_deinit>(dlsym(_rpc_lib_handle, "rpcmem_deinit"));
        _pfn_rpc_mem_alloc = reinterpret_cast<qnn::pfn_rpc_mem_alloc>(dlsym(_rpc_lib_handle, "rpcmem_alloc"));
        _pfn_rpc_mem_free = reinterpret_cast<qnn::pfn_rpc_mem_free>(dlsym(_rpc_lib_handle, "rpcmem_free"));
        _pfn_rpc_mem_to_fd = reinterpret_cast<qnn::pfn_rpc_mem_to_fd>(dlsym(_rpc_lib_handle, "rpcmem_to_fd"));
        if (nullptr == _pfn_rpc_mem_alloc || nullptr == _pfn_rpc_mem_free || nullptr == _pfn_rpc_mem_to_fd) {
            QNN_LOG_WARN("unable to access symbols in QNN RPC lib. dlerror(): %s", dlerror());
            dlclose(_rpc_lib_handle);
            return 9;
        }

        if (nullptr != _pfn_rpc_mem_init) { // make Qualcomm's SoC equipped low-end phone happy
            _pfn_rpc_mem_init();
        }

        /* TODO: not used, keep it for further usage
                 QnnContext_Config_t qnn_context_config = QNN_CONTEXT_CONFIG_INIT;
                 qnn_context_config.priority = QNN_PRIORITY_DEFAULT;
                 const QnnContext_Config_t * context_configs[] = {&qnn_context_config, nullptr};
        */
        _qnn_interface.qnn_context_create(_qnn_backend_handle, _qnn_device_handle, nullptr, &_qnn_context_handle);
        if (nullptr == _qnn_context_handle) {
            QNN_LOG_WARN("why failed to initialize qnn context\n");
            return 10;
        } else {
            QNN_LOG_DEBUG("initialize qnn context successfully\n");
        }

        if (_backend_name.find("Htp") != std::variant_npos) {
            // TODO: faster approach to probe the accurate capacity of rpc ion memory
            size_t candidate_size = 0;
            uint8_t *rpc_buffer = nullptr;
            const int size_in_mb = (1 << 20);
            size_t probe_slots[] = { 1024, 1536, 2048 - 48, 2048 };
            size_t probe_counts = sizeof(probe_slots) / sizeof(size_t);
            for (size_t idx = 0; idx < probe_counts; idx++) {
                rpc_buffer = static_cast<uint8_t *>(alloc_rpcmem(probe_slots[idx] * size_in_mb, 4));
                if (nullptr == rpc_buffer) {
                    QNN_LOG_INFO("alloc rpcmem %d (MB) failure, %s\n", probe_slots[idx], strerror(errno));
                    break;
                } else {
                    candidate_size = probe_slots[idx];
                    free_rpcmem(rpc_buffer);
                    rpc_buffer = nullptr;
                }
            }
            if (candidate_size > _rpcmem_capacity) _rpcmem_capacity = candidate_size;
            QNN_LOG_INFO("capacity of QNN rpc ion memory is about %d MB\n", _rpcmem_capacity);

            if (0 != init_htp_perfinfra()) {
                QNN_LOG_WARN("initialize HTP performance failure");
            }
            if (0 != set_rpc_polling()) {
                QNN_LOG_WARN("set RPC polling failure");
            }
            if (0 != set_high_performance_mode()) {
                QNN_LOG_WARN("set HTP high performance mode failure");
            }
        }

        QNN_LOG_DEBUG("leave qni_init\n");

        return 0;
    }

    int qnn_finalize() {
        int ret_status = 0;
        Qnn_ErrorHandle_t error = QNN_SUCCESS;

        if (nullptr != _pfn_rpc_mem_deinit) // make Qualcomm's SoC equipped low-end phone happy
            _pfn_rpc_mem_deinit();

        if (dlclose(_rpc_lib_handle) != 0) {
            QNN_LOG_WARN("failed to unload qualcomm's rpc lib, error:%s\n", dlerror());
        } else {
            QNN_LOG_DEBUG("succeed to close rpcmem lib\n");
        }

        if (_backend_name.find("Htp") != std::variant_npos) {
            _qnn_htp_perfinfra->destroyPowerConfigId(_qnn_power_configid);
        }

        if (nullptr != _qnn_context_handle) {
            error = _qnn_interface.qnn_context_free(_qnn_context_handle, _qnn_profile_handle);
            if (error != QNN_SUCCESS) {
                QNN_LOG_WARN("failed to free QNN context_handle: ID %u, error %d\n", _qnn_interface.get_backend_id(),
                             QNN_GET_ERROR_CODE(error));
            }
            _qnn_context_handle = nullptr;
        }

        if (nullptr != _qnn_profile_handle) {
            error = _qnn_interface.qnn_profile_free(_qnn_profile_handle);
            if (error != QNN_SUCCESS) {
                QNN_LOG_WARN("failed to free QNN profile_handle: ID %u, error %d\n", _qnn_interface.get_backend_id(),
                             QNN_GET_ERROR_CODE(error));
            }
            _qnn_profile_handle = nullptr;
        }

        if (nullptr != _qnn_device_handle) {
            error = _qnn_interface.qnn_device_free(_qnn_device_handle);
            if (error != QNN_SUCCESS) {
                QNN_LOG_WARN("failed to free QNN device_handle: ID %u, error %d\n", _qnn_interface.get_backend_id(),
                             QNN_GET_ERROR_CODE(error));
            }
            _qnn_device_handle = nullptr;
        }

        if (nullptr != _qnn_backend_handle) {
            error = _qnn_interface.qnn_backend_free(_qnn_backend_handle);
            if (error != QNN_SUCCESS) {
                QNN_LOG_WARN("failed to free QNN backend_handle: ID %u, error %d\n", _qnn_interface.get_backend_id(),
                             QNN_GET_ERROR_CODE(error));
            }
            _qnn_backend_handle = nullptr;
        }

        if (nullptr != _qnn_log_handle) {
            error = _qnn_interface.qnn_log_free(_qnn_log_handle);
            if (error != QNN_SUCCESS) {
                QNN_LOG_WARN("failed to free QNN log_handle: ID %u, error %d\n", _qnn_interface.get_backend_id(),
                             QNN_GET_ERROR_CODE(error));
            }
            _qnn_log_handle = nullptr;
        }

        unload_backend();

        unload_system();

        return ret_status;
    }

    const qnn_interface &get_qnn_interface() {
        if (!_qnn_interface.is_loaded()) {
            QNN_LOG_WARN("pls check why _qnn_interface is not loaded\n");
        }
        return _qnn_interface;
    }

    const QNN_INTERFACE_VER_TYPE &get_qnn_raw_interface() {
        if (!_qnn_interface.is_loaded()) {
            QNN_LOG_WARN("pls check why _qnn_interface is not loaded\n");
        }
        return _qnn_raw_interface;
    }

    const QNN_SYSTEM_INTERFACE_VER_TYPE &get_qnn_raw_system_interface() {
        if (!_qnn_interface.is_loaded()) {
            QNN_LOG_WARN("pls check why _qnn_interface is not loaded\n");
        }
        return _qnn_raw_system_interface;
    }

    const Qnn_LogHandle_t get_qnn_log_handle() { return _qnn_log_handle; }

    const Qnn_ProfileHandle_t get_qnn_profile_handle() { return _qnn_profile_handle; }

    const Qnn_DeviceHandle_t get_qnn_device_handle() { return _qnn_device_handle; }

    const Qnn_BackendHandle_t get_qnn_backend_handle() { return _qnn_backend_handle; }

    const Qnn_ContextHandle_t get_qnn_context_handle() { return _qnn_context_handle; }

    const QnnSystemContext_Handle_t get_qnn_system_handle() { return _qnn_system_handle; }

    const Qnn_GraphHandle_t get_qnn_graph_handle() { return _qnn_graph_handle; }

    int init_htp_perfinfra() {
        QnnDevice_Infrastructure_t device_infra = nullptr;
        int error = _qnn_raw_interface.deviceGetInfrastructure(&device_infra);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to get qnn device infra\n");
            return 1;
        } else {
            QNN_LOG_INFO("HTP backend perf_infrastructure creation ok\n");
        }

        QnnHtpDevice_Infrastructure_t *htp_infra = static_cast<QnnHtpDevice_Infrastructure_t *>(device_infra);
        QnnHtpDevice_PerfInfrastructure_t *htp_perfinfra = &htp_infra->perfInfra;
        uint32_t power_configid = 1;
        uint32_t device_id = 0;
        uint32_t core_id = 0;
        htp_perfinfra->createPowerConfigId(device_id, core_id, &power_configid);
        if (htp_infra->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
            QNN_LOG_INFO("HTP infra type = %d, which is not perf infra type", htp_infra->infraType);
        } else {
            QNN_LOG_INFO("HTP infra type = %d, which is perf infra type\n", htp_infra->infraType);
        }
        _qnn_htp_perfinfra = htp_perfinfra;
        _qnn_power_configid = power_configid;

        return 0;
    }

    int set_rpc_polling() {
        if (_qnn_htp_perfinfra) {
            QnnHtpPerfInfrastructure_PowerConfig_t rpc_polling_time;
            memset(&rpc_polling_time, 0, sizeof(rpc_polling_time));
            rpc_polling_time.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
            // use rpc polling time recommended 0-10000 us
            rpc_polling_time.rpcPollingTimeConfig = 9999;

            QnnHtpPerfInfrastructure_PowerConfig_t rpc_control_latency;
            memset(&rpc_control_latency, 0, sizeof(rpc_control_latency));
            rpc_control_latency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
            // use rpc control latency recommended 100 us, refer hexagon sdk
            rpc_control_latency.rpcControlLatencyConfig = 100;

            const QnnHtpPerfInfrastructure_PowerConfig_t *power_configs[] = { &rpc_polling_time, &rpc_control_latency,
                                                                              nullptr };
            Qnn_ErrorHandle_t qnn_status = _qnn_htp_perfinfra->setPowerConfig(_qnn_power_configid, power_configs);
            if (qnn_status != QNN_SUCCESS) {
                QNN_LOG_WARN("set htp perf failed\n");
            } else {
                QNN_LOG_INFO("set htp perf ok\n");
            }
        } else {
            QNN_LOG_WARN("can't set htp perf\n");
        }

        return 0;
    }

    int set_high_performance_mode() {
        if (nullptr == _qnn_htp_perfinfra) {
            QNN_LOG_WARN("perf intra is null\n");
            return 1;
        }

        QnnHtpPerfInfrastructure_PowerConfig_t power_config;
        memset(&power_config, 0, sizeof(power_config));
        power_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;

        power_config.dcvsV3Config.setDcvsEnable = 1;
        power_config.dcvsV3Config.dcvsEnable = 0;
        power_config.dcvsV3Config.contextId = _qnn_power_configid;
        power_config.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        power_config.dcvsV3Config.setSleepLatency = 1; // true to consider Latency parameter otherwise false
        power_config.dcvsV3Config.sleepLatency = 40;
        power_config.dcvsV3Config.setBusParams = 1;  // true to consider Bus parameter otherwise false
        power_config.dcvsV3Config.setCoreParams = 1; // true to consider Core parameter otherwise false
        power_config.dcvsV3Config.sleepDisable = 1;  // true to consider sleep/LPM modes, false to enable
        power_config.dcvsV3Config.setSleepDisable =
            1; // true to consider sleep disable/enable parameter otherwise false set sleep latency parameter
        // set Bus Clock Parameters
        power_config.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        power_config.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        power_config.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        // set Core Clock Parameters
        power_config.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        power_config.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        power_config.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

        // set power config with different performance parameters
        const QnnHtpPerfInfrastructure_PowerConfig_t *power_configs[] = { &power_config, nullptr };
        Qnn_ErrorHandle_t qnn_status = QNN_SUCCESS;
        qnn_status = _qnn_htp_perfinfra->setPowerConfig(_qnn_power_configid, power_configs);
        if (qnn_status != QNN_SUCCESS) {
            QNN_LOG_WARN("set htp high performance mode failed\n");
        } else {
            QNN_LOG_INFO("set htp high performance mode ok\n");
        }

        return 0;
    }

    std::string &get_qnn_graph_name() { return _graph_name; }

    bool is_rpcmem_initialized() { return _rpcmem_initialized; }

    void set_rpcmem_initialized(bool initialized) { _rpcmem_initialized = initialized; }

    size_t get_rpcmem_capacity() { return _rpcmem_capacity; }

    bool is_rpcmem_registered(Qnn_MemHandle_t handle) { return _qnn_mem_set.count(handle) != 0U; }

    void *alloc_rpcmem(size_t bytes, size_t alignment) {
        if (!_rpcmem_initialized) {
            QNN_LOG_WARN("rpc memory not initialized\n");
            return nullptr;
        }

        auto allocate_bytes = static_cast<int32_t>(bytes + alignment);
        void *buf = _pfn_rpc_mem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, allocate_bytes);
        if (buf == nullptr) {
            QNN_LOG_WARN("failed to allocate rpc memory\n");
            return nullptr;
        }

        auto aligned_buf = reinterpret_cast<void *>(qnn::align_to(alignment, reinterpret_cast<intptr_t>(buf)));
        bool status = _rpcmem_store_map.insert(std::pair<void *, void *>(aligned_buf, buf)).second;
        if (!status) {
            QNN_LOG_WARN("failed to allocate rpc memory\n");
            _pfn_rpc_mem_free(buf);
        }

        return aligned_buf;
    }

    void free_rpcmem(void *buf) {
        if (!_rpcmem_initialized) {
            QNN_LOG_WARN("rpc memory not initialized\n");
        } else if (0 == _rpcmem_store_map.count(buf)) {
            QNN_LOG_WARN("no allocated tensor\n");
        } else {
            _pfn_rpc_mem_free(_rpcmem_store_map[buf]);
            _rpcmem_store_map.erase(buf);
        }
    }

    int32_t rpcmem_to_fd(void *buf) {
        int32_t mem_fd = -1;
        if (!is_rpcmem_initialized()) {
            QNN_LOG_WARN("rpc memory not initialized\n");
        } else {
            mem_fd = _pfn_rpc_mem_to_fd(buf);
        }

        return mem_fd;
    }

    int register_rpcmem(void *p_data, Qnn_Tensor_t *p_tensor) {
        if (nullptr == p_data || (nullptr == p_tensor)) {
            QNN_LOG_WARN("invalid param\n");
            return 1;
        }

        if (!is_rpcmem_initialized()) {
            QNN_LOG_WARN("rpc memory not initialized\n");
            return 2;
        }

        if (is_rpcmem_allocated(p_data)) {
            QNN_LOG_WARN("rpc memory already allocated\n");
            return 3;
        }

        if (is_rpcmem_registered(QNN_TENSOR_GET_MEM_HANDLE(*p_tensor))) {
            QNN_LOG_WARN("tensor %s has been registered shared memory\n", QNN_TENSOR_GET_NAME(*p_tensor));
            return 4;
        }

        int32_t mem_fd = rpcmem_to_fd(p_data);
        if (mem_fd == -1) {
            QNN_LOG_WARN("failed to get file descriptor\n");
            return 5;
        }
        QNN_LOG_INFO("mem_fd %d\n", mem_fd);
        Qnn_MemDescriptor_t descriptor = { { QNN_TENSOR_GET_RANK(*p_tensor), QNN_TENSOR_GET_DIMENSIONS(*p_tensor),
                                             nullptr },
                                           QNN_TENSOR_GET_DATA_TYPE(*p_tensor),
                                           QNN_MEM_TYPE_ION,
                                           { { mem_fd } } };
        Qnn_MemHandle_t handle = nullptr;
        int error = QNN_SUCCESS;
        error = _qnn_interface.qnn_mem_register(_qnn_context_handle, &descriptor,
                                                /*numDescriptors=*/1, &handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to register shared memory, error %d, %s\n", QNN_GET_ERROR_CODE(error),
                         strerror(error));
            return 6;
        }

        QNN_TENSOR_SET_MEM_HANDLE(*p_tensor, handle);
        _qnn_mem_set.insert((std::pair<void *, Qnn_MemHandle_t>(p_data, handle)));

        QNN_LOG_INFO("tensor %s successfully register shared memory handler: %p\n", QNN_TENSOR_GET_NAME(*p_tensor), handle);
        return 0;
    }

    void *get_rpcmem_from_memhandle(Qnn_MemHandle_t mem_handle) {
        for (std::unordered_map<void *, Qnn_MemHandle_t>::iterator it = _qnn_mem_set.begin(); it != _qnn_mem_set.end();
             it++) {
            Qnn_MemHandle_t mem_handle = it->second;
            if (it->second == mem_handle) {
                return it->first;
            }
        }
        QNN_LOG_WARN("can't find rpcmem from qnn mem handle %p", mem_handle);
        return nullptr;
    }

    void unregister_rpcmem() {
        Qnn_ErrorHandle_t error = QNN_SUCCESS;

        if (_qnn_mem_set.empty()) {
            QNN_LOG_WARN("no rpcmem registered\n");
        }

        for (std::unordered_map<void *, Qnn_MemHandle_t>::iterator it = _qnn_mem_set.begin(); it != _qnn_mem_set.end();
             it++) {
            Qnn_MemHandle_t mem_handle = it->second;
            error = _qnn_interface.qnn_mem_de_register(&mem_handle, 1);
            if (error != QNN_SUCCESS) {
                QNN_LOG_WARN("failed to unregister shared memory, error %d\n", QNN_GET_ERROR_CODE(error));
            }
        }
        _qnn_mem_set.clear();
    }

    bool is_rpcmem_allocated(void *buf) { return _qnn_mem_set.count(buf) != 0U; }

    const qnn::qcom_socinfo &get_soc_info() { return _soc_info; }

private:
    int load_system() {
        Qnn_ErrorHandle_t error = QNN_SUCCESS;

        std::string system_lib_path = _lib_path + "libQnnSystem.so";
        QNN_LOG_DEBUG("system_lib_path:%s\n", system_lib_path.c_str());

        _system_lib_handle = dlopen(system_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (nullptr == _system_lib_handle) {
            QNN_LOG_WARN("can not open QNN library %s, error: %s\n", system_lib_path.c_str(), dlerror());
            return 1;
        }

        auto *get_providers = reinterpret_cast<qnn::pfn_qnnsysteminterface_getproviders *>(
            dlsym(_system_lib_handle, "QnnSystemInterface_getProviders"));
        if (nullptr == get_providers) {
            QNN_LOG_WARN("can not load QNN symbol QnnSystemInterface_getProviders: %s\n", dlerror());
            return 2;
        }

        uint32_t num_providers = 0;
        const QnnSystemInterface_t **provider_list = nullptr;
        error = get_providers(&provider_list, &num_providers);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to get providers, error %d\n", QNN_GET_ERROR_CODE(error));
            return 3;
        }

        if (num_providers != _required_num_providers) {
            QNN_LOG_WARN("providers is %d instead of required %d\n", num_providers, _required_num_providers);
            return 4;
        }

        if (nullptr == provider_list) {
            QNN_LOG_WARN("can not get providers\n");
            return 5;
        }

        QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface;
        bool found_valid_system_interface = false;
        for (size_t idx = 0; idx < num_providers; idx++) {
            if (QNN_SYSTEM_API_VERSION_MAJOR == provider_list[idx]->systemApiVersion.major &&
                QNN_SYSTEM_API_VERSION_MINOR <= provider_list[idx]->systemApiVersion.minor) {
                found_valid_system_interface = true;
                qnn_system_interface = provider_list[idx]->QNN_SYSTEM_INTERFACE_VER_NAME;
                break;
            }
        }
        if (!found_valid_system_interface) {
            QNN_LOG_WARN("unable to find a valid qnn system interface\n");
            return 6;
        } else {
            QNN_LOG_INFO("find a valid qnn system interface\n");
        }
        set_qnn_raw_system_interface(qnn_system_interface);

        _qnn_interface.set_qnn_system_interface(provider_list[0]);

        _qnn_interface.qnn_system_context_create(&_qnn_system_handle);
        if (nullptr == _qnn_system_handle) {
            QNN_LOG_WARN("can not create QNN system contenxt\n");
        } else {
            QNN_LOG_INFO("initialize qnn system successfully\n");
        }

        return 0;
    }

    int unload_system() {
        int result = 0;

        if (nullptr == _system_lib_handle) {
            QNN_LOG_WARN("system lib handle is null\n");
            return 1;
        }

        if (nullptr != _qnn_system_handle) {
            result = _qnn_interface.qnn_system_context_free(_qnn_system_handle);
            if (result != QNN_SUCCESS) {
                QNN_LOG_WARN("failed to free QNN system context\n");
            }
            _qnn_system_handle = nullptr;
        }

        int dlclose_error = dlclose(_system_lib_handle);
        if (dlclose_error != 0) {
            QNN_LOG_WARN("failed to close QnnSystem library, error %s\n", dlerror());
            return 2;
        }

        _system_lib_handle = nullptr;

        return result;
    }

    int load_backend(std::string &lib_path, const QnnSaver_Config_t **saver_config) {
        Qnn_ErrorHandle_t error = QNN_SUCCESS;
        QNN_LOG_DEBUG("lib_path:%s\n", lib_path.c_str());

        void *lib_handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (nullptr == lib_handle) {
            QNN_LOG_WARN("can not open QNN library %s, with error: %s", lib_path.c_str(), dlerror());
            return 1;
        }

        auto get_providers = qnn::load_qnn_functionpointers<qnn::pfn_qnninterface_getproviders *>(
            lib_handle, "QnnInterface_getProviders");
        if (nullptr == get_providers) {
            QNN_LOG_WARN("can not load symbol QnnInterface_getProviders : %s", dlerror());
            return 2;
        }

        std::uint32_t num_providers = 0;
        const QnnInterface_t **provider_list = nullptr;
        error = get_providers(&provider_list, &num_providers);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to get providers, error %d", QNN_GET_ERROR_CODE(error));
            return 3;
        }
        QNN_LOG_DEBUG("num_providers=%d\n", num_providers);
        if (num_providers != _required_num_providers) {
            QNN_LOG_WARN("providers is %d instead of required %d", num_providers, _required_num_providers);
            return 4;
        }

        if (nullptr == provider_list) {
            QNN_LOG_WARN("failed to get qnn interface providers\n");
            return 5;
        }
        bool found_valid_interface = false;
        QNN_INTERFACE_VER_TYPE qnn_interface;
        for (size_t idx = 0; idx < num_providers; idx++) {
            if (QNN_API_VERSION_MAJOR == provider_list[idx]->apiVersion.coreApiVersion.major &&
                QNN_API_VERSION_MINOR <= provider_list[idx]->apiVersion.coreApiVersion.minor) {
                found_valid_interface = true;
                qnn_interface = provider_list[idx]->QNN_INTERFACE_VER_NAME;
                break;
            }
        }

        if (!found_valid_interface) {
            QNN_LOG_WARN("unable to find a valid qnn interface\n");
            return 6;
        } else {
            QNN_LOG_INFO("find a valid qnn interface\n");
        }
        set_qnn_raw_interface(qnn_interface);

        BackendIdType backend_id = provider_list[0]->backendId;
        _lib_path_to_backend_id[lib_path] = backend_id;
        if (_loaded_backend.count(backend_id) > 0) {
            QNN_LOG_WARN("lib_path %s is loaded, but backend %d already exists\n", lib_path.c_str(), backend_id);
        }
        _loaded_backend[backend_id] = provider_list[0];
        if (_loaded_lib_handle.count(backend_id) > 0) {
            QNN_LOG_WARN("closing %p\n", _loaded_lib_handle[backend_id]);
            int dlclose_error = dlclose(_loaded_lib_handle[backend_id]);
            if (dlclose_error != 0) {
                QNN_LOG_WARN("fail to close %p with error %s\n", _loaded_lib_handle[backend_id], dlerror());
            }
        }
        _loaded_lib_handle[backend_id] = lib_handle;
        _backend_id = backend_id;

        return 0;
    }

    int unload_backend() {
        int dlclose_error = 0;
        for (auto &it : _loaded_lib_handle) {
            dlclose_error = dlclose(it.second);
            if (dlclose_error != 0) {
                QNN_LOG_WARN("failed to close QNN backend %d, error %s\n", it.first, dlerror());
            }
        }

        _loaded_lib_handle.clear();
        _lib_path_to_backend_id.clear();
        _loaded_backend.clear();

        return 0;
    }

    void set_qnn_raw_interface(QNN_INTERFACE_VER_TYPE &raw_interface) { _qnn_raw_interface = raw_interface; }

    void set_qnn_raw_system_interface(QNN_SYSTEM_INTERFACE_VER_TYPE &raw_interface) {
        _qnn_raw_system_interface = raw_interface;
    }

private:
    static constexpr const int _required_num_providers = 1;

    std::string _lib_path;
    std::string _backend_name;
    std::string _model_name; // Qualcomm's dedicated prebuilt model name, keep it for further usage
    BackendIdType _backend_id;

    bool _debug_tensor = false;
    bool _do_node_validations = true;

    QnnLog_Level_t _qnn_log_level = QNN_LOG_LEVEL_DEBUG;

    qnn::sdk_profile_level _profile_level = qnn::sdk_profile_level::profile_detail;

    qnn_interface _qnn_interface;

    void *_system_lib_handle = nullptr;

    Qnn_GraphHandle_t _qnn_graph_handle = nullptr;

    Qnn_LogHandle_t _qnn_log_handle = nullptr;

    Qnn_ProfileHandle_t _qnn_profile_handle = nullptr;

    Qnn_DeviceHandle_t _qnn_device_handle = nullptr;

    Qnn_BackendHandle_t _qnn_backend_handle = nullptr;

    Qnn_ContextHandle_t _qnn_context_handle = nullptr;

    QnnSystemContext_Handle_t _qnn_system_handle = nullptr;

    QnnHtpDevice_PerfInfrastructure_t *_qnn_htp_perfinfra = nullptr;
    uint32_t _qnn_power_configid = 1;

    QNN_INTERFACE_VER_TYPE _qnn_raw_interface;
    QNN_SYSTEM_INTERFACE_VER_TYPE _qnn_raw_system_interface;

    std::unordered_map<void *, Qnn_MemHandle_t> _qnn_mem_set;

    std::mutex _init_mutex;
    std::unordered_map<BackendIdType, void *> _loaded_lib_handle;
    std::unordered_map<std::string, BackendIdType> _lib_path_to_backend_id;
    std::unordered_map<BackendIdType, const QnnInterface_t *> _loaded_backend;

    void *_rpc_lib_handle = nullptr;
    std::atomic_bool _rpcmem_initialized{ false };
    qnn::pfn_rpc_mem_alloc _pfn_rpc_mem_alloc;
    qnn::pfn_rpc_mem_free _pfn_rpc_mem_free;
    qnn::pfn_rpc_mem_to_fd _pfn_rpc_mem_to_fd;
    qnn::pfn_rpc_mem_init _pfn_rpc_mem_init;
    qnn::pfn_rpc_mem_deinit _pfn_rpc_mem_deinit;
    std::unordered_map<void *, void *> _rpcmem_store_map;
    size_t _rpcmem_capacity = 512;

    std::string _graph_name;

    qnn::qcom_socinfo _soc_info = {};
};

} // namespace qnn
